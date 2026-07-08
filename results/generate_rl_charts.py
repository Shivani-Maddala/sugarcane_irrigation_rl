"""
results/generate_rl_charts.py

Generates publication-quality charts for the sugarcane DQN irrigation
model (rl/environment.py, rl/agent.py, rl/weather_provider.py, rl/config.py,
rl/evaluate.py -- all imported as-is, NOT modified).

Run from the project root:
    python -m results.generate_rl_charts

What this script does NOT do: it does not call rl/train.py directly, because
that function only logs {episode_reward, episode_irrigation_mm,
episode_drainage_mm, epsilon} -- not the per-step TD loss or the decomposed
reward terms (r0..r3) that several charts below need. Rather than editing
rl/train.py, `train_instrumented()` below re-implements the *same* training
loop (same agent.act/remember/train_step/decay_epsilon calls, same
episode structure) with additional logging. It reads the identical
rl/config.py hyperparameters, so results are equivalent to running
rl/train.py -- just with richer instrumentation. It writes its checkpoint to
the same results/rl_checkpoints/dqn_model.pt path rl/train.py uses, so
rl/evaluate.load_trained_agent() keeps working unchanged afterwards.

Charts produced (see each function's docstring for what/why):
  1. training_curves            -- TD loss & episode reward vs. training episode
  2. decomposed_rewards         -- r0..r3 components vs. training episode
  3. epsilon_decay              -- exploration-rate schedule
  4. policy_comparison          -- DQN vs. conventional baseline (irrigation,
                                    drainage, yield proxy), reusing
                                    rl.evaluate.compare_policies
  5. q_value_landscape          -- Q(s,a) vs. water depth & forecast rainfall
  6. action_distribution        -- how often each action is chosen, overall
                                    and when water is below h_min
  7. season_case_study          -- one season's water depth / rainfall /
                                    irrigation / action timeline
  8. rainfall_forecast_quality  -- TS / MAR / FAR of the synthetic weather
                                    provider's forecast noise model, by lead
                                    day and rainfall grade
  9. action_by_crop_condition   -- action choice vs. CNN-style crop-condition
                                    signal (the r3 stress term this project
                                    adds beyond the reference paper), using
                                    integration.pipeline.MockCropConditionProvider
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl import config
from rl.environment import SugarcaneIrrigationEnv
from rl.agent import DQNAgent
from rl.weather_provider import SyntheticWeatherProvider
from rl.evaluate import compare_policies, conventional_baseline_policy, load_trained_agent
from integration.pipeline import MockCropConditionProvider

from results.plot_utils import savefig, save_table, COLORS

RAIN_GRADES = ["NR", "LR", "MR", "HR", "ST"]  # no/light/moderate/heavy/storm rain
RAIN_THRESHOLDS_MM = [0.1, 10, 25, 50]        # upper edge of NR, LR, MR, HR (ST = above 50)


def _rain_grade(mm):
    if mm < RAIN_THRESHOLDS_MM[0]:
        return "NR"
    if mm < RAIN_THRESHOLDS_MM[1]:
        return "LR"
    if mm < RAIN_THRESHOLDS_MM[2]:
        return "MR"
    if mm < RAIN_THRESHOLDS_MM[3]:
        return "HR"
    return "ST"


# --------------------------------------------------------------------------
# Instrumented training run (see module docstring for why this exists
# instead of calling rl.train.train() directly)
# --------------------------------------------------------------------------

def train_instrumented(num_episodes=None, seed_base=0, verbose=True):
    """Same loop as rl.train.train(), same rl.agent.DQNAgent /
    rl.environment.SugarcaneIrrigationEnv / rl.weather_provider.SyntheticWeatherProvider,
    just with extra per-episode logging (loss, decomposed rewards, action
    counts) that the reference-style diagnostic charts need."""
    num_episodes = num_episodes or config.NUM_EPISODES
    agent = DQNAgent()

    history = {
        "episode": [], "episode_reward": [], "episode_loss": [],
        "episode_irrigation_mm": [], "episode_drainage_mm": [], "epsilon": [],
        "r0_mean": [], "r1_mean": [], "r2_mean": [], "r3_mean": [],
        "action_0_count": [], "action_1_count": [], "action_2_count": [],
    }

    for episode in range(num_episodes):
        weather = SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=seed_base + episode)
        env = SugarcaneIrrigationEnv(weather)
        state, _ = env.reset()

        total_reward, losses = 0.0, []
        r0s, r1s, r2s, r3s = [], [], [], []
        action_counts = [0, 0, 0]
        done = False
        while not done:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            r0s.append(info["r0"]); r1s.append(info["r1"])
            r2s.append(info["r2"]); r3s.append(info["r3"])
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        history["episode"].append(episode)
        history["episode_reward"].append(total_reward)
        history["episode_loss"].append(float(np.mean(losses)) if losses else np.nan)
        history["episode_irrigation_mm"].append(env.cumulative_irrigation_mm)
        history["episode_drainage_mm"].append(env.cumulative_drainage_mm)
        history["epsilon"].append(agent.epsilon)
        history["r0_mean"].append(float(np.mean(r0s)))
        history["r1_mean"].append(float(np.mean(r1s)))
        history["r2_mean"].append(float(np.mean(r2s)))
        history["r3_mean"].append(float(np.mean(r3s)))
        history["action_0_count"].append(action_counts[0])
        history["action_1_count"].append(action_counts[1])
        history["action_2_count"].append(action_counts[2])

        if verbose and (episode + 1) % max(num_episodes // 10, 1) == 0:
            print(f"  episode {episode+1}/{num_episodes} | reward={total_reward:.2f} | "
                  f"loss={history['episode_loss'][-1]:.4f} | epsilon={agent.epsilon:.3f}")

    checkpoint_dir = "results/rl_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    import torch
    torch.save(agent.q_network.state_dict(), os.path.join(checkpoint_dir, "dqn_model.pt"))
    with open(os.path.join(checkpoint_dir, "history_instrumented.json"), "w") as f:
        json.dump(history, f, indent=2)

    return agent, pd.DataFrame(history)


# --------------------------------------------------------------------------
# Chart 1: training loss & reward
# --------------------------------------------------------------------------

def chart_training_curves(history_df):
    """Two-panel figure: (left) per-episode mean TD loss, log-scaled since
    loss typically drops by 2-3 orders of magnitude early in training and a
    linear axis would flatten everything after episode ~20; (right)
    per-episode total reward with a rolling mean overlay to show the
    convergence trend through the episode-to-episode noise."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history_df["episode"], history_df["episode_loss"], color=COLORS["dqn"], linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training episode"); axes[0].set_ylabel("Mean TD loss (log scale)")
    axes[0].set_title("DQN training loss")

    reward = history_df["episode_reward"]
    rolling = reward.rolling(window=max(len(reward) // 20, 1), min_periods=1).mean()
    axes[1].plot(history_df["episode"], reward, color=COLORS["dqn"], alpha=0.3, linewidth=1, label="Episode reward")
    axes[1].plot(history_df["episode"], rolling, color=COLORS["dqn"], linewidth=2, label="Rolling mean")
    axes[1].set_xlabel("Training episode"); axes[1].set_ylabel("Total episode reward")
    axes[1].set_title("DQN training reward")
    axes[1].legend()

    fig.suptitle("DQN irrigation agent: training diagnostics")
    fig.tight_layout()
    savefig(fig, "rl_01_training_curves")
    save_table(history_df[["episode", "episode_loss", "episode_reward"]], "rl_01_training_curves")


# --------------------------------------------------------------------------
# Chart 2: decomposed reward terms
# --------------------------------------------------------------------------

def chart_decomposed_rewards(history_df):
    """r_t = r0 * r1 * r2 * r3 (rl/environment.py step()). Plotting each
    factor separately over training shows *which* incentive the agent is
    optimizing at any point: r0 = conformity to the irrigate-below-h_min
    baseline, r1 = rainfall-utilization (avoiding drainage waste), r2 =
    yield term from the crop-water-production function, r3 = the
    crop-condition stress penalty (this project's addition beyond the
    reference paper)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    terms = [("r0_mean", "r0: irrigation-baseline conformity"),
             ("r1_mean", "r1: rainfall-utilization reward"),
             ("r2_mean", "r2: yield reward (crop-water-production fn.)"),
             ("r3_mean", "r3: crop-condition stress penalty")]
    for ax, (col, title) in zip(axes.flat, terms):
        ax.plot(history_df["episode"], history_df[col], color=COLORS["dqn"], linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training episode")
    fig.suptitle("Decomposed reward terms over training (episode means)")
    fig.tight_layout()
    savefig(fig, "rl_02_decomposed_rewards")
    save_table(history_df[["episode", "r0_mean", "r1_mean", "r2_mean", "r3_mean"]], "rl_02_decomposed_rewards")


# --------------------------------------------------------------------------
# Chart 3: epsilon decay
# --------------------------------------------------------------------------

def chart_epsilon_decay(history_df):
    """The exploration-rate schedule (config.EPSILON_START -> EPSILON_MIN,
    decayed by config.EPSILON_DECAY per episode). Useful alongside chart 1
    to see whether the reward curve's improvement lines up with epsilon
    dropping (exploitation kicking in) rather than being a training
    instability artifact."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history_df["episode"], history_df["epsilon"], color="#6a51a3", linewidth=1.5)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Epsilon (exploration rate)")
    ax.set_title(f"Epsilon-greedy schedule (decay={config.EPSILON_DECAY}, min={config.EPSILON_MIN})")
    fig.tight_layout()
    savefig(fig, "rl_03_epsilon_decay")
    save_table(history_df[["episode", "epsilon"]], "rl_03_epsilon_decay")


# --------------------------------------------------------------------------
# Chart 4: DQN vs. conventional baseline
# --------------------------------------------------------------------------

def chart_policy_comparison(agent, num_seasons=20, stress_test=True):
    """Reuses rl.evaluate.compare_policies (unmodified) to run the trained
    DQN and the conventional (irrigate-when-h<h_min) baseline on the same
    `num_seasons` weather realizations, then plots irrigation water,
    drainage, and the yield proxy (actual/max ET) side by side as grouped
    bars with season-to-season std-dev error bars.

    stress_test=True (default): also runs the comparison a second time
    using a locally-defined drier weather variant (see
    _DrierWeatherProvider below), because with config.py's default
    SyntheticWeatherProvider rainfall is frequent/heavy enough that the
    conventional baseline may rarely-to-never cross h_min (you'll see this
    directly in rl.evaluate.compare_policies's own output: it returns
    water_savings_pct=None whenever baseline_irrigation_mm < 1mm, which is
    its built-in guard for exactly this degenerate case). Comparing under
    both regimes -- rather than silently only showing the flattering one --
    is the honest way to report this. Nothing in rl/weather_provider.py is
    modified: _DrierWeatherProvider is a small subclass defined only in
    this script.
    """

    def default_factory():
        return SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=np.random.randint(0, 1_000_000))

    scenarios = {"default_weather": default_factory}
    if stress_test:
        class _DrierWeatherProvider(SyntheticWeatherProvider):
            """Lower rain probability/amount than the default synthetic
            provider, so the field actually dries below h_min sometimes --
            needed to get a non-degenerate DQN-vs-baseline water comparison.
            Subclasses (rather than edits) rl.weather_provider.SyntheticWeatherProvider."""
            def __init__(self, num_days, seed=0):
                super().__init__(num_days, seed=seed)
                rng = np.random.default_rng(seed + 999_983)  # different stream than the parent's
                day_idx = np.arange(num_days)
                rain_prob = 0.06 + 0.14 * np.sin(np.pi * day_idx / num_days) ** 2
                rains = rng.random(num_days) < rain_prob
                self.rainfall = np.where(rains, rng.gamma(shape=2.0, scale=4.0, size=num_days), 0.0)

        scenarios["stress_test_weather"] = lambda: _DrierWeatherProvider(
            config.SEASON_LENGTH_DAYS, seed=np.random.randint(0, 1_000_000))

    rows = []
    for scenario_name, factory in scenarios.items():
        summary = compare_policies(agent, factory, num_seasons=num_seasons)
        print(f"  [{scenario_name}] {summary}")
        rows.append({"scenario": scenario_name, "policy": "DQN",
                      "irrigation_mm": summary["dqn_irrigation_mm"],
                      "drainage_mm": summary["dqn_drainage_mm"],
                      "yield_proxy": summary["yield_proxy_dqn"]})
        rows.append({"scenario": scenario_name, "policy": "Conventional",
                      "irrigation_mm": summary["baseline_irrigation_mm"],
                      "drainage_mm": summary["baseline_drainage_mm"],
                      "yield_proxy": summary["yield_proxy_baseline"]})
    df = pd.DataFrame(rows)

    metrics = [("irrigation_mm", "Seasonal irrigation water (mm)"),
               ("drainage_mm", "Seasonal drainage (mm)"),
               ("yield_proxy", "Yield proxy (actual ET / max ET)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    scenario_names = list(scenarios.keys())
    x = np.arange(len(scenario_names))
    width = 0.35
    for ax, (col, title) in zip(axes, metrics):
        dqn_vals = [df[(df.scenario == s) & (df.policy == "DQN")][col].iloc[0] for s in scenario_names]
        base_vals = [df[(df.scenario == s) & (df.policy == "Conventional")][col].iloc[0] for s in scenario_names]
        ax.bar(x - width / 2, dqn_vals, width, label="DQN", color=COLORS["dqn"])
        ax.bar(x + width / 2, base_vals, width, label="Conventional", color=COLORS["conventional"])
        ax.set_xticks(x); ax.set_xticklabels(scenario_names, rotation=10)
        ax.set_title(title, fontsize=9)
    axes[0].legend()
    fig.suptitle(f"DQN vs. conventional irrigation policy (mean over {num_seasons} seasons/scenario)")
    fig.tight_layout()
    savefig(fig, "rl_04_policy_comparison")
    save_table(df, "rl_04_policy_comparison")


# --------------------------------------------------------------------------
# Chart 5: Q-value landscape
# --------------------------------------------------------------------------

def chart_q_value_landscape(agent, n_grid=50):
    """Q(s, a) for each of the 3 actions, evaluated on a grid of
    (water depth, total forecast rainfall) with h_min/h_max/H_p held at
    their config.py values and crop_condition fixed at 0 (healthy). Shows
    what the trained network learned about *when* each action looks best --
    e.g. whether action 2 (100% quota) only dominates once water depth is
    near h_min AND forecast rain is low, as the reward design intends."""
    import torch
    h_values = np.linspace(0, config.H_MAX_MM, n_grid)
    precip_values = np.linspace(0, 2 * config.H_MAX_MM, n_grid)
    Q = np.zeros((n_grid, n_grid, config.NUM_ACTIONS))

    with torch.no_grad():
        for pi, precip in enumerate(precip_values):
            forecast = np.full(config.FORECAST_HORIZON_DAYS, precip / config.FORECAST_HORIZON_DAYS)
            for hi, h in enumerate(h_values):
                state = np.concatenate([
                    forecast, [h, config.H_MIN_MM, config.H_MAX_MM, config.H_P_MM, 0],
                ]).astype(np.float32)
                state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                Q[pi, hi, :] = agent.q_network(state_t).cpu().numpy()[0]

    fig, axes = plt.subplots(1, config.NUM_ACTIONS, figsize=(4.5 * config.NUM_ACTIONS, 4), sharey=True)
    vmin, vmax = Q.min(), Q.max()
    action_labels = [f"{int(a*100)}% quota" for a in config.ACTIONS]
    for a in range(config.NUM_ACTIONS):
        im = axes[a].imshow(Q[:, :, a], origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                             extent=[h_values.min(), h_values.max(), precip_values.min(), precip_values.max()])
        axes[a].axvline(config.H_MIN_MM, color="white", linestyle="--", linewidth=1)
        axes[a].set_title(f"Action {a} ({action_labels[a]})", fontsize=9)
        axes[a].set_xlabel("Water depth h_t (mm)")
    axes[0].set_ylabel("Forecast total rainfall, next 7 days (mm)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Q(s,a)")
    fig.suptitle("Learned Q-value landscape (crop condition = healthy)")
    savefig(fig, "rl_05_q_value_landscape")

    flat = pd.DataFrame({
        "water_depth_mm": np.repeat(h_values, n_grid),
        "forecast_precip_mm": np.tile(precip_values, n_grid),
        **{f"Q_action{a}": Q[:, :, a].T.flatten() for a in range(config.NUM_ACTIONS)},
    })
    save_table(flat.iloc[::5, :], "rl_05_q_value_landscape")  # subsample rows, full grid is large


# --------------------------------------------------------------------------
# Chart 6: action distribution
# --------------------------------------------------------------------------

def chart_action_distribution(agent, num_seasons=20):
    """How often the trained (greedy) policy chooses each action overall,
    vs. specifically on days where h_t < h_min (i.e. days the conventional
    baseline would irrigate). If the DQN has learned something beyond the
    baseline rule, its action split conditioned on h<h_min should differ
    from "always action 2", since it also has forecast rainfall to react to."""
    counts_all = np.zeros(config.NUM_ACTIONS)
    counts_below_min = np.zeros(config.NUM_ACTIONS)

    for seed in range(num_seasons):
        weather = SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=10_000 + seed)
        env = SugarcaneIrrigationEnv(weather)
        state, _ = env.reset()
        done = False
        while not done:
            h_t = state[config.FORECAST_HORIZON_DAYS]
            action = agent.act(state, greedy=True)
            counts_all[action] += 1
            if h_t < config.H_MIN_MM:
                counts_below_min[action] += 1
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    df = pd.DataFrame({
        "action": [f"A{a} ({int(config.ACTIONS[a]*100)}%)" for a in range(config.NUM_ACTIONS)],
        "count_all_days": counts_all,
        "count_when_below_h_min": counts_below_min,
    })
    df["pct_all_days"] = 100 * df["count_all_days"] / df["count_all_days"].sum()
    df["pct_when_below_h_min"] = 100 * df["count_when_below_h_min"] / max(df["count_when_below_h_min"].sum(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bar_colors = [COLORS["action_0"], COLORS["action_1"], COLORS["action_2"]]
    axes[0].bar(df["action"], df["pct_all_days"], color=bar_colors)
    axes[0].set_ylabel("% of all days"); axes[0].set_title("Action frequency, all days")
    axes[1].bar(df["action"], df["pct_when_below_h_min"], color=bar_colors)
    axes[1].set_ylabel("% of days with h_t < h_min"); axes[1].set_title("Action frequency, water below h_min")
    fig.suptitle(f"Greedy-policy action distribution ({num_seasons} seasons)")
    fig.tight_layout()
    savefig(fig, "rl_06_action_distribution")
    save_table(df, "rl_06_action_distribution")


# --------------------------------------------------------------------------
# Chart 7: single-season case study
# --------------------------------------------------------------------------

def chart_season_case_study(agent, seed=42):
    """One concrete season, day by day: water depth (with h_min/h_max
    bands), rainfall vs. irrigation applied, and the action taken. This is
    the "does the learned policy behave sensibly" sanity check that
    aggregate statistics (charts 4/6) can hide -- e.g. does the agent ever
    let h_t crash to 0, does it irrigate right before a big forecast rain
    event, etc."""
    weather = SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=seed)
    env = SugarcaneIrrigationEnv(weather)
    state, _ = env.reset()

    t_list, h_list, rain_list, irr_list, action_list = [], [], [], [], []
    done, t = False, 0
    while not done:
        action = agent.act(state, greedy=True)
        t_list.append(t); h_list.append(env.h)
        rain_list.append(weather.get_rainfall(t)); action_list.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        irr_list.append(info["irrigation_mm"])
        done = terminated or truncated
        t += 1

    df = pd.DataFrame({"day": t_list, "water_depth_mm": h_list, "rainfall_mm": rain_list,
                        "irrigation_mm": irr_list, "action": action_list})

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.6]})
    axes[0].plot(df["day"], df["water_depth_mm"], color=COLORS["dqn"], linewidth=1.2)
    axes[0].axhline(config.H_MIN_MM, color=COLORS["conventional"], linestyle="--", linewidth=1, label="h_min")
    axes[0].axhline(config.H_MAX_MM, color="green", linestyle="--", linewidth=1, label="h_max")
    axes[0].set_ylabel("Water depth (mm)"); axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Season case study (seed={seed}): water depth")

    axes[1].bar(df["day"], df["rainfall_mm"], color="#74c0fc", width=1.0, label="Rainfall")
    axes[1].bar(df["day"], df["irrigation_mm"], bottom=df["rainfall_mm"], color="#f76707", width=1.0, label="Irrigation")
    axes[1].set_ylabel("mm/day"); axes[1].legend(loc="upper right", fontsize=8)

    axes[2].step(df["day"], df["action"], where="post", color="#5f3dc4")
    axes[2].set_yticks(range(config.NUM_ACTIONS))
    axes[2].set_yticklabels([f"A{a}" for a in range(config.NUM_ACTIONS)])
    axes[2].set_xlabel("Day of season"); axes[2].set_ylabel("Action")

    fig.tight_layout()
    savefig(fig, "rl_07_season_case_study")
    save_table(df, "rl_07_season_case_study")


# --------------------------------------------------------------------------
# Chart 8: rainfall forecast quality (data-level, not model-level)
# --------------------------------------------------------------------------

def chart_rainfall_forecast_quality(num_seasons=30):
    """Threat score / missing-alarm-rate / false-alarm-rate of
    SyntheticWeatherProvider.get_forecast()'s noise model (uniform 0.6-1.3x
    multiplicative noise on the true future rainfall -- see
    rl/weather_provider.py), by lead day (1-7) and rainfall grade (NR/LR/MR/
    HR/ST). This characterizes the *data* the DQN has to work with (same
    purpose as the reference paper's Fig. 3), independent of any trained
    model."""
    horizon = config.FORECAST_HORIZON_DAYS
    hits = np.zeros((horizon, len(RAIN_GRADES)))
    misses = np.zeros((horizon, len(RAIN_GRADES)))
    false_alarms = np.zeros((horizon, len(RAIN_GRADES)))

    for seed in range(num_seasons):
        weather = SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=seed)
        for t in range(config.SEASON_LENGTH_DAYS - horizon):
            forecast = weather.get_forecast(t, horizon)
            for lead in range(horizon):
                actual_day = t + lead
                if actual_day >= config.SEASON_LENGTH_DAYS:
                    continue
                obs_grade = _rain_grade(weather.get_rainfall(actual_day))
                fc_grade = _rain_grade(max(forecast[lead], 0.0))
                for gi, grade in enumerate(RAIN_GRADES):
                    forecasted = fc_grade == grade
                    observed = obs_grade == grade
                    if forecasted and observed:
                        hits[lead, gi] += 1
                    elif forecasted and not observed:
                        false_alarms[lead, gi] += 1
                    elif observed and not forecasted:
                        misses[lead, gi] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        ts = hits / (hits + false_alarms + misses)
        mar = misses / (hits + misses)
        far = false_alarms / (hits + false_alarms)
    ts, mar, far = [np.nan_to_num(m) for m in (ts, mar, far)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, mat, title in zip(axes, [ts, mar, far], ["Threat score (TS)", "Missing alarm rate (MAR)", "False alarm rate (FAR)"]):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(RAIN_GRADES))); ax.set_xticklabels(RAIN_GRADES)
        ax.set_yticks(range(horizon)); ax.set_yticklabels(range(1, horizon + 1))
        ax.set_xlabel("Rainfall grade"); ax.set_ylabel("Lead time (day)")
        ax.set_title(title, fontsize=9)
        for lead in range(horizon):
            for gi in range(len(RAIN_GRADES)):
                ax.text(gi, lead, f"{mat[lead, gi]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[lead, gi] > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Synthetic weather provider: forecast quality by lead time & rainfall grade")
    fig.tight_layout()
    savefig(fig, "rl_08_rainfall_forecast_quality")

    rows = []
    for lead in range(horizon):
        for gi, grade in enumerate(RAIN_GRADES):
            rows.append({"lead_day": lead + 1, "grade": grade, "TS": ts[lead, gi], "MAR": mar[lead, gi], "FAR": far[lead, gi]})
    save_table(pd.DataFrame(rows), "rl_08_rainfall_forecast_quality")


# --------------------------------------------------------------------------
# Chart 9: action choice vs. CNN-style crop-condition signal
# --------------------------------------------------------------------------

def chart_action_by_crop_condition(agent, num_seasons=15):
    """Uses integration.pipeline.MockCropConditionProvider (existing,
    unmodified -- built specifically for exercising the RL<->CNN interface
    without a trained CNN, per its own docstring) to cycle the crop-condition
    input through healthy/moderate_stress/severe_stress, and checks whether
    the trained agent's action choice (on days with h_t < h_min) shifts with
    it via the r3 stress-penalty term. This is the chart with no equivalent
    in the reference paper, since their state space has no crop-condition
    channel."""
    condition_names = ["healthy", "moderate_stress", "severe_stress"]
    counts = np.zeros((config.CROP_CONDITION_LEVELS, config.NUM_ACTIONS))

    for seed in range(num_seasons):
        weather = SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=20_000 + seed)
        crop_condition_provider = MockCropConditionProvider(sequence=[0, 0, 1, 1, 2, 2, 1, 0], cycle=True)
        env = SugarcaneIrrigationEnv(weather, crop_condition_provider=crop_condition_provider)
        state, _ = env.reset()
        done, t = False, 0
        while not done:
            h_t = state[config.FORECAST_HORIZON_DAYS]
            condition = crop_condition_provider(t)
            action = agent.act(state, greedy=True)
            if h_t < config.H_MIN_MM:
                counts[condition, action] += 1
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            t += 1

    df = pd.DataFrame(counts, columns=[f"A{a}" for a in range(config.NUM_ACTIONS)])
    df.insert(0, "crop_condition", condition_names)
    totals = df[[f"A{a}" for a in range(config.NUM_ACTIONS)]].sum(axis=1).replace(0, 1)
    pct_df = df.copy()
    for a in range(config.NUM_ACTIONS):
        pct_df[f"A{a}"] = 100 * df[f"A{a}"] / totals

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(condition_names))
    width = 0.25
    bar_colors = [COLORS["action_0"], COLORS["action_1"], COLORS["action_2"]]
    for a in range(config.NUM_ACTIONS):
        ax.bar(x + (a - 1) * width, pct_df[f"A{a}"], width, label=f"A{a} ({int(config.ACTIONS[a]*100)}%)", color=bar_colors[a])
    ax.set_xticks(x); ax.set_xticklabels(condition_names)
    ax.set_ylabel("% of decisions (days with h_t < h_min)")
    ax.set_title("Action choice vs. CNN-derived crop-condition signal")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "rl_09_action_by_crop_condition")
    save_table(df, "rl_09_action_by_crop_condition")


def main(num_episodes=None, retrain=True):
    if retrain:
        print("Training DQN agent (instrumented run for charting)...")
        agent, history_df = train_instrumented(num_episodes=num_episodes)
    else:
        agent = load_trained_agent()
        history_path = "results/rl_checkpoints/history_instrumented.json"
        with open(history_path) as f:
            history_df = pd.DataFrame(json.load(f))

    print("Chart 1/9: training curves"); chart_training_curves(history_df)
    print("Chart 2/9: decomposed rewards"); chart_decomposed_rewards(history_df)
    print("Chart 3/9: epsilon decay"); chart_epsilon_decay(history_df)
    print("Chart 4/9: policy comparison"); chart_policy_comparison(agent)
    print("Chart 5/9: Q-value landscape"); chart_q_value_landscape(agent)
    print("Chart 6/9: action distribution"); chart_action_distribution(agent)
    print("Chart 7/9: season case study"); chart_season_case_study(agent)
    print("Chart 8/9: rainfall forecast quality"); chart_rainfall_forecast_quality()
    print("Chart 9/9: action vs. crop condition"); chart_action_by_crop_condition(agent)
    print("\nAll RL charts written to results/figures/, data to results/tables/.")


if __name__ == "__main__":
    main()
