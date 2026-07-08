"""
Evaluates the trained DQN irrigation policy against a conventional baseline
rule, the same style of comparison the reference paper made (their
"conventional flooded irrigation" vs "DQN irrigation" comparison, Fig. 5):
water use, drainage, and (here) time spent in a stressed crop-condition
state, averaged over multiple seasons.
"""

import numpy as np
import torch

from . import config
from .environment import SugarcaneIrrigationEnv
from .agent import DQNAgent, QNetwork


def conventional_baseline_policy(state):
    """Conventional rule: irrigate to full whenever water depth is below
    h_min, otherwise do nothing -- mirrors traditional farmer practice and
    the reference paper's baseline definition."""
    h_t = state[config.FORECAST_HORIZON_DAYS]  # index right after the 7 forecast values
    return 2 if h_t < config.H_MIN_MM else 0


def run_episode(env, policy_fn, agent=None):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state, greedy=True) if agent is not None else policy_fn(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return {
        "irrigation_mm": env.cumulative_irrigation_mm,
        "drainage_mm": env.cumulative_drainage_mm,
        "actual_et": env.cumulative_actual_et,
        "max_et": env.cumulative_max_et,
    }


def compare_policies(agent, weather_provider_factory, crop_condition_provider=None, num_seasons=10):
    dqn_results, baseline_results = [], []
    for _ in range(num_seasons):
        weather = weather_provider_factory()
        env_dqn = SugarcaneIrrigationEnv(weather, crop_condition_provider=crop_condition_provider)
        dqn_results.append(run_episode(env_dqn, None, agent=agent))

        env_base = SugarcaneIrrigationEnv(weather, crop_condition_provider=crop_condition_provider)
        baseline_results.append(run_episode(env_base, conventional_baseline_policy))

    def avg(results, key):
        return float(np.mean([r[key] for r in results]))

    summary = {
        "dqn_irrigation_mm": avg(dqn_results, "irrigation_mm"),
        "baseline_irrigation_mm": avg(baseline_results, "irrigation_mm"),
        "dqn_drainage_mm": avg(dqn_results, "drainage_mm"),
        "baseline_drainage_mm": avg(baseline_results, "drainage_mm"),
        "yield_proxy_dqn": avg(dqn_results, "actual_et") / max(avg(dqn_results, "max_et"), 1e-6),
        "yield_proxy_baseline": avg(baseline_results, "actual_et") / max(avg(baseline_results, "max_et"), 1e-6),
    }
    if summary["baseline_irrigation_mm"] < 1.0:
        summary["water_savings_pct"] = None  # baseline needed ~no irrigation this run; % savings is not meaningful
    else:
        summary["water_savings_pct"] = 100 * (1 - summary["dqn_irrigation_mm"] / summary["baseline_irrigation_mm"])
    return summary


def load_trained_agent(checkpoint_path="results/rl_checkpoints/dqn_model.pt"):
    agent = DQNAgent()
    agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    agent.q_network.eval()
    return agent


if __name__ == "__main__":
    from .weather_provider import SyntheticWeatherProvider
    agent = load_trained_agent()
    summary = compare_policies(
        agent,
        lambda: SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=np.random.randint(0, 100000)),
        num_seasons=10,
    )
    for k, v in summary.items():
        print(f"{k}: {v:.2f}" if v is not None else f"{k}: N/A")
