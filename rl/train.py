"""
Training loop: one episode = one full sugarcane ratoon season
(config.SEASON_LENGTH_DAYS days). Mirrors the reference paper's Algorithm 1
structure (Table 4): reset environment each episode, epsilon-greedy action
selection, store transitions, sample minibatches, periodically sync target
network, decay epsilon.
"""

import os
import json
import numpy as np

from . import config
from .environment import SugarcaneIrrigationEnv
from .agent import DQNAgent


def train(weather_provider_factory, crop_condition_provider=None, num_episodes=None, verbose=True):
    """
    weather_provider_factory: callable() -> WeatherDataProvider-like object.
      Called fresh each episode so each season can use a different weather
      realization (either a different real year, or a re-seeded synthetic
      provider during smoke-testing).
    """
    num_episodes = num_episodes or config.NUM_EPISODES
    agent = DQNAgent()
    history = {"episode_reward": [], "episode_irrigation_mm": [], "episode_drainage_mm": [], "epsilon": []}

    for episode in range(num_episodes):
        weather = weather_provider_factory()
        env = SugarcaneIrrigationEnv(weather, crop_condition_provider=crop_condition_provider)
        state, _ = env.reset()

        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        history["episode_reward"].append(total_reward)
        history["episode_irrigation_mm"].append(env.cumulative_irrigation_mm)
        history["episode_drainage_mm"].append(env.cumulative_drainage_mm)
        history["epsilon"].append(agent.epsilon)

        if verbose and (episode + 1) % max(num_episodes // 10, 1) == 0:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"reward={total_reward:.2f} | irrigation={env.cumulative_irrigation_mm:.1f}mm | "
                  f"drainage={env.cumulative_drainage_mm:.1f}mm | epsilon={agent.epsilon:.3f}")

    os.makedirs(config.__dict__.get("CHECKPOINT_DIR", "results/rl_checkpoints"), exist_ok=True)
    checkpoint_dir = "results/rl_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    import torch
    torch.save(agent.q_network.state_dict(), os.path.join(checkpoint_dir, "dqn_model.pt"))
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return agent, history


if __name__ == "__main__":
    from .weather_provider import SyntheticWeatherProvider
    train(lambda: SyntheticWeatherProvider(config.SEASON_LENGTH_DAYS, seed=np.random.randint(0, 100000)),
          num_episodes=20, verbose=True)
