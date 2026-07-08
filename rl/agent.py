"""
DQN agent for sugarcane irrigation decisions.

Implements the same algorithm structure as the reference paper's Algorithm 1
(Table 4): replay memory, epsilon-greedy action selection, minibatch sampling,
target-network cloning every C steps. Network is slightly larger than the
reference paper's (7-5 hidden units) since our state has one extra dimension
(crop condition) and we want a bit more capacity for the added stress term.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import config

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Linear(10, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=None, num_actions=None, device=None):
        self.state_dim = state_dim or config.STATE_DIM
        self.num_actions = num_actions or config.NUM_ACTIONS
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.steps_done = 0

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < config.BATCH_SIZE:
            return None

        batch = self.replay_buffer.sample(config.BATCH_SIZE)
        states = torch.as_tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + config.GAMMA * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % config.TARGET_UPDATE_EVERY == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)
