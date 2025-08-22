from __future__ import annotations
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .model import QNetwork

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 128,
        target_sync: int = 1000,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.action_dim = action_dim

        self.replay = ReplayBuffer(buffer_capacity, state_dim)

        self.eps = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.train_steps = 0

    def act(self, state_np: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.eps:
            return random.randrange(self.action_dim)
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
        return int(torch.argmax(qvals, dim=1).item())

    def remember(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = [x.to(self.device) for x in self.replay.sample(self.batch_size)]

        # Q(s,a)
        q_sa = self.q(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.q_target(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()

        # decaimento do epsilon
        if self.eps > self.eps_end:
            self.eps -= self.eps_decay
            self.eps = max(self.eps, self.eps_end)

        # sincroniza alvo
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(self.q.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state_dict)
        self.q_target.load_state_dict(self.q.state_dict())
