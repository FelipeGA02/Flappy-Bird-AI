import random
from collections import deque, namedtuple
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .model import QNetwork
Transition = namedtuple("Transition", ("state", "action", "reward",
"next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: str = None,
        gamma: float = 0.99, lr: float = 1e-3,
        buffer_size: int = 50_000, batch_size: int = 64,
        target_update: int = 1000, eps_start: float = 1.0, eps_end:
        float = 0.05, eps_decay: int = 50_000):
        self.device = device or ("cuda" if torch.cuda.is_available() else
        "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.online = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        # Epsilon-greedy schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.total_steps = 0
        
    def act(self, state: np.ndarray) -> int:
        self.total_steps += 1
        eps = self.epsilon()
        if random.random() < eps:
            return random.randint(0, 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
            device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(torch.argmax(q, dim=1).item())
        
    def epsilon(self) -> float:
        # decai linearmente até eps_end
        return max(self.eps_end, self.eps_start - (self.eps_start -
        self.eps_end) * (self.total_steps / self.eps_decay))

    def push(self, *args):
        self.replay.push(*args)

    def learn(self) -> Tuple[float, float]:
        if len(self.replay) < self.batch_size:
            return 0.0, 0.0
        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32,
        device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64,
        device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32,
        device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state),
        dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32,
        device=self.device).unsqueeze(1)
        # Q(s,a)
        q_values = self.online(states).gather(1, actions)
        # Q_target = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        
        # Periodicamente sincroniza a rede alvo
        if self.total_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
            
        # métrica opcional: TD-error médio e epsilon
        with torch.no_grad():
            td_error = torch.abs(q_values - target_q).mean().item()
        return float(loss.item()), float(td_error)
    
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
        "model": self.online.state_dict(),
        "steps": self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["model"])
        self.target.load_state_dict(self.online.state_dict())
        self.total_steps = ckpt.get("steps", 0)