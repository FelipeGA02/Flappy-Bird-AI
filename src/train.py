import argparse
from collections import deque
import numpy as np
from tqdm import trange
from .env import FlappyBirdEnv, Config
from .dqn_agent import DQNAgent
from .utils import set_seed, moving_average

def train(episodes: int = 2000, render_every: int = 0, seed: int = 42,
        save_path: str = "checkpoints/best.pt"):
    set_seed(seed)
    env = FlappyBirdEnv(render_mode=None, config=Config(seed=seed))
    agent = DQNAgent(state_dim=4, action_dim=2)
    best_score = -1
    scores = []
    ma50 = []
    
    for ep in trange(episodes, desc="Training"):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        ep_score = 0
        if render_every and (ep % render_every == 0):
            env.render_mode = "human"
        else:
            env.render_mode = None
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.push(obs, action, reward, next_obs, float(done))
            loss, td = agent.learn()
            obs = next_obs
            total_reward += reward
            ep_score = info.get("score", ep_score)
        
        scores.append(ep_score)
        ma50 = moving_average(scores, 50)
        
        if ep_score > best_score:
            best_score = ep_score
            agent.save(save_path)
    
    env.close()
    return np.array(scores), np.array(ma50)

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=6000)
parser.add_argument("--render_every", type=int, default=0,
help="render a cada N episódios (0 = nunca)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_path", type=str, default="checkpoints/best.pt")
args = parser.parse_args()
train(args.episodes, args.render_every, args.seed, args.save_path)