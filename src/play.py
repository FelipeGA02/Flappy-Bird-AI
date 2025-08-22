import argparse
import time
import numpy as np
from .env import FlappyBirdEnv, Config
from .dqn_agent import DQNAgent

def play(checkpoint: str, fps: int = 60, episodes: int = 5, seed: int = 123):
    env = FlappyBirdEnv(render_mode="human", config=Config(seed=seed))
    agent = DQNAgent(state_dim=4, action_dim=2)
    agent.load(checkpoint)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # ação greedy (sem exploração)
            q = agent.online
            with __import__("torch").no_grad():
                s = __import__("torch").tensor(obs,
                dtype=__import__("torch").float32).unsqueeze(0)
                a = int(__import__("torch").argmax(q(s), dim=1).item())
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        time.sleep(0.5)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    play(args.checkpoint, args.fps, args.episodes, args.seed)
