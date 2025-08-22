import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..game.flappy_bird import FlappyGame
from ..game.utils import SCREEN_W, SCREEN_H

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, assets_dir: str, render_mode: str | None = None, frame_skip: int = 1):
        super().__init__()
        self.game = FlappyGame(assets_dir=assets_dir, render_mode=render_mode)
        self.render_mode = render_mode
        self.frame_skip = frame_skip

        # Obs: [bird_y, bird_vy, pipe_dx, pipe_gap_y]
        high = np.array([SCREEN_H, 20.0, 500.0, SCREEN_H], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.array(self.game.reset(), dtype=np.float32)
        info = {"score": 0}
        return state, info

    def step(self, action):
        total_reward = 0.0
        done = False
        passed_pipe = False
        for _ in range(self.frame_skip):
            state_tuple, done = self.game.step(int(action))
            # shaping
            reward = 1.0
            # dar bônus quando a pontuação aumentar
            if self.game.score > 0 and (self.game.score % 1 == 0):
                # detecta passagem aproximada
                passed_pipe = True
            if done:
                reward = -100.0
            total_reward += reward
            if done:
                break

        if passed_pipe:
            total_reward += 10.0

        obs = np.array(state_tuple, dtype=np.float32)
        terminated = done
        truncated = False
        info = {"score": self.game.score}
        if self.render_mode == "human":
            self.game.render()
        return obs, total_reward, terminated, truncated, info

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
