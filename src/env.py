import pygame
import numpy as np
import gymnasium as gym


class Config:
    def __init__(self, seed=42):
        self.screen_width = 400
        self.screen_height = 600
        self.gravity = 1
        self.flap_strength = -8
        self.pipe_speed = 3
        self.pipe_width = 60
        self.gap_height = 150
        self.bird_x = 50
        self.fps = 30
        self.seed = seed
        np.random.seed(seed)


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, config: Config = None):
        super().__init__()
        self.cfg = config or Config()
        self.render_mode = render_mode

        # ações: 0 = não pular, 1 = pular
        self.action_space = gym.spaces.Discrete(2)

        # observações: [posição_y do pássaro, velocidade_y, dist_x até próximo cano, dist_y até gap]
        high = np.array(
            [self.cfg.screen_height, 10, self.cfg.screen_width, self.cfg.screen_height],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=high, shape=(4,), dtype=np.float32
        )

        self.clock = pygame.time.Clock()
        self.screen = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bird_y = self.cfg.screen_height // 2
        self.bird_vel = 0
        self.score = 0
        self.pipes = [{"x": self.cfg.screen_width, "gap_y": np.random.randint(100, self.cfg.screen_height - 100)}]
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        reward = 0.1
        terminated = False
        truncated = False

        # mover pássaro
        self.bird_vel += self.cfg.gravity
        if action == 1:  # pular
            self.bird_vel = self.cfg.flap_strength
        self.bird_y += self.bird_vel

        # mover canos
        for pipe in self.pipes:
            pipe["x"] -= self.cfg.pipe_speed

        # verificar colisão
        if self._check_collision():
            reward = -1
            terminated = True

        # remover canos fora da tela
        self.pipes = [p for p in self.pipes if p["x"] + self.cfg.pipe_width > 0]

        # adicionar novos canos
        if len(self.pipes) == 0 or self.pipes[-1]["x"] < self.cfg.screen_width - 200:
            self._add_pipe()

        # calcular o próximo cano
        next_x, next_y = float("inf"), 0
        for pipe in self.pipes:
            if pipe["x"] + self.cfg.pipe_width >= self.cfg.bird_x:
                next_x, next_y = pipe["x"], pipe["gap_y"]
                break

        # recompensa extra se passar pelo cano
        if next_x != float("inf"):
            if next_x + self.cfg.pipe_width < self.cfg.bird_x <= next_x + self.cfg.pipe_width + self.cfg.pipe_speed:
                reward = 1
                self.score += 1

        obs = (self.bird_y, self.bird_vel, next_x - self.cfg.bird_x, next_y - self.bird_y)
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, {"score": self.score}

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.cfg.screen_width, self.cfg.screen_height)
            )

        self.screen.fill((135, 206, 235))

        # Desenhar pássaro
        pygame.draw.circle(
            self.screen, (255, 255, 0), (self.cfg.bird_x, int(self.bird_y)), 12
        )

        # Desenhar canos
        for pipe in self.pipes:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (pipe["x"], 0, self.cfg.pipe_width, pipe["gap_y"] - self.cfg.gap_height // 2),
            )
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (pipe["x"], pipe["gap_y"] + self.cfg.gap_height // 2,
                self.cfg.pipe_width, self.cfg.screen_height),
            )

        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

    # ------------------ Helpers ------------------

    def _get_obs(self):
        next_x, next_y = float("inf"), 0
        for pipe in self.pipes:
            if pipe["x"] + self.cfg.pipe_width >= self.cfg.bird_x:
                next_x, next_y = pipe["x"], pipe["gap_y"]
                break
        return np.array(
            [self.bird_y, self.bird_vel, next_x - self.cfg.bird_x, next_y - self.bird_y],
            dtype=np.float32,
        )

    def _check_collision(self):
        if self.bird_y <= 0 or self.bird_y >= self.cfg.screen_height:
            return True
        for pipe in self.pipes:
            if (
                self.cfg.bird_x + 12 > pipe["x"]
                and self.cfg.bird_x - 12 < pipe["x"] + self.cfg.pipe_width
            ):
                if (
                    self.bird_y - 12 < pipe["gap_y"] - self.cfg.gap_height // 2
                    or self.bird_y + 12 > pipe["gap_y"] + self.cfg.gap_height // 2
                ):
                    return True
        return False

    def _add_pipe(self):
        gap_y = np.random.randint(100, self.cfg.screen_height - 100)
        self.pipes.append({"x": self.cfg.screen_width, "gap_y": gap_y})
