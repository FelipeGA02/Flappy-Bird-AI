import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
try:
    import pygame
except Exception:
    pygame = None

@dataclass
class Config:
    width: int = 288
    height: int = 512
    gravity: float = 0.35
    flap_impulse: float = -6.5
    max_vel: float = 10.0
    pipe_gap: int = 100
    pipe_width: int = 52
    pipe_spacing: int = 160 
    pipe_speed: float = 2.5
    floor_y: int = 450
    bird_x: int = 60
    bird_radius: int = 12
    seed: Optional[int] = None

class FlappyBirdEnv:
    """Ambiente minimalista do Flappy Bird (Gymnasium-like)."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, config: Config = Config()):
        self.cfg = config
        self.render_mode = render_mode
        self.rng = random.Random(self.cfg.seed)
        self.screen = None
        self.clock = None
        self.reset()

    # ---------------- Core API ----------------
    def reset(self, *, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.cfg.seed = seed
            self.rng = random.Random(seed)
        self.bird_y = self.cfg.height // 2
        self.bird_vy = 0.0
        self.ticks = 0
        self.score = 0
        self.pipes = [] # lista de (x, gap_y)
        # Cria dois canos iniciais
        start_x = self.cfg.width + 50
        for i in range(2):
            self._spawn_pipe(start_x + i * self.cfg.pipe_spacing)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert action in (0, 1), "ação inválida"
        self.ticks += 1
        # Física do pássaro
        if action == 1:
            self.bird_vy = self.cfg.flap_impulse
        else:
            self.bird_vy = min(self.bird_vy + self.cfg.gravity, self.cfg.max_vel)
        self.bird_y += self.bird_vy
        # Move canos e gera novos
        new_pipes = []
        for (x, gap_y) in self.pipes:
            x -= self.cfg.pipe_speed
            if x + self.cfg.pipe_width > 0:
                new_pipes.append((x, gap_y))
        self.pipes = new_pipes
        if len(self.pipes) == 0 or (self.pipes[-1][0] < self.cfg.width - self.cfg.pipe_spacing):
            self._spawn_pipe(self.cfg.width + 10)
        # Colisões
        terminated = False
        reward = 0.1 # sobreviver vale algo
        # chão / teto
        if self.bird_y + self.cfg.bird_radius >= self.cfg.floor_y or self.bird_y - self.cfg.bird_radius <= 0:
            terminated = True
            reward = -1.0
        else:
            # colisão com cano (AABB simplificado com círculo)
            bx, by, r = self.cfg.bird_x, self.bird_y, self.cfg.bird_radius
            next_x, gap_y = self._next_pipe()
            # retângulos dos canos
            top_rect = (next_x, 0, self.cfg.pipe_width, gap_y - self.cfg.pipe_gap // 2)
            bot_rect = (next_x, gap_y + self.cfg.pipe_gap // 2, self.cfg.pipe_width, self.cfg.floor_y - (gap_y + self.cfg.pipe_gap // 2))
            if self._circle_collides_rect(bx, by, r, top_rect) or self._circle_collides_rect(bx, by, r, bot_rect):
                terminated = True
                reward = -1.0
        # passou pelo cano?
        if next_x + self.cfg.pipe_width < bx <= next_x + self.cfg.pipe_width + self.cfg.pipe_speed:
            self.score += 1
            reward = 1.0

        truncated = False # sem limite de steps por episódio
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {"score": self.score}

    def render(self):
        if pygame is None:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
            pygame.display.set_caption("FlappyBirdEnv")
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        # fundo
        self.screen.fill((135, 206, 235))
        # chão
        pygame.draw.rect(self.screen, (222, 184, 135), (0, self.cfg.floor_y, self.cfg.width, self.cfg.height - self.cfg.floor_y))
        # canos
        for (x, gap_y) in self.pipes:
            top_h = gap_y - self.cfg.pipe_gap // 2
            bot_y = gap_y + self.cfg.pipe_gap // 2
            bot_h = self.cfg.floor_y - bot_y
            pygame.draw.rect(self.screen, (34, 139, 34), (x, 0, self.cfg.pipe_width, top_h))
            pygame.draw.rect(self.screen, (34, 139, 34), (x, bot_y, self.cfg.pipe_width, bot_h))
        # pássaro
        pygame.draw.circle(self.screen, (255, 215, 0), (self.cfg.bird_x, int(self.bird_y)), self.cfg.bird_radius)
        # placar
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame:
            pygame.quit()

    # --------------- Helpers -----------------
    def _spawn_pipe(self, x):
        margin = 60
        gap_y = self.rng.randint(margin, self.cfg.floor_y - margin)
        self.pipes.append((x, gap_y))

    def _next_pipe(self) -> Tuple[float, float]:
        # primeiro cano com x + width >= bird_x
        future = [p for p in self.pipes if p[0] + self.cfg.pipe_width >= self.cfg.bird_x - 1]
        future.sort(key=lambda p: p[0])
        return future[0]

    def _get_obs(self) -> np.ndarray:
        next_x, gap_y = self._next_pipe()
        dist_x = (next_x + self.cfg.pipe_width) - self.cfg.bird_x
        dist_y_top = (gap_y - self.cfg.pipe_gap // 2) - self.bird_y
        dist_y_bottom = (gap_y + self.cfg.pipe_gap // 2) - self.bird_y
        obs = np.array([
            dist_x / self.cfg.width,
            dist_y_top / self.cfg.height,
            dist_y_bottom / self.cfg.height,
            self.bird_vy / self.cfg.max_vel
        ], dtype=np.float32)
        return obs

    @staticmethod
    def _circle_collides_rect(cx, cy, cr, rect) -> bool:
        rx, ry, rw, rh = rect
        # ponto mais próximo do retângulo
        nx = max(rx, min(cx, rx + rw))
        ny = max(ry, min(cy, ry + rh))
        dx, dy = cx - nx, cy - ny
        return (dx * dx + dy * dy) <= cr * cr
