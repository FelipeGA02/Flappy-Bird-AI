from __future__ import annotations
import os
import math
import pygame
from typing import Tuple, List
from .utils import (
    SCREEN_W, SCREEN_H, FLOOR_H, GRAVITY, JUMP_VELOCITY, PIPE_SPEED, PIPE_GAP,
    PIPE_WIDTH, BIRD_X, Pipe, load_image, rects_collide, make_pipe_pair
)

class FlappyGame:
    def __init__(self, assets_dir: str, seed: int | None = None, render_mode: str | None = None):
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H)) if render_mode == "human" else None
        self.assets_dir = assets_dir

        # assets (com fallback se faltar)
        self.bg = load_image(os.path.join(assets_dir, "background.png"), (SCREEN_W, SCREEN_H))
        self.pipe_img = load_image(os.path.join(assets_dir, "pipe.png"), (PIPE_WIDTH, SCREEN_H))
        self.bird_img = load_image(os.path.join(assets_dir, "bird.png"), (34, 24))

        self.reset()

    def reset(self):
        self.bird_y = SCREEN_H // 2
        self.bird_vy = 0.0
        self.pipes: List[Pipe] = [make_pipe_pair(x=SCREEN_W + 40), make_pipe_pair(x=SCREEN_W + 40 + 160)]
        self.score = 0
        self.ticks = 0
        self.done = False
        return self.get_state()

    def flap(self):
        self.bird_vy = JUMP_VELOCITY

    def step(self, action: int):
        # ação
        if action == 1:
            self.flap()

        # física
        self.bird_vy += GRAVITY
        self.bird_y += self.bird_vy

        # mover canos
        for p in self.pipes:
            p.x -= PIPE_SPEED

        # reciclar canos
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.pipes.append(make_pipe_pair(self.pipes[-1].x + 160))

        # pontuação ao passar cano da frente
        front = self.pipes[0]
        passed = (front.x + PIPE_WIDTH) < BIRD_X <= (front.x + PIPE_WIDTH + PIPE_SPEED)
        if passed:
            self.score += 1

        # colisões
        bird_rect = pygame.Rect(BIRD_X, int(self.bird_y), self.bird_img.get_width(), self.bird_img.get_height())
        pipes_rects = self._pipes_rects()
        self.done = rects_collide(bird_rect, pipes_rects)

        self.ticks += 1
        return self.get_state(), self.done

    def _pipes_rects(self):
        rects = []
        for p in self.pipes:
            top_h = p.gap_y - PIPE_GAP // 2
            bottom_y = p.gap_y + PIPE_GAP // 2
            # topo
            rects.append(pygame.Rect(int(p.x), 0, PIPE_WIDTH, top_h))
            # baixo
            rects.append(pygame.Rect(int(p.x), bottom_y, PIPE_WIDTH, (SCREEN_H - FLOOR_H) - bottom_y))
        return rects

    def get_state(self):
        # estado: [bird_y, bird_vy, pipe_dx, pipe_gap_center]
        front = self.pipes[0]
        pipe_dx = front.x - BIRD_X
        return (
            float(self.bird_y),
            float(self.bird_vy),
            float(pipe_dx),
            float(front.gap_y),
        )

    def render(self):
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # fundo
        self.screen.blit(self.bg, (0, 0))

        # canos
        for p in self.pipes:
            top_h = p.gap_y - PIPE_GAP // 2
            bottom_y = p.gap_y + PIPE_GAP // 2
            # topo (flip)
            top_img = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(top_img, (int(p.x), top_h - top_img.get_height()))
            # baixo
            self.screen.blit(self.pipe_img, (int(p.x), bottom_y))

        # chão (faixa simples)
        pygame.draw.rect(self.screen, (222, 216, 149), (0, SCREEN_H - FLOOR_H, SCREEN_W, FLOOR_H))

        # pássaro
        self.screen.blit(self.bird_img, (BIRD_X, int(self.bird_y)))

        # placar
        font = pygame.font.SysFont("arial", 24)
        txt = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(txt, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
