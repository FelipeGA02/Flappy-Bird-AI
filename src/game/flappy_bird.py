from __future__ import annotations
import os
import pygame
from typing import List
from .utils import Pipe, load_image, rects_collide, make_pipe_pair

# Constantes base para escala
BASE_SCREEN_W = 288
BASE_SCREEN_H = 
BASE_BIRD_W = 34
BASE_BIRD_H = 24
BASE_PIPE_W = 52
BASE_PIPE_GAP = 100
BASE_FLOOR_H = 112

# Definir a tela final
SCREEN_W = 480   # largura da tela
SCREEN_H = 640   # altura da tela
FLOOR_H = int(BASE_FLOOR_H * SCREEN_H / BASE_SCREEN_H)

# Escala proporcional
BIRD_WIDTH = int(BASE_BIRD_W * SCREEN_W / BASE_SCREEN_W)
BIRD_HEIGHT = int(BASE_BIRD_H * SCREEN_H / BASE_SCREEN_H)
PIPE_WIDTH = int(BASE_PIPE_W * SCREEN_W / BASE_SCREEN_W)
PIPE_GAP = int(BASE_PIPE_GAP * SCREEN_H / BASE_SCREEN_H)
BIRD_X = int(50 * SCREEN_W / BASE_SCREEN_W)
PIPE_SPEED = int(2 * SCREEN_W / BASE_SCREEN_W)

# Física do pássaro
GRAVITY = 0.5 * SCREEN_H / BASE_SCREEN_H
JUMP_VELOCITY = -8 * SCREEN_H / BASE_SCREEN_H

class FlappyGame:
    def __init__(self, assets_dir: str, seed: int | None = None, render_mode: str | None = None):
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H)) if render_mode == "human" else None
        self.assets_dir = assets_dir

        # Carregar imagens
        self.bg = load_image(os.path.join(assets_dir, "background.png"), (SCREEN_W, SCREEN_H))
        self.bird_img = load_image(os.path.join(assets_dir, "bird.png"), (BIRD_WIDTH, BIRD_HEIGHT))
        self.pipe_img = load_image(os.path.join(assets_dir, "pipe.png"), (PIPE_WIDTH, SCREEN_H))

        self.reset()

    def reset(self):
        self.bird_y = SCREEN_H // 2
        self.bird_vy = 0.0
        self.pipes: List[Pipe] = [
            make_pipe_pair(x=SCREEN_W + 40),
            make_pipe_pair(x=SCREEN_W + 40 + 160)
        ]
        self.score = 0
        self.ticks = 0
        self.done = False
        return self.get_state()

    def flap(self):
        self.bird_vy = JUMP_VELOCITY

    def step(self, action: int):
        if action == 1:
            self.flap()

        # Física
        self.bird_vy += GRAVITY
        self.bird_y += self.bird_vy

        # Limites da tela
        if self.bird_y < 0:
            self.bird_y = 0
            self.bird_vy = 0
        if self.bird_y > SCREEN_H - FLOOR_H - BIRD_HEIGHT:
            self.bird_y = SCREEN_H - FLOOR_H - BIRD_HEIGHT
            self.done = True

        # Mover canos
        for p in self.pipes:
            p.x -= PIPE_SPEED

        # Reciclar canos
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.pipes.append(make_pipe_pair(self.pipes[-1].x + 160))

        # Pontuação
        front = self.pipes[0]
        passed = (front.x + PIPE_WIDTH) < BIRD_X <= (front.x + PIPE_WIDTH + PIPE_SPEED)
        if passed:
            self.score += 1

        # Colisão
        bird_rect = pygame.Rect(BIRD_X, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
        pipes_rects = self._pipes_rects()
        if rects_collide(bird_rect, pipes_rects):
            self.done = True

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
            rects.append(pygame.Rect(int(p.x), bottom_y, PIPE_WIDTH, SCREEN_H - FLOOR_H - bottom_y))
        return rects

    def get_state(self):
        front = self.pipes[0]
        pipe_dx = front.x - BIRD_X
        return float(self.bird_y), float(self.bird_vy), float(pipe_dx), float(front.gap_y)

    def render(self):
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fundo
        self.screen.blit(self.bg, (0, 0))

        # Canos
        for p in self.pipes:
            top_h = p.gap_y - PIPE_GAP // 2
            bottom_y = p.gap_y + PIPE_GAP // 2
            # top pipe
            top_img_scaled = pygame.transform.scale(self.pipe_img, (PIPE_WIDTH, top_h))
            self.screen.blit(pygame.transform.flip(top_img_scaled, False, True), (int(p.x), 0))
            # bottom pipe
            bottom_height = SCREEN_H - FLOOR_H - bottom_y
            bottom_img_scaled = pygame.transform.scale(self.pipe_img, (PIPE_WIDTH, bottom_height))
            self.screen.blit(bottom_img_scaled, (int(p.x), bottom_y))

        # Chão
        pygame.draw.rect(self.screen, (222, 216, 149), (0, SCREEN_H - FLOOR_H, SCREEN_W, FLOOR_H))

        # Pássaro
        self.screen.blit(self.bird_img, (BIRD_X, int(self.bird_y)))

        # Placar
        font = pygame.font.SysFont("arial", 24)
        txt = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(txt, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
