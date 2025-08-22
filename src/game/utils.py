from dataclasses import dataclass
import random
import pygame

SCREEN_W = 288
SCREEN_H = 512
FLOOR_H = 112
GRAVITY = 0.35
JUMP_VELOCITY = -6.5
PIPE_SPEED = 2.5
PIPE_GAP = 120
PIPE_WIDTH = 52
BIRD_X = 60

@dataclass
class Pipe:
    x: float
    gap_y: int  # centro do vão

def load_image(path: str, fallback_size=None):
    try:
        img = pygame.image.load(path).convert_alpha()
    except Exception:
        # placeholder colorida se asset não existir
        img = pygame.Surface(fallback_size or (50, 50), pygame.SRCALPHA)
        img.fill((200, 50, 50, 255))
    return img

def rects_collide(bird_rect: pygame.Rect, pipes_rects: list[pygame.Rect]) -> bool:
    for r in pipes_rects:
        if bird_rect.colliderect(r):
            return True
    return bird_rect.bottom >= SCREEN_H - FLOOR_H or bird_rect.top <= 0

def make_pipe_pair(x: float) -> Pipe:
    margin = 40
    min_center = margin + PIPE_GAP // 2
    max_center = (SCREEN_H - FLOOR_H) - margin - PIPE_GAP // 2
    gap_center = random.randint(min_center, max_center)
    return Pipe(x=x, gap_y=gap_center)
