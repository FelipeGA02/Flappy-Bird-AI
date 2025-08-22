import os
import random
import pygame

# --- Configurações da tela ---
SCREEN_W, SCREEN_H = 826, 499
FLOOR_H = 1

# --- Pássaro ---
BIRD_WIDTH, BIRD_HEIGHT = 100, 90
BIRD_X = 50
GRAVITY = 0.3
JUMP_VELOCITY = -8

# --- Cano ---
PIPE_WIDTH = 100
PIPE_GAP = 220
PIPE_SPEED = 5
HORIZONTAL_GAP = 400  
SPEED_INCREMENT = 0.001 

# --- Classes utilitárias ---
class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(PIPE_GAP // 2, SCREEN_H - FLOOR_H - PIPE_GAP // 2)

def load_image(path, size=None):
    """
    Carrega a imagem do caminho especificado.
    Se `size` for fornecido, redimensiona a imagem.
    """
    try:
        img = pygame.image.load(path).convert_alpha()
        if size:
            img = pygame.transform.scale(img, size)
    except Exception as e:
        print(f"Erro ao carregar {path}: {e}")
        img = pygame.Surface(size or (50, 50), pygame.SRCALPHA)
        img.fill((255, 0, 0, 255))  # fallback vermelho
    return img

def rects_collide(rect, rects):
    return any(rect.colliderect(r) for r in rects)

def make_pipe_pair(x):
    return Pipe(x)

# --- Jogo ---
class FlappyGame:
    def __init__(self, assets_dir="assets"):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.assets_dir = assets_dir

        # Carrega imagens reais
        self.bg = load_image(os.path.join(assets_dir, "background.png"), (SCREEN_W, SCREEN_H))
        self.bird_img = load_image(os.path.join(assets_dir, "bird.png"), (BIRD_WIDTH, BIRD_HEIGHT))
        self.pipe_img = load_image(os.path.join(assets_dir, "pipe.png"), (PIPE_WIDTH, SCREEN_H))

        self.reset()

    def reset(self):
        self.bird_y = SCREEN_H // 2
        self.bird_vy = 0
        self.pipes = [
            make_pipe_pair(SCREEN_W + 40),
            make_pipe_pair(SCREEN_W + 40 + HORIZONTAL_GAP)
        ]
        self.score = 0
        self.done = False
        self.pipe_speed = PIPE_SPEED  # resetar velocidade inicial

    def flap(self):
        self.bird_vy = JUMP_VELOCITY

    def step(self, action):
        if action == 1:
            self.flap()
        self.bird_vy += GRAVITY
        self.bird_y += self.bird_vy

        # Limites da tela
        if self.bird_y < 0:
            self.bird_y = 0
            self.bird_vy = 0
        if self.bird_y > SCREEN_H - FLOOR_H - BIRD_HEIGHT:
            self.bird_y = SCREEN_H - FLOOR_H - BIRD_HEIGHT
            self.done = True

        # Movimentação dos canos
        for p in self.pipes:
            p.x -= self.pipe_speed

        # Aumentar a velocidade gradualmente
        self.pipe_speed += SPEED_INCREMENT

        # Reciclar canos
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.pipes.append(make_pipe_pair(self.pipes[-1].x + HORIZONTAL_GAP))
            self.score += 1  # pontua ao passar do cano

        # Colisão
        bird_rect = pygame.Rect(BIRD_X, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
        pipe_rects = []
        for p in self.pipes:
            top_height = max(p.gap_y - PIPE_GAP // 2, 0)
            bottom_y = p.gap_y + PIPE_GAP // 2
            bottom_height = max(SCREEN_H - FLOOR_H - bottom_y, 0)
            pipe_rects.append(pygame.Rect(p.x, 0, PIPE_WIDTH, top_height))
            pipe_rects.append(pygame.Rect(p.x, bottom_y, PIPE_WIDTH, bottom_height))
        if rects_collide(bird_rect, pipe_rects):
            self.done = True

    def render(self):
        # Fundo
        self.screen.blit(self.bg, (0, 0))

        # Canos com imagem
        for p in self.pipes:
            # topo (invertido)
            top_height = max(p.gap_y - PIPE_GAP // 2, 0)
            top_img = pygame.transform.scale(self.pipe_img, (PIPE_WIDTH, top_height))
            top_img = pygame.transform.flip(top_img, False, True)
            self.screen.blit(top_img, (p.x, 0))

            # fundo
            bottom_y = p.gap_y + PIPE_GAP // 2
            bottom_height = max(SCREEN_H - FLOOR_H - bottom_y, 0)
            bottom_img = pygame.transform.scale(self.pipe_img, (PIPE_WIDTH, bottom_height))
            self.screen.blit(bottom_img, (p.x, bottom_y))

        # Chão
        pygame.draw.rect(self.screen, (222, 216, 149), (0, SCREEN_H - FLOOR_H, SCREEN_W, FLOOR_H))

        # Pássaro
        self.screen.blit(self.bird_img, (BIRD_X, int(self.bird_y)))

        # Score
        font = pygame.font.SysFont("arial", 24)
        txt = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(txt, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

# --- Loop principal ---
game = FlappyGame()
running = True
while running:
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            action = 1
    game.step(action)
    game.render()
    if game.done:
        print("Game Over! Score:", game.score)
        game.reset()

pygame.quit()
