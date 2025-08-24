import pygame
import os
import random

ground_image = pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'base.png')))
background_image = pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'bg.png')))
bird_images = [
    pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'bird1.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'bird2.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'bird3.png'))),
]
pipe_image = pygame.transform.scale2x(pygame.image.load(os.path.join('assets', 'pipe.png')))

window_width = 500
window_height = 800

pygame.font.init()
score_font = pygame.font.SysFont('roboto', 20)
class Bird:
    
    birdImages = bird_images
    
    rotation = 25
    rotation_speed = 20
    animation_time = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.height = self.y
        self.time = 0
        self.bird_image_index = 0
        self.image = self.birdImages[0]

    def jump(self):
        self.speed = -10.5
        self.time = 0
        self.height = self.y

    def move(self):
        self.time += 1
        range = 1.5 * (self.time**2) + self.speed * self.time

        if range > 16:
            range = 16
        elif range < 0:
            range -= 2

        self.y += range

        if range < 0 or self.y < (self.height + 50):
            if self.angle < self.rotation:
                self.angle = self.rotation
        else:
            if self.angle > -90:
                self.angle -= self.rotation_speed

    def draw(self, window):
        self.bird_image_index += 1

        if self.bird_image_index < self.animation_time:
            self.image = self.birdImages[0]
        elif self.bird_image_index < self.animation_time*2:
            self.image = self.birdImages[1]
        elif self.bird_image_index < self.animation_time*3:
            self.image = self.birdImages[2]
        elif self.bird_image_index < self.animation_time*4:
            self.image = self.birdImages[1]
        elif self.bird_image_index >= self.animation_time*4 + 1:
            self.image = self.birdImages[0]
            self.bird_image_index = 0

        if self.angle <= -80:
            self.image = self.birdImages[1]
            self.bird_image_index = self.animation_time*2

        rotation_image = pygame.transform.rotate(self.image, self.angle)
        after_center_image = self.image.get_rect(topleft=(self.x, self.y)).center
        retangle = rotation_image.get_rect(center=after_center_image)
        window.blit(rotation_image, retangle.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)

class Pipe:
    range = 200
    speed = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top_position = 0
        self.base_position = 0
        self.top_pipe = pygame.transform.flip(pipe_image, False, True)
        self.pipe_base = pipe_image
        self.passthrow = False
        self.pipe_height()

    def pipe_height(self):
        self.height = random.randrange(50, 450)
        self.top_position = self.height - self.top_pipe.get_height()
        self.base_position = self.height + self.range

    def move(self):
        self.x -= self.speed

    def draw(self, window):
        window.blit(self.top_pipe, (self.x, self.top_position))
        window.blit(self.pipe_base, (self.x, self.base_position))

    def colidir(self, bird):
        bird_mask = bird.get_mask()
        topo_mask = pygame.mask.from_surface(self.top_pipe)
        base_mask = pygame.mask.from_surface(self.pipe_base)

        top_range = (self.x - bird.x, self.top_position - round(bird.y))
        base_range = (self.x - bird.x, self.base_position - round(bird.y))

        top_point = bird_mask.overlap(topo_mask, top_range)
        base_point = bird_mask.overlap(base_mask, base_range)

        if base_point or top_point:
            return True
        else:
            return False

class Ground:
    speed = 5
    width = ground_image.get_width()
    image = ground_image

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.speed
        self.x2 -= self.speed

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, window):
        window.blit(self.image, (self.x1, self.y))
        window.blit(self.image, (self.x2, self.y))


def draw_window(window, birds, pipes, ground, score):
    window.blit(background_image, (0, 0))
    for bird in birds:
        bird.draw(window)
    for pipe in pipes:
        pipe.draw(window)

    text = score_font.render(f"Pontuação: {score}", 1, (255, 255, 255))
    window.blit(text, (window_width - 10 - text.get_width(), 10))
    ground.draw(window)
    pygame.display.update()


def main():
    birds = [Bird(230, 350)]
    ground = Ground(730)
    pipes = [Pipe(700)]
    window = pygame.display.set_mode((window_width, window_height))
    points = 0
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for bird in birds:
                        bird.jump()

        for bird in birds:
            bird.move()
        ground.move()

        add_pipe = False
        remove_pipes = []
        for pipe in pipes:
            for i, bird in enumerate(birds):
                if pipe.colidir(bird):
                    birds.pop(i)
                if not pipe.passthrow and bird.x > pipe.x:
                    pipe.passthrow = True
                    add_pipe = True
            pipe.move()
            if pipe.x + pipe.top_pipe.get_width() < 0:
                remove_pipes.append(pipe)

        if add_pipe:
            points += 1
            pipes.append(Pipe(600))
        for pipe in remove_pipes:
            pipes.remove(pipe)

        for i, bird in enumerate(birds):
            if (bird.y + bird.image.get_height()) > ground.y or bird.y < 0:
                birds.pop(i)

        draw_window(window, birds, pipes, ground, points)

if __name__ == '__main__':
    main()
