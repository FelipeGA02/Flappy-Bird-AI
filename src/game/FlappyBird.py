import pygame
import os
import random
import neat

ai_playing = True
generation = 0

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
        displacement = 1.5 * (self.time ** 2) + self.speed * self.time

        if displacement > 16:
            displacement = 16
        elif displacement < 0:
            displacement -= 2

        self.y += displacement

        if displacement < 0 or self.y < (self.height + 50):
            if self.angle < self.rotation:
                self.angle = self.rotation
        else:
            if self.angle > -90:
                self.angle -= self.rotation_speed

    def draw(self, window):
        self.bird_image_index += 1

        if self.bird_image_index < self.animation_time:
            self.image = self.birdImages[0]
        elif self.bird_image_index < self.animation_time * 2:
            self.image = self.birdImages[1]
        elif self.bird_image_index < self.animation_time * 3:
            self.image = self.birdImages[2]
        elif self.bird_image_index < self.animation_time * 4:
            self.image = self.birdImages[1]
        elif self.bird_image_index >= self.animation_time * 4 + 1:
            self.image = self.birdImages[0]
            self.bird_image_index = 0

        if self.angle <= -80:
            self.image = self.birdImages[1]
            self.bird_image_index = self.animation_time * 2

        rotation_image = pygame.transform.rotate(self.image, self.angle)
        after_center_image = self.image.get_rect(topleft=(self.x, self.y)).center
        retangle = rotation_image.get_rect(center=after_center_image)
        window.blit(rotation_image, retangle.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    range = 160
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
        self.x -= Pipe.speed

    def draw(self, window):
        window.blit(self.top_pipe, (self.x, self.top_position))
        window.blit(self.pipe_base, (self.x, self.base_position))

    def colide(self, bird):
        bird_mask = bird.get_mask()
        topo_mask = pygame.mask.from_surface(self.top_pipe)
        base_mask = pygame.mask.from_surface(self.pipe_base)

        top_range = (self.x - bird.x, self.top_position - round(bird.y))
        base_range = (self.x - bird.x, self.base_position - round(bird.y))

        top_point = bird_mask.overlap(topo_mask, top_range)
        base_point = bird_mask.overlap(base_mask, base_range)

        return bool(base_point or top_point)


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
    if ai_playing:
        texto = score_font.render(f"Geração: {generation}", 1, (255, 255, 255))
        window.blit(texto, (10, 10))

    ground.draw(window)
    pygame.display.update()


def main(genomes, config):
    global generation
    generation += 1
    if ai_playing:
        redes = []
        genomes_list = []
        birds = []
        for _, genome in genomes:
            rede = neat.nn.FeedForwardNetwork.create(genome, config)
            redes.append(rede)
            genome.fitness = 0
            genomes_list.append(genome)
            birds.append(Bird(230, 350))
    else:
        birds = [Bird(230, 350)]
    ground = Ground(730)
    pipes = [Pipe(700)]
    window = pygame.display.set_mode((window_width, window_height))
    points = 0
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(45)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
            if not ai_playing:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        for bird in birds:
                            bird.jump()

        if len(pipes) == 0:
            pipes.append(Pipe(window_width))

        pipe_index = 0
        if len(birds) > 0:
            for i, pipe in enumerate(pipes):
                if pipe.x + pipe.top_pipe.get_width() > birds[0].x:
                    pipe_index = i
                    break
        else:
            running = False
            break

        for i, bird in enumerate(birds):
            bird.move()
            if ai_playing:
                genomes_list[i].fitness += 0.1
                output = redes[i].activate((
                    bird.y,
                    abs(bird.y - pipes[pipe_index].height),
                    abs(bird.y - pipes[pipe_index].base_position)
                ))
                if output[0] > 0.5:
                    bird.jump()

        ground.move()

        add_pipe = False
        remove_pipes = []

        for pipe in pipes:
            for i, bird in enumerate(birds):
                if pipe.colide(bird):
                    birds.pop(i)
                    if ai_playing:
                        genomes_list[i].fitness -= 1
                        genomes_list.pop(i)
                        redes.pop(i)
                if not pipe.passthrow and bird.x > pipe.x:
                    pipe.passthrow = True
                    add_pipe = True
            pipe.move()
            if pipe.x + pipe.top_pipe.get_width() < 0:
                remove_pipes.append(pipe)

        if add_pipe:
            points += 1

            velocidade_base = 5
            distancia_base = 140
            multiplicador = 1.03

            Pipe.speed = velocidade_base * (multiplicador ** points)
            distancia_entre_pipes = int(distancia_base * (multiplicador ** points))

            pipes.append(Pipe(window_width + distancia_entre_pipes))

            if ai_playing:
                for genome in genomes_list:
                    genome.fitness += 5

        for pipe in remove_pipes:
            pipes.remove(pipe)

        for i, bird in enumerate(birds):
            if (bird.y + bird.image.get_height()) > ground.y or bird.y < 0:
                birds.pop(i)
                if ai_playing:
                    genomes_list.pop(i)
                    redes.pop(i)

        draw_window(window, birds, pipes, ground, points)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(True))
    populacao.add_reporter(neat.StatisticsReporter())

    if ai_playing:
        populacao.run(main, 100)
    else:
        main(None, None)


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    path_config = os.path.join(path, 'configs.txt')
    run(path_config)
