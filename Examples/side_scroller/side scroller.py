import random
import time

import numpy as np
import pygame
from pygame.locals import *

from KidNeuralNetwork.NeuralNetwork import NeuralNetwork as NN

pygame.init()
start_time = time.time()
W, H = 800, 447
win = pygame.display.set_mode((W, H))
pygame.display.set_caption("Side Scroller")

bg = pygame.image.load("images/bg.jpeg")

clock = pygame.time.Clock()

lives = 3
saw_images = [
    pygame.image.load("images/SAW0.png"),
    pygame.image.load("images/SAW1.png"),
    pygame.image.load("images/SAW2.png"),
    pygame.image.load("images/SAW3.png"),
]
wood_image = pygame.image.load("images/spike.png")
player_gameOverImg = pygame.image.load("images/0.png")
player_run = [pygame.image.load(("images/" + str(x) + ".png")) for x in range(8, 16)]
player_jump = [pygame.image.load(("images/" + str(x) + ".png")) for x in range(1, 8)]
player_slide = [
    pygame.image.load(("images/S1.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S2.png")),
    pygame.image.load(("images/S3.png")),
    pygame.image.load(("images/S4.png")),
    pygame.image.load(("images/S5.png")),
]

FontForScore = pygame.font.Font("freesansbold.ttf", 22)


class saw(object):
    
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.count = 0
    
    def draw(self):
        if self.count >= 8:
            self.count = 0
        win.blit(
            pygame.transform.scale(saw_images[self.count // 2], (64, 64)),
            (self.x, self.y),
        )
        self.count += 1


class wood(saw):
    def draw(self):
        win.blit(wood_image, (self.x, self.y))


# a mutate function, that is passed on to the mutate function of the brain of the bird, when it is reproducing
def mutate(arr, mutate_rate=0.2, deviation=0.2):
    arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if random.random() <= mutate_rate:
                arr[i][j] += random.uniform(-1 * deviation, deviation)
    return arr


# get the pipe closest to a bird
def get_closest_obstacle(runner, obstacles):
    for o in obstacles:
        if o.x + o.w >= runner.x:
            return o


# a function to "remap" a value from and old range to a new range
def remap(old_val, old_min, old_max, new_min, new_max):
    # percentages will be the same
    # (old_val-old_min)/(old_max-old_min) = (new_val - new_min)/(new_max - new_min)
    new_val = (((old_val - old_min) / (old_max - old_min)) * (new_max - new_min)) + new_min
    return new_val


class Player(object):
    jumpList = [
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -2,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -3,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
        -4,
    ]
    
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.width, self.height = 64, 64
        self.jumping = False
        # self.sliding = False
        # self.slideCount = 0
        self.jumpCount = 0
        self.runCount = 0
        # self.slideUp = False
        self.gameOver = False
        self.score = 0
        self.fitness = 0
        if brain:
            self.brain = brain.copy()
            self.brain.mutate(mutate)
        
        else:
            self.brain = NN(5, 64, 1)
    
    def move(self):
        if self.gameOver:
            self.sliding = False
            self.jumping = False
            self.slideCount = 0
            self.jumpCount = 0
            self.runCount = 0
            self.slideUp = False
            return
        
        
        elif self.jumping:
            self.y -= self.jumpList[self.jumpCount] * 1.2
            self.jumpCount += 1
            if self.jumpCount > 108:
                self.jumpCount = 0
                self.jumping = False
                self.runCount = 0
        
        # elif self.sliding or self.slideUp:
        #     if self.slideCount < 20:
        #         self.y += 1
        #     elif self.slideCount == 80:
        #         self.y -= 19
        #         self.sliding = False
        #         self.slideUp = True
        #     if self.slideCount >= 110:
        #         self.slideCount = 0
        #         self.slideUp = False
        #         self.runCount = 0
        #
        #     self.slideCount += 1
        
        self.score += 1
    
    def think(self, obstacles):
        closest = get_closest_obstacle(self, obstacles)
        if not closest:
            return
        ox, oy, ow, oh = remap(closest.x, 0, W, 0, 1), remap(closest.y, 0, H, 0, 1), remap(closest.w, 0, W, 0,
                                                                                           1), remap(closest.h, 0, H, 0,
                                                                                                     1)
        curr_action = 0  # nothing
        if self.jumping:
            curr_action = 1

        
        output = self.brain.feed_forward([ox, oy, ow, oh, curr_action]).flatten()
        if output>=0.5:
            if not self.jumping:
                self.jumping = True
        
    
    def draw(self, win):
        if self.jumping:
            win.blit(player_jump[self.jumpCount // 18], (self.x, self.y))
        #
        # elif self.sliding or self.slideUp:
        #     win.blit(player_slide[self.slideCount // 11], (self.x, self.y))
        
        else:
            if self.runCount > 42:
                self.runCount = 0
            win.blit(player_run[self.runCount // 6], (self.x, self.y))
            
            self.runCount += 1
    
    def copy(self):
        return Player(400, 317, self.brain)


def draw(runners, obstacles, high_score,current_score,gen, bgX, bgX2):
    win.blit(bg, (bgX, 0))
    win.blit(bg, (bgX2, 0))
    
    textForScore = FontForScore.render("High Score: %s" % int(high_score), True, (0, 0, 0))
    textRectForScore = textForScore.get_rect()
    textRectForScore.center = (60, 40)
    win.blit(textForScore, textRectForScore)
    textForScore = FontForScore.render("Score: %s" % int(current_score), True, (0, 0, 0))
    textRectForScore = textForScore.get_rect()
    textRectForScore.center = (120, 100)
    win.blit(textForScore, textRectForScore)

    textForScore = FontForScore.render("Gen: %s" % int(gen), True, (0, 0, 0))
    textRectForScore = textForScore.get_rect()
    textRectForScore.center = (60, 150)
    win.blit(textForScore, textRectForScore)
    for o in obstacles:
        o.draw()
    for runner in runners:
        runner.draw(win)
    pygame.display.update()


def CheckCollision(p1, p2):
    if isinstance(p2, wood):
        ox1 = p2.x - 24
        ox2 = p2.x + 24
        oy1 = 0
        oy2 = 330
    else:
        ox1 = p2.x - 32
        ox2 = p2.x + 32
        oy1 = p2.y - 32
        oy2 = p2.y + 32
    if p1.x >= ox1 and p1.x <= ox2 and p1.y >= oy1 and p1.y <= oy2:
        return True
    else:
        return False


def pool_section(runners):
    index = 0
    r = random.random()
    while r > 0:
        r -= runners[index].fitness
        index += 1
    
    index -= 1
    
    return runners[index].copy()


def new_gen(all_runners):
    # if we are using the best bird, just set the score and fitness of the best bird to 0
    # if use_best or player_mode:
    #     all_birds = all_birds.copy()
    #     all_birds[0].score = 0
    #     all_birds[0].fitness = 0
    #     return all_birds
    
    # calculate the sum of all the birds' scores
    scores_sum = 0
    new_birds = []
    for runner in all_runners:
        scores_sum += runner.score
    # assign each bird a fitness or probability if reproducing
    for runner in all_runners:
        runner.fitness = runner.score / scores_sum
    
    # generate a new population of birds
    for _ in range(len(all_runners)):
        runner = pool_section(all_runners)
        new_birds.append(runner)
    
    return new_birds


def logic(active_runners, all_runners, obstacles, ):
    for o in obstacles:
        for runner in reversed(active_runners):
            if CheckCollision(runner, o):
                runner.gameOver = True
                active_runners.remove(runner)
    best_runner = None
    for runner in all_runners:
        if not best_runner:
            best_runner = runner.copy()
            best_runner.score = runner.score
        
        elif runner.score > best_runner.score:
            best_runner = runner.copy()
            best_runner.score = runner.score
    
    for runner in active_runners:
        runner.think(obstacles)
        runner.move()
        if not best_runner:
            best_runner = runner.copy()
            best_runner.score = runner.score
        elif runner.score > best_runner.score:
            best_runner = runner.copy()
            best_runner.score = runner.score
    for o in obstacles:
        if o.x < o.w * -1:
            obstacles.remove(o)
        else:
            o.x -= 2
    
    if len(active_runners) == 0:
        new_birds = new_gen(all_runners)
        obstacles.clear()
        return [best_runner, new_birds]
    return [best_runner, None]


def main():
    run = True
    best = Player(400, 317)
    obstacles = [
]
    bgX = 0
    bgX2 = bg.get_width()
    population = 100
    active_runners = []
    for i in range(population):
        active_runners.append(Player(400, 317))
    
    all_runners = active_runners.copy()
    pygame.time.set_timer(USEREVENT + 1, 500)
    randtime = pygame.time.set_timer(USEREVENT + 2, random.randint(3000, 5000))
    
    gen = 0
    high_score = 0
    score = 0
    fps = 0
    loops = 100
    counter = 0
    spacing = 150
    
    while run:
        clock.tick(fps)
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         run = False
        #
        #     if event.type == USEREVENT + 1:
        #         fps += 1
        #         if fps>100:
        #             fps = 100
        #
        #     if event.type == USEREVENT + 2:
        #         randobject = random.randrange(0, 2)
        #         if randobject == 0:
        #             obstacles.append(saw(900, 310, 64, 64))
        #         elif randobject == 1:
        #             obstacles.append(wood(900, 0, 48, 320))
        
        bgX -= 2
        bgX2 -= 2

        if bgX < bg.get_width() * -1:
            bgX = bg.get_width()
        if bgX2 < bg.get_width() * -1:
            bgX2 = bg.get_width()
            
        for _ in range(loops):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
        

            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                fps += 1
                if fps > 100:
                    fps = 100
    
                print(fps)

            elif keys[K_DOWN]:
                fps -= 1
                if fps < 0:
                    fps = 0
                print(fps)
            elif keys[K_u]:
                loops += 1
                if loops > 100:
                    loops = 100
                print(loops)

            elif keys[K_d]:
                loops -= 1
                if loops < 1:
                    loops = 1
                print(loops)

            res = logic(active_runners, all_runners, obstacles)
            score = res[0].score
            if res[0].score > best.score:
                best = res[0]
                high_score = best.score
            if res[1]:
                gen+=1
                active_runners = res[1]
                all_runners = active_runners.copy()
            if counter%spacing==0:
                counter = 0
                obstacles.append(saw(900, 310, 64, 64))
                
            counter+=1
        
        
        draw(active_runners, obstacles, high_score,score,gen, bgX, bgX2)
    
    pygame.quit()


main()
