import pickle
import random,datetime,time

import pygame

from KidNeuralNetwork.NeuralNetwork import NeuralNetwork as NN

pygame.init()

# pygame constants
WIDTH = 600
HEIGHT = 800
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird Neuro Evoultion!")
clock = pygame.time.Clock()
SCORE_FONT = pygame.font.SysFont("comicsans", 50, bold=True, italic=False)

# number of frames in the bird animation and other bird constants
NUM_BIRD_IMAGES = 8
BIRD_DIMENSIONS = 100
BIRD_IMAGES = []
BIRD_MASKS = []
# load the bird images
for n in range(1, NUM_BIRD_IMAGES + 1):
    image = pygame.image.load("images/" + "bird" + str(n) + ".tiff").convert_alpha()
    BIRD_IMAGES.append(pygame.transform.scale(image, (BIRD_DIMENSIONS, BIRD_DIMENSIONS)))
    BIRD_MASKS.append(pygame.mask.from_surface(pygame.transform.scale(image, (BIRD_DIMENSIONS, BIRD_DIMENSIONS))))


# load the background image
BG_IMG = pygame.transform.scale(pygame.image.load("images/" + "bg.png"), (WIDTH, HEIGHT))

# pipe constants
PIPE_WIDTH = 100
PIPE_HEIGHT = HEIGHT
PIPE_IMG = pygame.transform.scale(pygame.image.load("images/" + "pipe.png"), (PIPE_WIDTH, PIPE_HEIGHT))
pipe_top_mask = pygame.mask.from_surface(pygame.transform.flip(PIPE_IMG,False,True))  # get the mask of the top pipe
pipe_bottom_mask = pygame.mask.from_surface(PIPE_IMG)  # get the mask of the bottom pipe
# what is gonna be the population size of each generation
POPULATION_SIZE = 100

# should we use the best bird that we have saved?
use_best = True

player_mode = False

PIXEL_PERFECT = True


# other bird specific constants
# putting them in a class of their own for clarity
class Bird_Props:
    MAX_ROTATION = 25
    ROT_VEL = 5
    ANIMATION_TIME = 5
    IMAGES = BIRD_IMAGES
    JUMP_VEL = -10
    GRAVITY = 5
    MAX_DISTANCE = 15


# the bird class
class Bird:
    def __init__(self, x=200, y=200, brain=None):
        self.x = x  # define the x position of the bird, this will remain constant throughout the game
        self.y = y  # the y pos of the bird, will change
        self.vel = 0  # the velocity of the bird, increases if the bird wants to go up
        self.tilt = 0  # how much should the bird tilt by, tilt upwards if it is flying up, and straight down if not. for animation purposes only
        self.img_count = 0  # which image to blit, for animation only
        self.tick_count = 0  # how much time has passed by since the last time the bird flew up. Used in the physics to move the bird
        self.score = 0  # what is the score of the bird? better the score more the chances of reproducing.
        self.fitness = 0  # normalised score, represents the probability of reproducing.
        
        # if a brain was passed, then copy it and mutate it a little
        if brain:
            self.brain = brain
            self.brain.mutate(mutate)
        # if not just create one
        else:
            self.brain = NN(6, 64, 1)
    
    # the mechanism to make the bird go up, just inc the vel and the action will be considered in the move function
    def up(self):
        self.vel = Bird_Props.JUMP_VEL
        self.tick_count = 0
    
    # move the bird
    def move(self):
        # increase the time since it last jumped
        self.tick_count += 1
        
        # calculate the distance to move, -ve if upwards +ve if downwards
        # d = ut + 0.5at^2
        d = self.vel * self.tick_count + 0.5 * Bird_Props.GRAVITY * self.tick_count ** 2
        
        # let the bird only move a max amount of distance
        if d > Bird_Props.MAX_DISTANCE:
            d = Bird_Props.MAX_DISTANCE
        
        # move the bird by the distance calculated
        self.y = self.y + d
        self.vel = self.vel + Bird_Props.GRAVITY*self.tick_count
        if self.vel>0:
            self.vel= 0
        
        # tilt the bird accordingly
        if d < 0:
            self.tilt = Bird_Props.MAX_ROTATION
        
        elif self.tilt > -90:
            self.tilt -= Bird_Props.ROT_VEL
        
        # constrain the bird to the limits of the window
        self.constrain()
        # self.score += 1  # give the bird a positive increment in score for every frame survived
    
    # show the bird
    def show(self, win):
        img = self.get_img()
        img = pygame.transform.rotate(img, self.tilt)
        win.blit(img, (self.x, self.y))
        self.img_count = (self.img_count + 1) % NUM_BIRD_IMAGES
    
    # get the current image of the bird
    def get_img(self):
        if self.tilt < 0:
            img = Bird_Props.IMAGES[0]
        else:
            img = Bird_Props.IMAGES[self.img_count]
        return img
    
    # the function to constrain the bird to the limits of the screen
    def constrain(self):
        if self.y + BIRD_DIMENSIONS / 2 > HEIGHT:
            self.y = HEIGHT - BIRD_DIMENSIONS / 2
            self.tilt = 0
        if self.y < 0:
            self.tilt = 0
            self.vel, self.y = 0, 0
    
    # feed-forward some inputs to the bird's brain, and move up or down accordingly
    def think(self, pipes):
        # get the pipe closest ahead
        closest_pipe = get_closest_pipe(self, pipes)
        if not closest_pipe:
            return
        
        # normalise the top-y, bottom-y and x-cor of the pipe. this is so that we aren't dealing with huge numbers and the math is easier
        pipe_top = remap(closest_pipe.top, 0, HEIGHT, 0, 1)
        pipe_bottom = remap(closest_pipe.bottom, 0, HEIGHT, 0, 1)
        pipe_x = remap(closest_pipe.x, self.x, WIDTH, 0, 1)
        # normalise the y-pos, velocity and x-cor of the bird. this is so that we aren't dealing with huge numbers and the math is easier
        bird_y = remap(self.y, 0, HEIGHT, 0, 1)
        bird_vel = remap(self.vel, 0, Bird_Props.JUMP_VEL, 0, 1)
        bird_x = remap(self.x, 0, WIDTH, 0, 1)
        
        # form an input array with the above normalised values, which will then be fed-forward to the brain!
        inputs = [pipe_top, pipe_bottom, pipe_x, bird_y, bird_vel, bird_x]
        output = self.brain.feed_forward(inputs)  # the output will have only one value
        # flatten the output and check if the bird wants to move up!
        if output.flatten()[0] >= 0.5:
            self.up()
    
    # a function to copy the bird, mutation and other stuff are done in __init__
    def copy(self):
        return Bird(brain=self.brain.copy())
    
    # get the mask of the bird, this is for pixel perfect collision checking
    def get_mask(self):
        if self.tilt < 0:
            i = 0
        else:
            i = self.img_count
        return BIRD_MASKS[i]

# a mutate function, that is passed on to the mutate function of the brain of the bird, when it is reproducing
def mutate(arr, mutate_rate=0.2, deviation=0.2):
    arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if random.random() <= mutate_rate:
                arr[i][j] += random.uniform(-1 * deviation, deviation)
    return arr


# the pipe class, involves both the top and the bottom pipe
class Pipe:
    # some of the pipe constants
    GAP = 200
    VEL = 5
    TOP_IMG = pygame.transform.flip(PIPE_IMG,False,True)
    BOTTOM_IMG = PIPE_IMG
    TOP_MASK = pygame.mask.from_surface(TOP_IMG)
    BOTTOM_MASK = pygame.mask.from_surface(BOTTOM_IMG)

    def __init__(self, x=WIDTH):
        self.x = x  # the x cor of the pipe, this will be changing
        self.top, self.bottom = 0, 0  # initialise the top and bottom cors of the pipe to 0
        
        self.set_height()  # calculate and set the top and bottom cors
    
    def set_height(self):
        self.top = random.randrange(self.GAP, HEIGHT - self.GAP)
        self.bottom = self.top + self.GAP
    
    # the function to move the pipe
    def move(self):
        self.x -= self.VEL
    
    # show the pipe on the window
    def show(self, win):
        win.blit(self.TOP_IMG, (self.x, self.top - self.TOP_IMG.get_height()))
        win.blit(self.BOTTOM_IMG, (self.x, self.bottom))
    
    # check for a collision
    def collide(self, bird, pixel_perfect=False):
        if not pixel_perfect:
            # the "non" pixel-perfect collision. Much Faster!
            # if the x-cor of the bird is in between the starting x-cor of the pipe and the ending-xcor of the pipe,
            # then the bird may collide with the pipe
            if self.x < bird.x + BIRD_DIMENSIONS and bird.x < self.x + PIPE_WIDTH:
                # if the y-cor of the bird is less than the top pipes y-cor there is a collision!
                # or if the y-cor of the bird + the height of the bird is greater than the y-cor of the bottom pipe, there is a collision!
                if self.top > bird.y+BIRD_DIMENSIONS or self.bottom < bird.y:
                    return True
                bird.score += (1 / self.GAP) * self.VEL
                bird.score = round(bird.score,2)
            return False
        
        else:
            # pixel perfect collision, uses pygame masks. very slow :(
            # works only if the pipe and the bird have transparent background
            bird_mask = bird.get_mask()
            top_offset = (self.x - bird.x, self.top - self.TOP_IMG.get_height() - round(bird.y))
            bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

            b_point = bird_mask.overlap(self.BOTTOM_MASK, bottom_offset)
            t_point = bird_mask.overlap(self.TOP_MASK, top_offset)
            if b_point or t_point:
                return True
            bird.score += (1 / self.GAP) * self.VEL
            bird.score = round(bird.score, 2)
            return False
    
    # check if the pipe has gone beyond the screen
    def off_screen(self):
        if self.x + PIPE_WIDTH < 0:
            return True


# a function to "remap" a value from and old range to a new range
def remap(old_val, old_min, old_max, new_min, new_max):
    # percentages will be the same
    # (old_val-old_min)/(old_max-old_min) = (new_val - new_min)/(new_max - new_min)
    new_val = (((old_val - old_min) / (old_max - old_min)) * (new_max - new_min)) + new_min
    return new_val


# get the pipe closest to a bird
def get_closest_pipe(bird, pipes):
    for pipe in pipes:
        if pipe.x+PIPE_WIDTH >= bird.x:
            return pipe


# setup the birds, if we are using the best bird then just add that to the birds array,
# otherwise generate a random population
def setup(birds):
    if player_mode:
        birds.append(Bird())
    elif use_best:
        with open("best_bird", "rb") as f:
            birds.append(pickle.load(f))
            birds[0].score = 0
    else:
        for b in range(POPULATION_SIZE):
            birds.append(Bird())


# draw all the graphics on the window
def draw(win, pipes, birds, score, gen_number,max_score):
    win.blit(BG_IMG, (0, 0))
    for bird in birds:
        bird.show(win)
    
    for pipe in pipes:
        pipe.show(win)
    
    text = SCORE_FONT.render(f"Score: {score}", True, (200, 200, 200))
    win.blit(text, (10, 10 + text.get_height()))
    text = SCORE_FONT.render(f"Generation: {gen_number}", True, (200, 200, 200))
    win.blit(text, (10, 60 + text.get_height()))
    text = SCORE_FONT.render(f"High Score: {max_score}", True, (200, 200, 200))
    win.blit(text, (10, 110 + text.get_height()))
    
    pygame.display.update()


# get a new bird given a list of old birds and their fitness - fitness is the probability of reproducing
# i dont understand how this algorithm works
def pool_section(birds):
    index = 0
    r = random.random()
    while r > 0:
        r -= birds[index].fitness
        index += 1
    
    index -= 1
    
    return birds[index].copy()


# generate a new generation of birds, given the old one
def new_gen(all_birds):
    # if we are using the best bird, just set the score and fitness of the best bird to 0
    if use_best or player_mode:
        all_birds = all_birds.copy()
        all_birds[0].score = 0
        all_birds[0].fitness = 0
        return all_birds
    
    # calculate the sum of all the birds' scores
    scores_sum = 0
    new_birds = []
    for bird in all_birds:
        scores_sum += bird.score
    # assign each bird a fitness or probability if reproducing
    for bird in all_birds:
        bird.fitness = bird.score / scores_sum
    
    # generate a new population of birds
    for _ in range(len(all_birds)):
        bird = pool_section(all_birds)
        new_birds.append(bird)
    
    return new_birds


# the main logic of the game
# returns [max_score, new_gen (if required)]
def logic(pipes, active_birds, all_birds,best_bird):
    max_score = 0  # the max score for the current gen
    
    # make every bird think and move
    for bird in active_birds:
        if not player_mode:
            bird.think(pipes)
        bird.move()
        if bird.score > max_score:
            max_score = bird.score
    
    # make every pipe move and check for any collisions
    for p in range(len(pipes) - 1, -1, -1):
        pipe = pipes[p]
        pipe.move()
        for b in range(len(active_birds) - 1, -1, -1):
            bird = active_birds[b]
            if pipe.collide(bird,PIXEL_PERFECT):
                active_birds.pop(b)
        # remove the pipe if it goes off screen
        if pipe.off_screen():
            pipes.pop(p)
    
    # if the entire gen has died out create a new one
    if len(active_birds) == 0:
        for bird in all_birds:
            if bird.score >best_bird.score:
                best_bird = bird.copy()
                best_bird.score = bird.score
        new_birds = new_gen(all_birds)
        return [max_score, new_birds,best_bird]
    return [max_score]

def get_best(birds):
    bb = birds[0]
    for bird in birds[1:]:
        if bird.score > bb.score:
            bb = bird
    return bb
# the main game loop
def main():
    # set up the birds
    active_birds = []
    setup(active_birds)
    all_birds = active_birds.copy()
    gen_number = 0  # the gen number
    current_max_score = 0  # the current max score
    best_bird_current_game = Bird()
    
    # initialise the pipes
    pipes = []
    pipe_interval = 75  # how frequently should we add pipes?
    pipe_counter = 0  # when the pipe counter reaches the pipe interval add a pipe
    
    fps = 30  # the fps
    train_for = 1  # how many times should we train, before drawing. just exists to speed up training
    
    run = True  # while this is true the game will be on!
    
    while run:
        clock.tick(fps)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # if we were training, then get the best bird and save it
                if not use_best:
                    bb = best_bird_current_game
                    for bird in active_birds:
                        if bird.score > bb.score:
                            bb = bird
                    try:
                        with open("best_bird", "rb") as read_file:
                            prev_best = pickle.load(read_file)
                            print(prev_best.score, bb.score)
                            if bb.score > prev_best.score:
                                with open("best_bird", "wb") as write_file:
                                    pickle.dump(bb, write_file)
                    except:
                        with open("best_bird", "wb") as write_file:
                            pickle.dump(bb, write_file)
                run = False
            
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    with open(f"{datetime.datetime.now()}", "wb") as write_file:
                        pickle.dump(get_best(active_birds), write_file)
        
        # game controls
        keys = pygame.key.get_pressed()
        
        if player_mode:
            if keys[pygame.K_UP]:
                active_birds[0].up()
        else:
            if keys[pygame.K_UP]:
                fps += 1
                if fps > 100:
                    fps = 100
            
            elif keys[pygame.K_DOWN]:
                fps -= 1
                if fps < 0:
                    fps = 0
            elif keys[pygame.K_u]:
                train_for += 1
                if train_for > 100:
                    train_for = 100
            elif keys[pygame.K_d]:
                train_for -= 1
                if train_for < 0:
                    train_for = 0
            elif keys[pygame.K_r]:
                train_for = 1
                fps = 30
        
        for t in range(train_for):
            # perform the game logic
            res = logic(pipes, active_birds, all_birds,best_bird_current_game)
            current_max_score = res[0]  # change the current_max_score accordingly
            
            if len(res) > 1:
                # create a new gen, if required
                new_birds = res[1]
                best_bird_current_game = res[2]
                gen_number += 1
                active_birds = new_birds
                all_birds = new_birds.copy()
                pipes.clear()
            
            # add a new pipe if required
            if pipe_counter % pipe_interval == 0:
                pipe_counter = 0
                pipes.append(Pipe(WIDTH))
            pipe_counter += 1
        
        # draw the game
        draw(WINDOW, pipes, active_birds, current_max_score, gen_number,best_bird_current_game.score)


if __name__ == '__main__':
    main()
