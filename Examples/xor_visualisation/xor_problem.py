from KidNeuralNetwork.NeuralNetwork import NeuralNetwork as NN
from random import choice
import pygame

pygame.init()
clock = pygame.time.Clock()
fps = 30

width = 500
height = 500
resolution = 20
cols = int(width / resolution)
rows = int(height / resolution)

window = pygame.display.set_mode((width, height))

nn = NN(2, 2, 1)
lr = 0.1
training_data = [{
    "input": [0, 1],
    "output": [1]
}, {
    "input": [1, 0],
    "output": [1]
}, {
    "input": [1, 1],
    "output": [0]
}, {
    "input": [0, 0],
    "output": [0]
}]


def draw():
    window.fill((255, 255, 255))
    
    nn.learning_rate = lr
    
    for i in range(5000):
        d = choice(training_data)
        nn.train(d["input"], d["output"])
    
    for i in range(rows):
        for j in range(cols):
            x = i * resolution
            y = j * resolution
            
            input_1 = i / (rows - 1)
            input_2 = j / (cols - 1)
            output = nn.feed_forward([input_1, input_2]).flatten()
            color = (output[0] * 255, output[0] * 255, output[0] * 255)
            pygame.draw.rect(window, color, pygame.Rect(x, y, resolution, resolution))
    
    pygame.display.update()


def main():
    global lr
    
    run = True
    while run:
        clock.tick(fps)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                break
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_UP:
                    lr += 0.01
                    if lr > 1:
                        lr = 1
                elif e.key == pygame.K_DOWN:
                    lr -= 0.01
                    if lr < 0:
                        lr = 0
                elif e.key == pygame.K_r:
                    nn.reload()
                    
        draw()


if __name__ == '__main__':
    main()
