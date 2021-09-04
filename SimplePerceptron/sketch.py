from SimplePerceptron.Perceptron import Perceptron
import pygame
from random import uniform

pygame.init()
clock = pygame.time.Clock()
fps = 30

width = 1000
height = 1000

window = pygame.display.set_mode((width, height))

neuron = Perceptron(3,0.0001) # create a new perceptron with 3 inputs ( 1 of them is the bias )


num_training_points = 100
points = []


x_min = -1
y_min = -1
x_max = 1
y_max = 1


def f(x):
    return 0.89292828244*x + 0.0029  # calculate the y according to a given x


def setup():
    for i in range(num_training_points):
        x = uniform(x_min,x_max)
        y = uniform(y_min,y_max)
        answer = 1
        if y <= f(x):
            answer = -1
            
        points.append({
            "input":[x,y,1], # 1 is the bias
            "output": answer
        })


def remap(OldValue, OldMin, OldMax, NewMin, NewMax):
    return (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin


def draw():
    
    window.fill(0)

    x1 = remap(x_min, x_min, x_max, 0, width)
    y1 = remap(f(x_min), y_min, y_max, 0, height)
    x2 = remap(x_max, x_min, x_max, 0, width)
    y2 = remap(f(x_max), y_min, y_max, 0, height)
    pygame.draw.line(window,(255,255,255),(x1, y1), (x2, y2),5) # draw the line which we are checking against
    

    # draw the line that the perceptron things is the answer
    # for a perceptron with 3 weights,
    # w0*x_cor + w1*y_cor + w2*bias = 0
    # so , y_cor = -(w0*x_cor + w2*bias)/w1
    # here bias is taken as 1
    # therefore , y = -(w0*x + w2)/w1
    weights = neuron.weights

    x1 = x_min
    y1 = (-weights[2] - weights[0] * x1) / weights[1]
    x2 = x_max
    y2 = (-weights[2] - weights[0] * x2) / weights[1]
    # map the x and y co-ordinates to the correct pixels
    x1 = remap(x1, x_min, x_max, 0, width)
    y1 = remap(y1, y_min, y_max, 0, height)
    x2 = remap(x2, x_min, x_max, 0, width)
    y2 = remap(y2, y_min, y_max, 0, height)
    
    pygame.draw.line(window,(250,250,0),(x1,y1),(x2,y2),2)
    
    
    # train the points
    for pt in points:
        neuron.train(pt["input"],pt["output"])


    # draw the points
    for pt in points:
        # get the guess of the neuron about a point, it does not know the correct answer
        guess = neuron.feed_forward(pt["input"])
        
        color = (255,255,255) # white
        if pt["output"]>0:
            color = (0,200,0) # green

        x = remap(pt["input"][0],x_min,x_max,0,width)
        y = remap(pt["input"][1],y_min,y_max,0,height)

        pygame.draw.circle(window,color,(x,y),8,3)
        
        if guess!=pt["output"]: # if the guess is wrong draw a red circle
            pygame.draw.circle(window, (255,0,0), (x, y), 4)

    pygame.display.update()


def main():
    setup()
    run = True
    while run:
        clock.tick(fps)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                break
            
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    setup()
            
                elif e.key == pygame.K_c:
                    points.clear()
                 
        draw()
        
        if pygame.mouse.get_pressed()[0]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # get the x and y co-ordinates of the mouse and map it to the values used by the neuron
            x = remap(mouse_x, 0,width,x_min,x_max)
            y = remap(mouse_y, 0,height,y_min,y_max)
            answer = 1
            if y <= f(x):
                answer = -1

            points.append({
                "input": [x, y, 1],  # 1 is the bias
                "output": answer
            })

main()