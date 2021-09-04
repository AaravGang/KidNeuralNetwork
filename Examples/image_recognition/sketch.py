from random import shuffle
import numpy as np
import math
import time
# np.save("apples.npy",apples)

import pygame
from KidNeuralNetwork.NeuralNetwork import NeuralNetwork

pygame.init()


width,height = 784,784

class Image:
    def __init__(self,img_arr,label):
        self.image = img_arr
        self.label = label
        
        
eye_images = np.load("eyes.npy")
eiffel_tower_images = np.load("eiffel_tower.npy")
apple_images = np.load("apples.npy")
mouth_images = np.load("mouth.npy")

images = [eye_images,mouth_images,apple_images]
labels = ["eyes","mouth","apple"]
all_images = []


def get_normalised(arr):
    return arr/255


for i in range(len(images)):
    for j in range(len(images[i])):
        all_images.append(Image(get_normalised(images[i][j]),i))
        
    
shuffle(all_images)

training_percent = 0.8
training,testing = all_images[:int(training_percent*len(all_images))],all_images[int(training_percent*len(all_images)):]

nn = NeuralNetwork(784,10,len(images),False)


window = pygame.display.set_mode((width,height))

pixels = []
resolution = 28


def get_pixels(arr):
    image = []
    for x in range(resolution):
        image.append([])
        for y in range(resolution):
            image[x].append([])
            for z in range(3):
                image[x][y].append(arr[y * resolution + x])
    return np.array(image).astype("uint8")


def setup():
    for i in range(width*height//(resolution**2)):
        pixels.append(get_pixels(apple_images[i]))
        
        
setup()


def train():
    for n,data in enumerate(training):
        inputs = data.image
        target = [1 if i==data.label else 0 for i in range(len(images))]
        nn.train(inputs,target)
    print("trained once")



def test(image=None):
    if image is None:
        image = []
    if len(image)>0:
        output = nn.feed_forward(image)
        print(output.flatten())
        print(output.flatten().argmax())
        print(labels[output.flatten().argmax()])
    else:
        count =0
        for data in testing:
            inputs = data.image
            target = data.label
            output = nn.feed_forward(inputs)
            if output.argmax()==target:
                count+=1
                
        percent_correct = count/len(testing)
        print(percent_correct)



for i in range(5):
    shuffle(training)
    train()
    test()


def draw_dataset():
    for i in range(width//resolution):
        for j in range(height//resolution):
            image = pixels[i*width//resolution + j]
            surf = pygame.surfarray.make_surface(image)
            window.blit(surf,(i*resolution,j*resolution))
        

def draw():
    window.fill(0)
    
    pygame.display.update()

# draw()


def main():

    run = True
    m_p_x = -1
    m_p_y = -1
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    screen_pixels = []
                    for i in range(0,width,resolution):
                        for j in range(0,height,resolution):
                            screen_pixels.append(sum(pygame.Surface.get_at(window,(i,j))[:3])/3)
                    
                    screen_pixels = get_normalised(np.array(screen_pixels))
                    test(screen_pixels)
                if e.key == pygame.K_c:
                    window.fill(0)
                    pygame.display.update()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            if math.dist((m_p_x,m_p_y),(mouse_x,mouse_y))<40:
                pygame.draw.line(window,(255,255,255),(m_p_x,m_p_y),(mouse_x,mouse_y),3)
                pygame.display.update()
        m_p_x = mouse_x
        m_p_y = mouse_y
            
    
        # draw()
main()