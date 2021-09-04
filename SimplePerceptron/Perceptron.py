from random import uniform



class Perceptron:
    def __init__(self,n,lr):
        self.weights = [uniform(-1,1) for _ in range(n)] # initialise the weights, number of weights is same as number of inputs
        
        self.c = lr # learning rate is constant

        
    def train(self,inputs,desired):
        
        guess = self.feed_forward(inputs) # get the current guess of the neuron
        
        
        # Compute the factor for changing the weight based on the error
        # Error = desired output - guessed output
        # Multiply by learning constant
        
        error = desired - guess
        
        # adjust the weights based on the error and the learning rate
        for i in range(len(self.weights)):
            self.weights[i] += inputs[i]*error*self.c
            
        
    def feed_forward(self,inputs):
        
        # calculate the cumulative sum
        # s = w0*i0 + w1*i1 + w2*i2 ...
        s = sum(map(lambda x,w:x*w,inputs,self.weights))
        
        output = activate(s) # calculate the output using the activation function
        
        return output
    
    def get_weights(self):
        return self.weights
        
   
   
def activate(a):
    return 1 if a>=0 else -1




        