import numpy as np
from random import uniform
import math


def randomize(arr):
    for r in range(len(arr)):
        for c in range(len(arr[0])):
            arr[r][c] = uniform(-1,1)
    
    return arr


# the sigmoid function. | loads of calculus i don't understand!
def sigmoid(x):
    return 1/(1+math.exp(-x))


sigmoid_numpy = np.vectorize(sigmoid) # numpy don't work with normal py funcs, convert to numpy style func

# derivative of the sigmoid function
def derivative_sigmoid(y):
    # return sigmoid(x) * (1 - sigmoid(x))
    # here y is assumed to already be sigmoid(x)
    return y * (1-y)


derivative_sigmoid_numpy = np.vectorize(derivative_sigmoid)  # numpy don't work with normal py funcs, convert to numpy style func


# converts any arbitrary array to a probabilities
def soft_max(arr):
    numerator = np.exp(arr)
    denominator = np.sum(numerator)
    soft_max_output = numerator/denominator
    return soft_max_output
#
# def soft_max(arr):
#     s = sum(arr)
#     soft_max_output = arr/s
#     return soft_max_output


# soft_max_numpy = np.vectorize(soft_max) # numpy don't work with normal py funcs, convert to numpy style func


# make a vector from a list or return the array if it is already one.
def from_array(l):
    arr = np.ndarray(shape=(len(l),1))
    for i in range(len(arr)):
        arr[i][0] = l[i]
    return arr
    

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, use_soft_max=False):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # the weights matrix, for weights on input -> hidden (2D)
        self.weights_ih = randomize(np.ndarray(shape=(self.hidden_nodes,self.input_nodes)))

        # the weights matrix, for weights on hidden -> output (2D)
        self.weights_ho = randomize(np.ndarray(shape=(self.output_nodes, self.hidden_nodes)))
        
        # the biases, just 1 col
        self.bias_h = randomize(np.ndarray(shape=(self.hidden_nodes,1)))
        self.bias_o = randomize(np.ndarray(shape=(self.output_nodes,1)))
        
        # the learning rate, controls how much the network can learn per training.
        self.learning_rate = 0.1
        
        # should the neural network calculate probabilities?
        self.use_soft_max = use_soft_max
        
    
    def feed_forward(self,input_array):
        # input array is expected to be a list, that is 1 row and self.input_nodes columns
        if len(input_array) != self.input_nodes:
            raise Exception("INPUT ARRAY LENGTH DOES NOT MATCH INPUT_NODES.")
    
        # create an array of the inputs to work with
        inputs = from_array(input_array)
    
        # GENERATING THE HIDDEN LAYERS OUTPUTS!
        # the hidden layer is gonna be the matrix product of sef.weights_ih and inputs | order matters!
        hidden = np.dot(self.weights_ih, inputs)
        # add the hidden bias to the hidden vector | numpy magic!
        hidden = np.add(hidden, self.bias_h)
        # apply the activation function
        hidden = sigmoid_numpy(hidden)
    
        #############################################
    
        # GENERATING THE OUTPUT LAYERS OUTPUT!
    
        # matrix product of the hidden->output weights | order matters!
        outputs = np.dot(self.weights_ho, hidden)
        # add the output bias
        outputs = np.add(outputs, self.bias_o)
        # apply the activation function
        outputs = sigmoid_numpy(outputs)
        # apply soft max function to return probabilities if specified while initialising the neural network
        if self.use_soft_max:
            outputs = soft_max(outputs)
        # return the output
        return outputs
    
    
    def train(self,input_array,target_array):
        # input array is expected to be a list, that is 1 row and self.input_nodes columns
        if len(input_array) != self.input_nodes:
            raise Exception("INPUT ARRAY LENGTH DOES NOT MATCH INPUT_NODES.")
    
        # create an array of the inputs to work with
        inputs = from_array(input_array)
    
        # GENERATING THE HIDDEN LAYERS OUTPUTS!
        # the hidden layer is gonna be the matrix product of sef.weights_ih and inputs | order matters!
        hidden = np.dot(self.weights_ih, inputs)
        # add the hidden bias to the hidden vector | numpy magic!
        hidden = np.add(hidden, self.bias_h)
        # apply the activation function
        hidden = sigmoid_numpy(hidden)
    
        #############################################
    
        # GENERATING THE OUTPUT LAYERS OUTPUT!
    
        # matrix product of the hidden->output weights | order matters!
        outputs = np.dot(self.weights_ho, hidden)
        # add the output bias
        outputs = np.add(outputs, self.bias_o)
        # apply the activation function
        outputs = sigmoid_numpy(outputs)
        # apply soft max function to return probabilities if specified while initialising the neural network
        if self.use_soft_max:
            outputs = soft_max(outputs)
    
        # convert target_array to a matrix
        targets = from_array(target_array)
        
        # calculate the error
        # error = targets - inputs
        output_errors = np.subtract(targets, outputs)

        # gradient = outputs * (1 - outputs) | the derivative, calculus i don't understand!
        # Calculate gradient
        output_gradients = derivative_sigmoid(outputs)  # apply the derivative function, calculus!
        output_gradients = np.multiply(output_gradients, output_errors) # the normal element wise product
        output_gradients = np.multiply(output_gradients, self.learning_rate)  # multiply each element if the gradients by the learning rate
        
        # Calculate the deltas
        hidden_T = hidden.T  # transpose the hidden outputs
        weight_ho_deltas = np.dot(output_gradients,hidden_T)  # calculate the matrix product of the output_gradients and hidden_T | order matters!
        
        # adjust the weights by the deltas for the hidden->output connections
        self.weights_ho = np.add(self.weights_ho, weight_ho_deltas)
        # Adjust the bias by its deltas (which is just the gradients)
        self.bias_o = np.add(self.bias_o, output_gradients)
        
        # BACK-PROPAGATION, TO ADJUST THE WEIGHTS PRIOR TO THE OUTPUT ONES
        # would have done this in a loop of there were more than 1 hidden layers, but just 1 so hard code it!
        
        weights_ho_T = self.weights_ho.T # the transposed weights, of the next layer, in this case the hidden->output layer
        
        # calculate the hidden errors, simple formula -> for every output error, increase the hidden error of a connection based on its weights
        # if an output error is 2, and there are 3 hidden connections to the output layer,
        # with weights, 0.5, 0.3, 0.2
        # the error contribution of each connection will be proportional to its weights,
        # so the error contribution of w0 will be 1, of w1 will be 0.6, of w2 will be 0.4
        # do this for every output node, and sum the errors
        # basically the matrix product
        hidden_errors = np.dot(weights_ho_T,output_errors) # using output errors here cuz, the next layer is the output layer, if it were something else we would use that layer's errors.
        
        # calculate the hidden gradient
        hidden_gradients = derivative_sigmoid(hidden)  # apply the derivative function
        hidden_gradients = np.multiply(hidden_gradients, hidden_errors) # element wise multiplication with the hidden_errors
        hidden_gradients = np.multiply(hidden_gradients, self.learning_rate) # multiply each element by the learning rate
        
        # calculate the input->hidden deltas, it would be the previous->this_layers deltas if it were a multi dimensional network
        inputs_T = inputs.T # get the transposed version on the inputs
        weights_ih_deltas = np.dot(hidden_gradients,inputs_T) # calculate the weight deltas
        
        # adjust the weights
        self.weights_ih = np.add(self.weights_ih, weights_ih_deltas)
        # adjust the biases
        self.bias_h =np.add(self.bias_h, hidden_gradients)
        
        
        
        # TRAINING BASED ON GIVEN INPUT IS DONE!
        # NOTE: A LOT OF TRAINING IS REQUIRED FOR ACCURATE RESULTS
        
    
    def reload(self):
        # the weights matrix, for weights on input -> hidden
        self.weights_ih = randomize(np.ndarray(shape=(self.hidden_nodes, self.input_nodes)))
    
        # the weights matrix, for weights on hidden -> output
        self.weights_ho = randomize(np.ndarray(shape=(self.output_nodes, self.hidden_nodes)))
    
        # the biases, just 1 col
        self.bias_h = randomize(np.ndarray(shape=(self.hidden_nodes, 1)))
        self.bias_o = randomize(np.ndarray(shape=(self.output_nodes, 1)))
    
        # the learning rate, controls how much the network can learn per training.
        self.learning_rate = 0.1
        
        
    def get_weights(self):
        weights = {"ih": self.weights_ih, "ho": self.weights_ho}
        return weights
    
    
    def get_biases(self):
        biases = {"ih":self.bias_h,"ho":self.bias_o}
        return biases
    
    
    def get_brain(self):
        return {"ih":{"weights":self.weights_ih,"biases":self.bias_h},"ho":{"weights":self.weights_ho,"biases":self.bias_o}}

        
    def copy(self):
        copy = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self.use_soft_max)
        copy.weights_ih = self.weights_ih.copy()
        copy.weights_ho = self.weights_ho.copy()
        copy.bias_h = self.bias_h.copy()
        copy.bias_o = self.bias_o.copy()
        return copy
        
    
    def mutate(self,func):
        self.weights_ih = func(self.weights_ih)
        self.weights_ho = func(self.weights_ho)
        self.bias_h = func(self.bias_h)
        self.bias_o = func(self.bias_o)



