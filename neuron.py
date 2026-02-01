#Importing numpy for math computations

import numpy as np

#Our activation function

def sigmoid(x):
    #Function : f(x) = 1 / (1 + e^(-x))
    return 1/(1+np.exp(-x))

#Defining object neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        #Weight inputs, add bias , then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# w1 = 0, w2 =1
weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

# x1 = 2, x2 =3
x = np.array([2, 3])
print(n.feedforward(x))
