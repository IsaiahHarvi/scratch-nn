import numpy as np 

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        # weighted sum of the inputs and weights
        return np.dot(self.weights, inputs) + self.bias