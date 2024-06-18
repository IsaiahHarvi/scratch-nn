import numpy as np
from neuron import Neuron
from activation_functions import RelU, SoftMax


class NeuralNetwork:
    def __init__(self, in_features: int, hidden_layers: list, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.weights = []
        self.biases = []

        self._create_layers()

    def predict(self, x):
        return self._forward(x)

    def _forward(self, x):
        layers = [x]
        for i in range(len(self.weights) - 1):  # exclude output layer
            layers.append(
                RelU(np.dot(layers[i], self.weights[i]) + self.biases[i])
            )
        return SoftMax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1]) # softmax output
    
    def _create_layers(self):
        def he_init(size): # huge improvement over a random init
            return np.random.randn(*size) * np.sqrt(2 / size[0])

        # Input Layer
        self.weights.append(he_init((self.in_features, self.hidden_layers[0])))
        self.biases.append(np.zeros((1, self.hidden_layers[0])))

        # Hidden Layers
        for i in range(1, len(self.hidden_layers)): # exclude input layer
            self.weights.append(he_init((self.hidden_layers[i-1], self.hidden_layers[i])))
            self.biases.append(np.zeros((1, self.hidden_layers[i])))

        # Output Layer
        self.weights.append(he_init((self.hidden_layers[-1], self.out_features)))
        self.biases.append(np.zeros((1, self.out_features)))

    def _cross_entropy(self, y, y_pred):
        eps = 1e-10 # prevent log(0)
        y_pred = np.clip(y_pred, eps, 1. - eps)
        probabilities = y_pred[range(y.shape[0]), y]
        return np.sum(-np.log(probabilities)) / y.shape[0] # loss
    
    def _backpropogation(self, x, y, lr):
        layers = [x]
        # forward pass
        for i in range(len(self.weights) - 1): # exclude output layer
            layers.append(
                RelU(np.dot(layers[i], self.weights[i]) + self.biases[i])
            )
        y_pred = SoftMax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1])
        loss = self._cross_entropy(y, y_pred)

        grad = {}
        d_output = y_pred # gradient of loss wrt output layer
        d_output[range(x.shape[0]), y] -= 1 # error term
        d_output /= x.shape[0]

        grad['weights' + str(len(self.weights) - 1)] = np.dot(layers[-1].T, d_output)
        grad['biases' + str(len(self.biases) - 1)] = np.sum(d_output, axis=0, keepdims=True)
        
        # calculate gradients wrt hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            d_hidden = np.dot(d_output, self.weights[i + 1].T)
            d_hidden[layers[i + 1] <= 0] = 0  # derivative of ReLU

            grad['weights' + str(i)] = np.dot(layers[i].T, d_hidden)
            grad['biases' + str(i)] = np.sum(d_hidden, axis=0, keepdims=True)

            d_output = d_hidden

        # update weights and biases 
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad['weights' + str(i)]
            self.biases[i] -= lr * grad['biases' + str(i)]

        return loss

    def train(self, x, y, epochs=100, lr=0.01):
        losses = []
        for i in range(epochs):
            loss = self._backpropogation(x, y, lr)
            losses.append(loss)
            print(f'Epoch {i+1}/{epochs} - Loss: {loss}')

        return losses