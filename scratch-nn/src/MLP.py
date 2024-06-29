import numpy as np
from network import NeuralNetwork
from activation_functions import RelU, SoftMax

class MLP(NeuralNetwork):
    def __init__(self, in_features: int, hidden_layers: list[int], out_features: int) -> None:
        super().__init__(in_features, hidden_layers, out_features)
        self.name = self.__class__.__name__
        
    def predict(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        return self._forward(x)

    def _forward(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        layers: list = [x]
        for i in range(len(self.weights) - 1): # exclude output layer
            layers.append(
                RelU(np.dot(layers[i], self.weights[i]) + self.biases[i])
            )
        return SoftMax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1]), layers
    
    def _create_layers(self) -> None:
        def he_init(size): # huge improvement over a random init
            """ https://arxiv.org/pdf/1502.01852v1 """
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

    def _backpropogation(self, x, y, lr) -> float:
        y_pred, layers = self._forward(x)
        loss = self._cross_entropy(y, y_pred) 

        grad = {}
        d_output = y_pred # gradient of loss wrt output layer
        d_output[range(x.shape[0]), y] -= 1 # error term
        d_output /= x.shape[0]

        grad[f"w-{len(self.weights)-1}"] = np.dot(layers[-1].T, d_output)
        grad[f"b-{len(self.weights)-1}"] = np.sum(d_output, axis=0, keepdims=True)
        
        # calculate gradients wrt hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            d_hidden = np.dot(d_output, self.weights[i + 1].T)
            d_hidden[layers[i + 1] <= 0] = 0  # derivative of ReLU

            grad[f"w-{i}"] = np.dot(layers[i].T, d_hidden)
            grad[f"b-{i}"] = np.sum(d_hidden, axis=0, keepdims=True)

            d_output = d_hidden

        # update weights and biases 
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad[f"w-{i}"]
            self.biases[i] -= lr * grad[f"b-{i}"]

        return loss
