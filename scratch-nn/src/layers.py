import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features) # He initialization
        self.bias = np.zeros((1, out_features))
        self.grad_weight = None
        self.grad_bias = None
        self.grad = None
        self.x = None
        self.z = None

    def __call__(self, x) -> np.ndarray:
        return self.forward(x)
    
    def forward(self, x) -> np.ndarray:
        self.x = x
        self.z = np.dot(x, self.weight) + self.bias
        return self.z
    
    def backpropogate(self, grad, lr) -> np.ndarray:
        self.grad_weight = np.dot(self.x.T, grad)
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)

        # gradient for propogation to the previous layer 
        self.grad = np.dot(grad, self.weight.T)

        # update weights and biases
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias

        return self.grad
    
    def wb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.weight, self.bias    
    