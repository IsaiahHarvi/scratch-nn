import numpy as np
from network import NeuralNetwork
from activation_functions import RelU, SoftMax

class CNN(NeuralNetwork):
    def __init__(self, in_features: int, hidden_layers: list[int], out_features: int) -> None:
        super().__init__(in_features, hidden_layers, out_features)
    ...