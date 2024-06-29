import numpy as np
# from plot_utils import plot_loss_animation

class NeuralNetwork:
    def __init__(self, in_features: int, hidden_layers: list[int], out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.weights = []
        self.biases = []
        self.id = (len(hidden_layers) + 2)
        self._create_layers()

    def predict(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        return self._forward(x)

    def _forward(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        raise NotImplementedError

    def _create_layers(self) -> None:
        raise NotImplementedError

    def _cross_entropy(self, y, y_pred) -> float:
        eps = 1e-10 # prevent log(0)
        y_pred = np.clip(y_pred, eps, 1. - eps)
        probabilities = y_pred[range(y.shape[0]), y]
        return np.sum(-np.log(probabilities)) / y.shape[0]
    
    def _backpropogation(self, x, y, lr) -> float:
        raise NotImplementedError

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> list[float]:
        losses = []
        for i in range(epochs):
            loss = self._backpropogation(x, y, lr)
            losses.append(loss)
            print(f'Epoch {i+1}/{epochs} - {loss=:.5f}')
        # plot_loss_animation(losses)
        return losses
