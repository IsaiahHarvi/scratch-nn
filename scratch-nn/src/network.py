import numpy as np
from activation_functions import RelU, SoftMax
from layers import Linear


class NeuralNetwork:
    def __init__(self, sequential: list[Linear | RelU | SoftMax]) -> None:
        self.sequential = sequential
        self.layers = [l for l in sequential if isinstance(l, Linear)]
        self.id = len(self.layers)

    def __call__(self, x):
        return self._forward(x)
    
    def predict(self, x):
        return self._forward(x)

    def _forward(self, x):
        for layer in self.sequential:
            x = layer(x)
        return x

    def _cross_entropy(self, y, y_pred) -> float:
        eps = 1e-10 # prevent log(0)
        y_pred = np.clip(y_pred, eps, 1. - eps)
        probabilities = y_pred[range(y.shape[0]), y]
        return np.sum(-np.log(probabilities)) / y.shape[0]
    
    def backpropagation(self, x, y, lr) -> float:
        y_pred = self._forward(x)
        loss = self._cross_entropy(y, y_pred)

        grad = y_pred
        grad[range(x.shape[0]), y] -= 1
        grad /= x.shape[0]

        # Backward pass
        for i in range(len(self.sequential) - 1, -1, -1):
            layer = self.sequential[i]
            if isinstance(layer, RelU):
                grad *= layer.derivative(self.sequential[i-1].z)
            elif isinstance(layer, Linear):
                grad = layer.backpropogate(grad, lr)

        return loss
    
    def info(self, dir, epochs, lr) -> None:
        s = f"{'-'*50}\n"
        s = f"{epochs=}\t{lr=}\n{' '*50}\n"
        for idx, layer in enumerate(self.sequential):
            if (name := layer.__class__.__name__) not in ["RelU", "SoftMax"]:
                s += f"{idx}{' '*len(str(idx))}| {name}({layer.weight.shape[0]}, {layer.weight.shape[1]})\n"
            else:
                s += f"{idx}{' '*len(str(idx))}| {layer.__class__.__name__}\n"

        print(f"\n{s}", end="\n\n")

        with open(f"{dir}/_model_info.txt", "w") as f:
            f.write(s)

