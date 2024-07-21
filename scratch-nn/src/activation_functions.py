import numpy as np


class RelU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class SoftMax:
    def __call__(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        exps = np.exp(x - x_max)
        return exps / np.sum(exps, axis=1, keepdims=True)
