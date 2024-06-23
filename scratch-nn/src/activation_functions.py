import numpy as np

def RelU(x):
    return np.maximum(0, x)

def SoftMax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=1, keepdims=True)
