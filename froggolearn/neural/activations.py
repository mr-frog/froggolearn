import numpy as np
from froggolearn.utils import sigmoid
from scipy.special import xlogy

def check_type(type):
    types = ["log", "relu"]
    if not type in types:
        raise ValueError("Activation Type not recognized")

def activation(X, type):
    check_type(type)
    if type == "log":
        return sigmoid(X)
    if type == "relu":
        return np.clip(X, 0, np.finfo(X.dtype).max)

def derivative(X, type):
    check_type(type)
    if type == "log":
        return X * (1 - X)
    if type == "relu":
        return np.heaviside(X, 0)

def cost(h, y, weights, type, l2 = 1):
    if type not in ["log"]:
        raise ValueError("Loss Type not recognized")
    if type == "log":
        reg_sum = 0
        for weight in weights:
            reg_sum += np.power(weight[1:], 2).sum()
        return -(((xlogy(y, h) + xlogy(1 - y, 1 - h)).sum() / h.shape[0])
                  + (l2 / (2 * h.shape[0])) * reg_sum)
