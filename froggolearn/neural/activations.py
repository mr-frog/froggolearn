import numpy as np
from froggolearn.utils import sigmoid
from scipy.special import xlogy
def check_type(type):
    types = ["logistic", "relu", "leaky", "elu", "softmax"]
    if not type in types:
        raise ValueError("Activation Type not recognized")

def activation(X, type):
    check_type(type)
    if type == "logistic":
        return sigmoid(X)
    if type == "relu":
        ax = np.clip(X, 0, np.inf)
        return ax
    if type == "leaky":
        a = 0.01
        ax = np.clip(X, 0, np.finfo(X.dtype).max)
        ax[X < 0] = X[X < 0] * a
        return ax
    if type == "elu":
        a = 0.01
        ax = np.clip(X, 0, np.finfo(X.dtype).max)
        ax[X < 0] = (np.exp(X[X < 0]) - 1) * a
        return ax
    if type == "softmax":
        if len(X.shape) != 1:
            ax = X - X.max(axis=1)[:, np.newaxis]
            ax = np.exp(ax)
            ax = ax / ax.sum(axis=1)[:, np.newaxis]
        else:
            ax = X - X.max()
            ax = np.exp(ax)
            ax = ax / ax.sum()
        return ax

def derivative(X, type):
    check_type(type)
    if type == "log":
        return X * (1 - X)
    if type == "relu":
        dx = 1 * (X > 0)
        return dx
    if type == "leaky":
        a = 0.01
        dx = np.ones_like(X)
        dx[X < 0] = a
        return dx
    if type == "elu":
        a = 0.01
        dx = np.ones_like(X)
        dx[X < 0] = X[X < 0] + a
        return dx

def cost(h, y, weights, type, l2 = 1):
    if type not in ["log"]:
        raise ValueError("Loss Type not recognized")
    if type == "log":
        reg_sum = 0
        for weight in weights:
            reg_sum += np.power(weight[1:], 2).sum()
        return -(((xlogy(y, h) + xlogy(1 - y, 1 - h)).sum() / h.shape[0])
                  + (l2 / (2 * h.shape[0])) * reg_sum)
