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
    elif type == "relu":
        ax = np.clip(X, 0, np.inf)
        return ax
    elif type == "leaky":
        a = 0.01
        ax = X
        ax[X < 0] = X[X < 0] * a
        return ax
    elif type == "elu":
        a = 0.01
        ax = X
        ax[X < 0] = (np.exp(X[X < 0]) - 1) * a
        return ax
    elif type == "softmax":
        if len(X.shape) != 1:
            ax = X - X.max(axis=1)[:, np.newaxis]
            ax = np.exp(ax)
            ax = ax / ax.sum(axis=1)[:, np.newaxis]
        else:
            ax = X - X.max()
            ax = np.exp(ax)
            ax = ax / ax.sum()
        return ax
    else:
        return X

def derivative(X, type):
    check_type(type)
    if type == "log":
        return X * (1 - X)
    elif type == "relu":
        dx = 1 * (X > 0)
        return dx
    elif type == "leaky":
        a = 0.01
        dx = np.ones_like(X)
        dx[X < 0] = a
        return dx
    elif type == "elu":
        a = 0.01
        dx = np.ones_like(X)
        dx[X < 0] = X[X < 0] + a
        return dx
    else:
        return 1

def cost(h, y, weights, type, l2 = 1):
    if type not in ["log"]:
        raise ValueError("Loss Type not recognized")
    if type == "log":
        reg_sum = 0
        for weight in weights:
            reg_sum += np.power(weight[1:], 2).sum()
        return -(((xlogy(y, h) + xlogy(1 - y, 1 - h)).sum() / h.shape[0])
                  + (l2 / (2 * h.shape[0])) * reg_sum)
