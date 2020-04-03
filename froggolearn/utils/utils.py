import numpy as np
import pandas as pd
from scipy.special import expit

def standardize_data(data_matrix):
    """returns (X-mean(X))/std(X), stdev and mean vector"""
    s = np.std(data_matrix, axis=0)
    m = np.mean(data_matrix, axis=0)
    return np.divide(data_matrix - m, s), s, m

def sigmoid(x):
    return expit(x)

def check_data(d1, d2, dim = 1):
    if not isinstance(d1, pd.DataFrame) or not isinstance(d2, pd.DataFrame):
        raise TypeError("Both input vectors must be of pd.DataFrame type.")

    if dim == 1:
        if not d1.shape[1] == 1 or not d2.shape[1] == 1:
            raise ValueError("Input must be 1-dimensional vectors.")

    if not d1.shape == d2.shape:
        raise ValueError("Both input vectors must have the same dimensions.")
