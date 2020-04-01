import numpy as np
from scipy.special import expit

def standardize_data(data_matrix):
    """returns (X-mean(X))/std(X), stdev and mean vector"""
    s = np.std(data_matrix, axis=0)
    m = np.mean(data_matrix, axis=0)
    return np.divide(data_matrix - m, s), s, m

def sigmoid(x):
    return expit(x)
