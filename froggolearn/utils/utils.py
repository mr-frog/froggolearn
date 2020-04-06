import numpy as np
import pandas as pd
from scipy.special import expit

def standardize_data(data_matrix):
    """returns (X-mean(X))/std(X), stdev and mean vector"""
    s = np.std(data_matrix, axis=0)
    m = np.mean(data_matrix, axis=0)
    return np.divide(data_matrix - m, s), s, m

def sigmoid(x):
    """returns sigmoid function based on scipys expit function"""
    return expit(x)

def check_input_dims(d1, d2, dim = 1):
    """
    Checks if input dimensions are appropriate
    """
    if dim == 1:
        if not len(d1.shape) == 1 or not len(d2.shape) == 1:
            raise ValueError("Input must be %s-dimensional vectors."%dim)

    if not d1.shape == d2.shape:
        raise ValueError("Both input vectors must have the same dimensions.")

def check_input_type(*inps):
    """
    Checks type and values of input and converts to np.ndarray if necessary.
    """
    inp_values_list = []
    for inp in inps:
        if not isinstance(inp, np.ndarray):
            try:
                inp_values = np.array(inp)
                if inp_values.shape[1] == 1:
                    inp_values = inp_values.reshape(-1)
            except:
                raise TypeError("Input is not of type or convertible to type"
                                "np.ndarray. Got %s and %s, sorry :("%type(inp))
        else:
            inp_values = inp.copy()
        inp_values_list.append(inp_values)
    return inp_values_list

class LabelEncoder:

    def __init__(self):
        pass
