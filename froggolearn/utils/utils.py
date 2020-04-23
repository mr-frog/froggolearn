import numpy as np
from scipy.special import expit

def standardize_data(data_matrix):
    """returns (X-mean(X))/std(X), stdev and mean vector"""
    s = np.std(data_matrix, axis=0)
    if isinstance(s, np.ndarray):
        for i in range(len(s)):
            if s[i] == 0:
                s[i] = 1
    elif s == 0:
        s = 1
    m = np.mean(data_matrix, axis=0)
    return np.divide(data_matrix - m, s), s, m

def shuffle(X, y):
    assert X.shape[0] == y.shape[0], ("X and y must be of same length,"
                                      "got %s and %s"%(X.shape[0], y.shape[0]))
    shuffled_X = np.empty(X.shape, dtype=X.dtype)
    shuffled_y = np.empty(y.shape, dtype=y.dtype)
    permutation = np.random.permutation(X.shape[0])
    for old_index, new_index in enumerate(permutation):
        shuffled_X[new_index] = X[old_index]
        shuffled_y[new_index] = y[old_index]
    return shuffled_X, shuffled_y

def sigmoid(x):
    """returns sigmoid function based on scipys expit function"""
    return expit(x)

def check_input_dims(d1, d2, dim = 1):
    """
    Checks if input dimensions are appropriate
    """

    if dim == 1:
        if not len(d1.shape) == 1:
            if d1.shape[1] == 1:
                d1 = d1.reshape(-1)
        if not len(d2.shape) == 1:
            if d2.shape[1] == 1:
                d2 = d2.reshape(-1)
        if not len(d1.shape) == 1 or not len(d2.shape) == 1:
            raise ValueError("Input must be %s-dimensional vectors."
            " Got %s and %s"%(dim, d1.shape, d2.shape))
    if not d1.shape == d2.shape:
        raise ValueError("Both input vectors must have the same dimensions.")
    return d1, d2

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

def bias(X, bias=1, axis=1):
    """
    Adds bias unit to input vector/matrix
    """
    if len(X.shape) == 1:
        axis = 0
    return np.insert(X, 0, 1, axis = axis)

class LabelEncoder:

    def __init__(self):
        pass
