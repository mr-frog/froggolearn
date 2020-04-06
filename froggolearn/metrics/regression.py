import pandas as pd
import numpy as np
from ..utils.utils import check_input_dims, check_input_type

__all__ = ["r2_score", "mean_squared_error"]

def r2_score(y_true, y_predict):
    """
    Returns r2 score for given data.
    """
    y_true_values, y_predict_values = check_input_type(y_true, y_predict)
    check_input_dims(y_true_values, y_predict_values, 1)
    SQE = ((y_true_values - y_predict_values)**2).sum(axis=0)
    SQT = ((y_true_values - np.average(y_true_values))**2).sum(axis=0)
    r2_sco = 1 - (SQE/SQT)
    return r2_sco

def mean_squared_error(y_true, y_predict):
    """
    Returns mean squared error for given data.
    """
    y_true_values, y_predict_values = check_input_type(y_true, y_predict)
    check_input_dims(y_true_values, y_predict_values, 1)
    return np.average((y_true_values - y_predict_values)**2, axis = 0)
