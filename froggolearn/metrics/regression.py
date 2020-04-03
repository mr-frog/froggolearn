import pandas as pd
import numpy as np
from ..utils.utils import check_data

__all__ = ["r2_score", "mean_squared_error"]

def r2_score(y_true, y_predict):
    check_data(y_true, y_predict)
    y_true_values = y_true.values.reshape(-1)
    y_predict_values = y_predict.values.reshape(-1)
    SQE = ((y_true_values - y_predict_values)**2).sum(axis=0)
    SQT = ((y_true_values - np.average(y_true_values))**2).sum(axis=0)
    r2_sco = 1 - (SQE/SQT)
    return r2_sco

def mean_squared_error(y_true, y_predict):
    check_data(y_true, y_predict)
    y_true_values = y_true.values.reshape(-1)
    y_predict_values = y_predict.values.reshape(-1)

    return np.average((y_true_values - y_predict_values)**2, axis = 0)
