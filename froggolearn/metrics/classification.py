import pandas as pd
import numpy as np
from ..utils.utils import check_input_dims, check_input_type

__all__ = ["score_matrix", "confusion_matrix", "accuracy_score"]

def accuracy_score(y_true, y_predict):
    """
    Returns accuracy score for given data.
    """
    y_true_values, y_predict_values = check_input_type(y_true, y_predict)
    check_input_dims(y_true_values, y_predict_values, 1)
    TP = 0
    for label in np.unique(y_true_values):

        correct_predict = (np.count_nonzero(
                            np.logical_and(y_predict_values == label,
                                           y_predict_values == y_true_values)))
        TP += correct_predict
    acc_score = TP / len(y_true)
    return acc_score

def score_matrix(y_true, y_predict, beta = 1):
    """
    Returns Matrix carrying Precision, Recall and f1score for all labels
    in given data.
    """
    y_true_values, y_predict_values = check_input_type(y_true, y_predict)
    check_input_dims(y_true_values, y_predict_values, 1)

    names = ['Label', 'Precision', 'Recall', 'f1score']
    sco_ma = pd.DataFrame(columns=names)

    for label in np.unique(y_true_values):
        actual = np.count_nonzero(y_true_values == label)
        total_predict = np.count_nonzero(y_predict_values == label)
        correct_predict = (np.count_nonzero(
                            np.logical_and(y_predict_values == label,
                                           y_predict_values == y_true_values)))
        if total_predict != 0:
            precision = correct_predict / total_predict
        else:
            precision = np.finfo(float).eps
        recall = correct_predict / actual
        f1 = (1+beta**2) * (precision * recall)/(beta**2 * precision + recall)

        row = pd.DataFrame([[label, precision, recall, f1]], columns = names)
        sco_ma = sco_ma.append(row, ignore_index=True)

    return sco_ma

def confusion_matrix(y_true, y_predict):
    """
    Returns Cofusion Matrix for given data.
    """
    y_true_values, y_predict_values = check_input_type(y_true, y_predict)
    check_input_dims(y_true_values, y_predict_values, 1)

    y_true = pd.Series(y_true_values, name = 'True')
    y_predict = pd.Series(y_predict_values, name = 'Predicted')
    return pd.crosstab(y_true, y_predict, margins = True)
