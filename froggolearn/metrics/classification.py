import pandas as pd
import numpy as np
from ..utils.utils import check_data

__all__ = ["score_matrix", "confusion_matrix", "accuracy_score"]

def accuracy_score(y_true, y_predict):
    check_data(y_true, y_predict)

    y_true_values = y_true.values.reshape(-1)
    y_predict_values = y_predict.values.reshape(-1)
    TP = 0
    for label in np.unique(y_true_values):

        correct_predict = np.count_nonzero(np.logical_and(y_predict_values == label, y_predict_values == y_true_values))
        TP += correct_predict
    acc_score = TP / len(y_true)
    return acc_score

def score_matrix(y_true, y_predict, beta = 1):
    check_data(y_true, y_predict)

    names = ['Label', 'Precision', 'Recall', 'f1score']
    sco_ma = pd.DataFrame(columns=names)
    y_true_values = y_true.values.reshape(-1)
    y_predict_values = y_predict.values.reshape(-1)

    for label in np.unique(y_true_values):

        actual = np.count_nonzero(y_true_values == label)
        total_predict = np.count_nonzero(y_predict_values == label)
        correct_predict = np.count_nonzero(np.logical_and(y_predict_values == label, y_predict_values == y_true_values))

        precision = correct_predict / total_predict
        recall = correct_predict / actual
        f1 = (1+beta**2) * (precision * recall)/(beta**2 * precision + recall)

        row = pd.DataFrame([[label, precision, recall, f1]], columns = names)
        sco_ma = sco_ma.append(row, ignore_index=True)

    return sco_ma

def confusion_matrix(y_true, y_predict):
    check_data(y_true, y_predict)

    y_true = pd.Series(y_true.values.reshape(-1), name = 'True')
    y_predict = pd.Series(y_predict.values.reshape(-1), name = 'Predicted')
    return pd.crosstab(y_true, y_predict, margins = True)
