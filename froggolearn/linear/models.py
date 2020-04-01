import pandas as pd
import numpy as np
from .algorithms import gradientdescent, normalequation, bfgs
from ..utils import standardize_data, sigmoid

from matplotlib import pyplot as plt

__all__ = ["LinearRegression", "LogisticRegression"]

class LinearRegression:
    """Fit Data using LinearRegression. Implements either Gradient Descent
       or Normal Equation."""

    def __init__(self):
        self.coef = []
        self.type = ""
        self.intercept = 0
        self.sx, self.mx = 1, 0
        self.sy, self.my = 1, 0
        self.isfit = False
        self.isscaled = False
        self.intercept_fit = False

    def cost_func(self, theta, X_values, y_values):
        elements = len(y_values)
        predict = (theta @ X_values.T)
        return 1/(2*elements) * np.power((predict - y_values), 2).sum()

    def delta_func(self, theta, X_values, y_values):
        predict = (theta @ X_values.T)
        elements = len(y_values)
        return 1/elements * ((predict - y_values)).T @ X_values

    def fit(self, X, y, mode='ne', fit_intercept=True, standardize=True):
        """Mode can be ne for Normal Equation or gd for Gradient Descent"""

        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise TypeeError("Input X and y must be pd.DataFrame type.")

        X_values = X.values

        if y.shape[1] == 1:
            y_values = y.values.reshape(-1)
        else:
            raise ValueError("Only 1-dimensional targetvectors are supported.")

        if standardize:
            X_values, sx, mx = standardize_data(X_values)
            y_values, sy, my = standardize_data(y_values)
            self.sx, self.mx = sx, mx
            self.sy, self.my = sy, my
            self.isscaled = True

        if fit_intercept:
            X_values = np.insert(X_values, 0, 1, axis=1)
            self.intercept_fit = True
        theta = np.zeros(shape=(X_values.shape[1]))

        if mode == "ne":
            type = "Normal Equation"
            theta = normalequation(X_values, y_values)

        elif mode == "gd":
            type = "Gradient Descent"
            theta = gradientdescent(X_values, y_values, self.cost_func, self.delta_func)

        elif mode == "bfgs":
            type = "BFGS"
            theta = bfgs(func = self.cost_func, x0 = theta, args=(X_values, y_values), jac = self.delta_func)

        if fit_intercept:
            self.intercept = theta[0]
            self.coef = theta[1:]
        else:
            self.coef = theta

        self.type = type
        self.isfit = True

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be pd.DataFrame type.")
        if not self.isfit:
            print("Model is not fitted.")
            return
        if self.isscaled:
            X = np.divide(X - self.mx, self.sx)
        return ((self.coef @ X.T + self.intercept) * self.sy + self.my)

    def unscale_coef(self):
        if not self.isfit:
            print("Model is not fitted.")
            return
        if not self.isscaled:
            print("Data is not scaled.")
            return

        sx, mx = self.sx, self.mx
        sy, my = self.sy, self.my

        if self.intercept_fit:
            coef = self.coef
            intercept = self.intercept
            coef = coef * (sy/sx)
            intercept = intercept * sy + my - sum(np.divide(coef * sy * mx, sy))
            self.intercept = intercept
            self.coef = coef
        else:
            coef = coef * (sy/sx)
            self.coef = coef
        self.isscaled = False


class LogisticRegression:
    """Fit Classification-Problem using a Logistic Regression approach."""

    def __init__(self):
        self.coef = []
        self.type = ""
        self.intercept = 0
        self.sx, self.mx = 1, 0
        self.isfit = False
        self.isscaled = False

    def cost_func(self, theta, X_values, y_values):
        elements = len(y_values)
        hypothesis = sigmoid(theta@X_values.T)
        return 1 / (elements) * -y_values @ np.log(hypothesis) - (1 - y_values) @ np.log(1 - hypothesis)

    def delta_func(self, theta, X_values, y_values):
        hypothesis = sigmoid(theta@X_values.T)
        elements = len(y_values)
        return (1 / elements) * X_values.T @ (hypothesis - y_values)

    def fit(self, X, y, standardize=True, mode='bfgs'):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise TypeError("Input X and y must be pd.DataFrame type.")

        X_values = X.values
        if y.shape[1] == 1:
            y_values = y.values.reshape(-1)
        else:
            raise ValueError("Only 1-dimensional targetvectors are supported.")

        if standardize:
            X_values, sx, mx = standardize_data(X_values)
            self.sx, self.mx = sx, mx
            self.isscaled = True

        X_values = np.insert(X_values, 0, 1, axis=1)
        theta = np.zeros(shape=(X_values.shape[1]))
        if mode == 'gd':
            theta = gradientdescent(X_values, y_values, self.cost_func, self.delta_func)
        elif mode == 'bfgs':
            theta = bfgs(func = self.cost_func, x0 = theta, args=(X_values, y_values), jac = self.delta_func)

        self.coef = theta
        self.isfit = True

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be pd.DataFrame type.")
        if not self.isfit:
            print("Model is not fitted.")
            return
        if self.isscaled:
            X = np.divide(X - self.mx, self.sx)
        X.insert(0, 'x0', 1)
        return np.around(sigmoid(self.coef @ X.T))
