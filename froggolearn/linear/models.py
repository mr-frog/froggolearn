import numpy as np
from .solvers import gradientdescent, normalequation, lbfgsb
from scipy.special import xlogy
from ..utils.utils import *

__all__ = ["LinearRegression", "LogisticRegression"]

class LinearRegression:
    """
    Least Squares Linear Regression.

    Parameters:
    ---
    penalty : str, default 'l2'

    l_val : float, default 0.1

    fit_intercept : bool, default True

    solver : str, default 'lbfgsb'

    standardize : bool, default True

    Public Methods:
    ---
    fit(X, y)
        Fit the Regressor to a certain training set (X, y)

    predict(X)
        Predict labels based on a test set (X)

    unscale_coef()
        Unscales the coefficents e.g. to compare them with other models
    """

    def __init__(self, penalty='l2', l_val=0.1, fit_intercept=True,
                 solver='lbfgsb', standardize=True):
        if penalty not in ['l2', None]:
            raise ValueError("Only 'l2' or None penalty is supported as of now."
                             " Got %s, sorry :("%penalty)

        self.penalty = penalty
        self.l_val = l_val
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.standardize = standardize

        self.sx, self.mx = 1, 0
        self.sy, self.my = 1, 0
        self.intercept = 0
        self.isfit = False
        self.is_standardized = False

    def _cost_func(self, t, X_v, y_v, penalty, l_val):
        e = len(y_v)
        h = (t @ X_v.T)
        if penalty == 'l2':
            return (1/(2*e) * np.power((h - y_v), 2).sum()
                    + ((l_val / (2 * e)) * np.matmul(t[1:], t[1:])))
        elif penalty == None:
            return (1/(2*e) * np.power((h - y_v), 2).sum())

    def _delta_func(self, t, X_v, y_v, penalty, l_val):
        e = len(y_v)
        h = (t @ X_v.T)
        if penalty == 'l2':
            t2 = t.copy()
            t2[0] = 0
            return ((1 / e) * X_v.T @ (h - y_v) + (l_val/e) * t2)
        elif penalty == None:
            return ((1 / e) * X_v.T @ (h - y_v))

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : matrix (m x n) of np.array type
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        y : vector (m, 1) of np.array type
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """

        X_values, y_values = check_input_type(X, y)

        if self.standardize:
            X_values, sx, mx = standardize_data(X_values)
            y_values, sy, my = standardize_data(y_values)
            self.sx, self.mx = sx, mx
            self.sy, self.my = sy, my
            self.is_standardized = True

        if self.fit_intercept:
            X_values = np.insert(X_values, 0, 1, axis=1)

        theta = np.zeros(shape=(X_values.shape[1]))

        if self.solver == "ne":
            theta = normalequation(X_values, y_values, self.l_val)
        if self.solver == 'gd':
            theta = gradientdescent(self._cost_func, theta, (X_values, y_values,
                                    self.penalty, self.l_val), self.delta_func)
        elif self.solver == 'lbfgsb':
            theta = lbfgsb(self._cost_func, theta, (X_values, y_values,
                           self.penalty, self.l_val), self.delta_func)

        if self.fit_intercept:
            self.intercept = theta[0]
            self.coef = theta[1:]
        else:
            self.coef = theta

        self.isfit = True

    def predict(self, X):
        """
        Predicts Values based on the training fit and input Values X

        Parameters
        ----------
        X : matrix (m x n) of np.array type
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        Returns
        -------
        y : Vector (m x n) of np.array type
            Vector carrying the predicted labels based on X
        """

        if not self.isfit:
            print("Model is not fitted.")
            return

        if not isinstance(X, np.ndarray):
            try:
                X_values = np.array(X)
            except:
                raise TypeError("X is not of type or convertible to type"
                                "np.ndarray. Got %s, sorry :("%(type(X)))
        else:
            X_values = X.copy()

        if self.is_standardized:
            X = np.divide(X - self.mx, self.sx)

        return ((self.coef @ X.T + self.intercept) * self.sy + self.my)

    def unscale_coef(self):
        """Unscales the models coefficents"""
        if not self.isfit:
            print("Model is not fitted.")
            return
        if not self.is_standardized:
            print("Data is not scaled.")
            return

        sx, mx = self.sx, self.mx
        sy, my = self.sy, self.my
        coef = self.coef
        intercept = self.intercept
        coef = coef * (sy/sx)
        self.coef = coef
        if self.fit_intercept:
            intercept = intercept * sy + my - sum(np.divide(coef * sy * mx, sy))
            self.intercept = intercept

        self.is_standardized = False
        self.sx, self.mx = 1, 0
        self.sy, self.my = 1, 0


class LogisticRegression:
    """Fit Classification-Problem using a Logistic Regression approach."""

    def __init__(self, penalty='l2', l_val=1.0, fit_intercept=True,
                 solver='lbfgsb', multi_class='ovr', standardize=True):

        if penalty not in ['l2', None]:
            raise ValueError("Only 'l2' or None penalty is supported as of now."
                             " Got %s, sorry :("%penalty)
        if multi_class != 'ovr':
            raise ValueError("Only One-VS-Rest multi-target classification is"
                             "supported as of now. Got %s, sorry:("%multi_class)

        self.penalty = penalty
        self.l_val = l_val
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.multi_class = multi_class
        self.standardize = standardize

        self.labels = []
        self.sx, self.mx = 1, 0
        self.isfit = False

    def _cost_func(self, t, X_v, y_v, penalty, l_val):
        e = len(y_v)
        h = sigmoid(t@X_v.T)
        if penalty == 'l2':
            return -(((xlogy(y_v, h) + xlogy(1 - y_v, 1 - h)).sum() / h.shape[0])
                          + (l_val / (2 * h.shape[0])) * np.power(t[1:], 2))
            #return (((1 / e) * -y_v @ np.log(h) - (1 - y_v) @ np.log(1 - h))
                    #+ ((l_val / (2 * e)) * np.power(t[1:], 2)))
        elif penalty == None:
            return (((1 / e) * -y_v @ np.log(h) - (1 - y_v) @ np.log(1 - h)))

    def delta_func(self, t, X_v, y_v, penalty, l_val):
        h = sigmoid(t@X_v.T)
        e = len(y_v)
        if penalty == 'l2':
            t2 = t.copy()
            t2[0] = 0
            return ((1 / e) * X_v.T @ (h - y_v) + (l_val/e) * t2)
        elif penalty == None:
            return ((1 / e) * X_v.T @ (h - y_v))

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : matrix (m x n) of np.array type
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        y : vector (m, 1) of np.array type
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        X_values, y_values = check_input_type(X, y)

        if self.standardize:
            X_values, sx, mx = standardize_data(X_values)
            self.sx, self.mx = sx, mx

        if self.fit_intercept:
            X_values = np.insert(X_values, 0, 1, axis=1)

        theta = np.zeros(shape=(X_values.shape[1]))

        if self.multi_class == 'ovr':
            labels = np.unique(y_values)
            thetas = np.zeros(shape=(len(labels), X_values.shape[1]))

            for i in range(len(labels)):
                label = labels[i]
                mask = (y_values == label)
                y_multi = np.ones_like(y_values)
                y_multi[~mask] = 0
                y_multi = y_multi.astype('float64')
                if self.solver == 'gd':
                    theta = gradientdescent(self._cost_func, theta, (X_values,
                                            y_multi, self.penalty, self.l_val),
                                            self.delta_func)
                elif self.solver == 'lbfgsb':
                    theta = lbfgsb(self._cost_func, theta, (X_values, y_multi,
                                   self.penalty, self.l_val), self.delta_func)
                else:
                    raise ValueError("Solver not supported. "
                                     "Got %s, must be 'gd' or 'lbfgsb'"
                                     %self.solver)
                thetas[i] = theta
            self.coef = thetas
            self.labels = labels
            self.isfit = True

    def predict(self, X):
        """
        Predicts Values based on the training fit and input Values X

        Parameters
        ----------
        X : matrix (m x n) of np.array type
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        Returns
        -------
        y : Vector (m x n) of np.array type
            Vector carrying the predicted labels based on X
        """
        if not self.isfit:
            print("Model is not fitted.")
            return
        if not isinstance(X, np.ndarray):
            try:
                X_values = np.array(X)
            except:
                raise TypeError("X is not of type or convertible to type"
                                "np.ndarray. Got %s, sorry :("%(type(X)))
        else:
            X_values = X.copy()
        if self.standardize:
            X_values = np.divide(X_values - self.mx, self.sx)
        if self.fit_intercept:
            X_values = np.insert(X_values, 0, 1, axis=1)
        y_predict = self.labels[np.argmax(sigmoid(self.coef @ X_values.T),
                                          axis = 0)]
        return y_predict
