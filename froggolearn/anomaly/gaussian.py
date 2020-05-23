import numpy as np
from numpy.linalg import det, slogdet, pinv
from scipy.stats import multivariate_normal
import sys
class GaussianAnomalyDetection:
    def __init__(self, cutoff='auto'):
        """
        Simple Gaussian Anomaly Detection System
        If cutoff = 'auto' uses some very simple math to decide the cutoff for
        labeling outliers.
        """
        self.cutoff = cutoff

    def detect(self, X):
        """
        Returns a label for each feature vector in X, where 1 indicates
        an outlier, and 0 a non-outlier.
        """
        X = X.copy()
        mu = np.mean(X, axis=0)
        sig = np.var(X, axis=0)
        pre_exp = (1 / (np.sqrt(2 * np.pi) * sig))
        exp = np.exp(-(X - mu)**2 / (2 * sig**2))
        prob = np.prod(pre_exp*exp, axis=1)
        if self.cutoff == 'auto':
            self.cutoff = np.mean(prob) - np.std(prob)
        return np.array([prob < self.cutoff])

class MultivariateGaussian:
    def __init__(self, cutoff='auto'):
        self.cutoff = cutoff

    def detect(self, X):
        m = X.shape[0]
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X.T)
        _prob = multivariate_normal(mean=mu, cov=Sigma)
        prob = _prob.pdf(X)
        if self.cutoff == 'auto':
            self.cutoff = np.mean(prob) - np.std(prob)
        return np.array([prob < self.cutoff])
