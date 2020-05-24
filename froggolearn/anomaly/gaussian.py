import numpy as np
from scipy.stats import multivariate_normal

class GaussianAnomalyDetection:
    def __init__(self, cutoff='auto'):
        """
        Simple Gaussian Anomaly Detection System
        If cutoff = 'auto' uses some very simple math to decide the cutoff for
        labeling outliers.
        """
        self.cutoff = cutoff

    def fit(self, X):
        """
        Fits the parameters of a gaussian distribution describing the
        dataset X.
        """
        self.mu = np.mean(X, axis=0)
        self.sigma = np.var(X, axis=0)

    def predict(self, X):
        """
        Returns labels [1, 0] for [outlier, inlier].
        """
        prob = self.probability(X)
        if self.cutoff == 'auto':
            self.cutoff = np.mean(prob) - 2 * np.std(prob)
        return np.array([prob < self.cutoff])

    def probability(self, X):
        """
        Returns the probability of occurance in the gaussian distribution
        for each element in a given dataset X.
        """
        pre_exp = (1 / (np.sqrt(2 * np.pi) * self.sigma))
        exp = np.exp(-(X - self.mu)**2 / (2 * self.sigma**2))
        return np.prod(pre_exp*exp, axis=1)


class MultivariateGaussian:
    def __init__(self, cutoff='auto'):
        """
        Simple Gaussian Anomaly Detection System
        If cutoff = 'auto' uses some very simple math to decide the cutoff for
        labeling outliers.
        """
        self.cutoff = cutoff

    def fit(self, X):
        """
        Fits the parameters of a multivariate gaussian distribution describing
        the dataset X using scipy.stats.multivariate_normal.
        """
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X.T)
        self.mvg = multivariate_normal(mean=mu, cov=Sigma)

    def predict(self, X):
        """
        Returns labels [1, 0] for [outlier, inlier].
        """
        prob = self.mvg.pdf(X)
        if self.cutoff == 'auto':
            self.cutoff = np.mean(prob) - 2 * np.std(prob)
        return np.array([prob < self.cutoff])

    def probability(self, X):
        """
        Returns the probability of occurance in the gaussian distribution
        for each element in a given dataset X.
        """
        return self.mvg.pdf(X)
