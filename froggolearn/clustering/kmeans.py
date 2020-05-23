from ..utils import standardize_data, cast_to_3d
import numpy as np
from numpy.linalg import norm

class KMeansClustering:
    def __init__(self, n_centroids=2, max_iter=300, n_inits=10, tol=1e-4):
        assert n_centroids >= 2, "There must be at least 2 cluster centroids"
        self.n_centroids = n_centroids
        self.max_iter = max_iter
        self.n_inits = n_inits
        self.tol = tol
        self.isfit = False

    def _initialize_centroids(self, X, k):
        """Randomly initialize k cluster centroids chosen from elements of X"""
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def _calc_distortion(self, X, labels, centroids):
        distortion = X - np.take(centroids, labels, axis=0)
        distortion = np.power(norm(distortion), 2)
        distortion *= (1 / len(labels))
        return distortion

    def _get_labels(self, X_3d, centroids):
        return np.argmin(np.power(norm(X_3d - centroids.T, axis=1), 2),
                                       axis=1)

    def _move_centroids(self, X, centroids, labels):
        for k in range(self.n_centroids):
            if len(X[labels == k]) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
            else:
                centroids[k] = self._initialize_centroids(X, 1)
        return centroids

    def fit(self, X):
        X = X.copy()
        distortions = np.ones(shape=(self.n_inits))*np.inf
        labels = np.random.randint(0, self.n_centroids, X.shape[0])
        X, _, _ = standardize_data(X)
        X_3d = cast_to_3d(X, self.n_centroids)
        for n in range(self.n_inits):
            centroids = self._initialize_centroids(X, self.n_centroids)
            for _ in range(self.max_iter):
                od = self._calc_distortion(X, labels, centroids)
                labels = self._get_labels(X_3d, centroids)
                centroids = self._move_centroids(X, centroids, labels)
                d = self._calc_distortion(X, labels, centroids)
                if (od - d) < self.tol:
                    break
            distortions[n] = d
            if distortions[n] == min(distortions):
                self.centroids = centroids
                self.distortion = distortions[n]
        self.isfit = True

    def predict(self, X):
        assert self.isfit, "Model must be fit first."
        X_3d = cast_to_3d(X, self.n_centroids)
        return self._get_labels(X_3d, self.centroids)
