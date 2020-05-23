import numpy as np
from numpy.linalg import svd
from ..utils import standardize_data


class PrincipalComponentAnalysis:
    def __init__(self, n_components=2, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit_transform(self, X):
        X = X.copy()

        #Center Data
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # Perform singular value decomposition
        U, S, V = svd(X, full_matrices=False)

        # Flip Signs
        abs_max = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[abs_max, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]

        components = V
        variance = (S ** 2) / (X.shape[0] - 1)

        self.components = components[:self.n_components]
        self.variance = variance[:self.n_components]

        U = U[:, :self.n_components]

        if self.whiten:
            U *= np.sqrt(X.shape[0] - 1)
        else:
            U *= S[:self.n_components]
        # Return transformed Data
        return U
