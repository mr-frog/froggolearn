import numpy as np
from .activations import activation, derivative, cost
from ..utils.utils import shuffle, bias, check_input_type, standardize_data

def make_topology(X, y, layers, neurons):
    """
    Creates a list of weight-matrices corresponding to the given NN-Topology.
    """
    weights = []
    for layer in range(len(neurons)):
        if layer == 0:
            layer_shape = (X.shape[1] + 1, neurons[layer])
            weights.append(2 * np.random.random(layer_shape) - 1)
        if layer == max(range(len(neurons))):
            layer_shape = (neurons[layer] + 1, y.shape[1])
            weights.append(2 * np.random.random(layer_shape) - 1)
        else:
            layer_shape = (neurons[layer] + 1, neurons[layer + 1])
            weights.append(2 * np.random.random(layer_shape) - 1)
    return weights

def calc_num_gradient(X, y, weights):
    eps = 1e-4
    numerical_gradients = weights.copy()
    for k in range(len(weights)):
        pos_weights = weights.copy()
        neg_weights = weights.copy()
        w_vec = weights[k].copy().ravel()
        n_vec = np.zeros_like(w_vec)
        for element in range(len(w_vec)):
            pos = w_vec.copy()
            neg = w_vec.copy()
            pos[element] = float(pos[element]) + eps
            neg[element] = float(neg[element]) - eps
            pos_weights[k] = pos.reshape(weights[k].shape)
            neg_weights[k] = neg.reshape(weights[k].shape)
            h_pos = forward_propagate(X, weights = pos_weights)[-1]
            h_neg = forward_propagate(X, weights = neg_weights)[-1]
            j_pos = cost_func(h_pos, y, pos_weights)
            j_neg = cost_func(h_neg, y, neg_weights)
            n_vec[element] = (j_pos - j_neg) / (2 * eps)
        numerical_gradients[k] = n_vec.reshape(weights[k].shape)
    return numerical_gradients

class NNClassifier():
    def __init__(self, n_neurons, n_iter = 100, type = "log", solver = "sgd",
                 batchsize = 200, rate = 0.5, l2 = 0.0001, verbose = False):
        self.neurons = n_neurons
        self.layers = len(self.neurons) + 2
        self.n_iter = n_iter
        self.type = type
        self.solver = solver
        self.batchsize = batchsize
        self.rate = rate
        self.l2 = l2
        self.verbose = verbose

    def forward_propagate(self, X, weights):
        activations = []
        for layer in range(len(weights) + 1):
            if layer == 0:
                act = X
                activations.append(act)
            else:
                act = activation(np.dot(bias(activations[layer - 1]),
                                        weights[layer - 1]), self.type)
                activations.append(act)
        return activations

    def backward_propagate(self, X, y, weights):
        """
        Backward propagate over given batch of samples.

        Parameters
        ----------
        X : matrix (m x n)
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        y : vector (m, 1)
            Target vector relative to X.

        weights : list (lengths = layers - 1)
            The i-th element of the list corresponds to the matrix representing the
            transition from the i-th to the i + 1-th layer of the network

        Returns
        -------
        Gradients : list (lengths = layers - 1)
            The i-th element of the list corresponds to the gradient with which the
            i-th element in the weights list should be updated
        """
        deltas = [0] * len(weights)
        gradients = [0] * len(weights)
        errors = [0] * (len(weights) + 1)
        for i in range(X.shape[0]):
            activations = self.forward_propagate(X[i, :], weights)
            for j in reversed(range(len(activations))):
                if j == max(range(len(activations))):
                    errors[j] = activations[j] - y[i]
                    sliced_e = errors[j].copy()[np.newaxis]
                    sliced_a = bias(activations[j - 1]).copy()[np.newaxis]
                    deltas[j - 1] = deltas[j - 1] + np.dot(sliced_a.T, sliced_e)
                elif j != 0:
                    errors[j] = (np.dot(errors[j + 1], weights[j].T[:, 1:])
                                 * derivative(activations[j], "log"))
                    sliced_e = errors[j].copy()[np.newaxis]
                    sliced_a = bias(activations[j - 1]).copy()[np.newaxis]
                    deltas[j - 1] = deltas[j - 1] + np.dot(sliced_a.T, sliced_e)
        for l in range(len(weights)):
            gradients[l] = deltas[l].copy()
            gradients[l][:1] = deltas[l][:1] / X.shape[0]
            gradients[l][1:] = (np.divide(deltas[l][1:], X.shape[0])
                                + np.dot(self.l2, weights[l][1:]))
        del deltas
        del errors
        return gradients

    def fit(self, X_val, y_val):
        X, y = X_val.copy(), y_val.copy()
        X, y = check_input_type(X, y)
        self.classes = np.unique(y)
        y_test = y
        y_multi = np.zeros(shape=(y.shape[0], len(self.classes)))
        for iter in range(len(self.classes)):
            y_multi[:, iter] = (y == self.classes[iter]).ravel()
        y = y_multi
        weights = make_topology(X, y, self.layers, self.neurons)
        X, _, _ = standardize_data(X)
        batchsize = min(self.batchsize, X.shape[0])
        for i in range(self.n_iter):

            X_sh, y_sh = shuffle(X, y)
            if (X.shape[0] % batchsize) != 0:
                n_batches = range(int(np.floor(X.shape[0] / batchsize) + 1))
            else:
                n_batches = range(int(np.floor(X.shape[0] / batchsize)))
            for n in n_batches:
                if self.verbose:
                    print("\r%s/%s, (%s/%s)   "%(i, self.n_iter, n, max(n_batches)), end ='')
                batchstart = n * batchsize
                batchend = min((n + 1) * batchsize, X.shape[0])
                batch = slice(batchstart, batchend)
                gradients = self.backward_propagate(X_sh[batch], y_sh[batch],
                                                    weights)
                for l in range(len(weights)):
                    weights[l] = weights[l] - self.rate * gradients[l]
        if self.verbose:
            print("\n\n")
        self.weights = weights
    def predict(self, X_val):
        weights = self.weights
        X = X_val.copy()
        y_pred = self.forward_propagate(X, weights)
        return self.classes[np.argmax(y_pred[-1].T, axis = 0)]
