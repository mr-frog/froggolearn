import numpy as np
from .activations import activation, derivative, cost
from ..utils.utils import shuffle, bias, check_input_type, standardize_data
from ..metrics import accuracy_score
def make_topology(X, y, layers, neurons, type, random_state):
    """
    Creates a list of weight-matrices corresponding to the given NN-Topology.
    Uses the initialization method recommended by Glorot et al.

    Input
    ---
    X :


    y :


    layers :


    nerons :

    type :


    random_state
    """
    if isinstance(random_state, int):
        np.random.seed(int(random_state))
    factor = 6.
    if type == "log":
        factor = 2.
    weights = []
    for layer in range(len(neurons)):
        if layer == 0:
            layer_shape = (X.shape[1] + 1, neurons[layer])
            bounds = np.sqrt(factor / (layer_shape[0] + layer_shape[1]))
            weights.append(np.random.uniform(-bounds, bounds, layer_shape))
        if layer == max(range(len(neurons))):
            layer_shape = (neurons[layer] + 1, y.shape[1])
            bounds = np.sqrt(factor / (layer_shape[0] + layer_shape[1]))
            weights.append(np.random.uniform(-bounds, bounds, layer_shape))
        else:
            layer_shape = (neurons[layer] + 1, neurons[layer + 1])
            bounds = np.sqrt(factor / (layer_shape[0] + layer_shape[1]))
            weights.append(np.random.uniform(-bounds, bounds, layer_shape))
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
            j_pos = cost(h_pos, y, pos_weights)
            j_neg = cost(h_neg, y, neg_weights)
            n_vec[element] = (j_pos - j_neg) / (2 * eps)
        numerical_gradients[k] = n_vec.reshape(weights[k].shape)
    return numerical_gradients

class NNClassifier():
    def __init__(self, n_neurons=(100,), n_iter=100, type="relu", solver="sgd",
                 batchsize=200, rate=0.01, l2=0.0001, verbose=False, bias=1,
                 random_state = False):
        self.neurons = n_neurons
        self.layers = len(self.neurons) + 2
        self.n_iter = n_iter
        self.type = type
        self.solver = solver
        self.batchsize = batchsize
        self.rate = rate
        self.l2 = l2
        self.verbose = verbose
        self.bias = bias
        self.random_state = random_state

    def forward_propagate(self, X, weights):
        """
        Forward propagate over given batch of samples.

        Input
        ---
        X : matrix (dimensions = m, n)
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        weights : list (lengths = layers - 1)
            The i-th element of the list corresponds to the matrix representing
            the transition from the i-th to the i + 1-th layer of the network.

        Output
        ---
        activations : list (lengths = layers)
            The i-th element of the list corresponds to the activation of the
            i-th layer of the network.
        """
        activations = []
        for layer in range(len(weights) + 1):
            if layer == 0:
                act = X
                activations.append(act)
            else:
                act = activation((bias(activations[layer - 1], self.bias) @
                                        weights[layer - 1]), self.type)
                activations.append(act)
        return activations

    def backward_propagate(self, X, y, weights):
        """
        Backward propagate over given batch of samples.

        Input
        ---
        X : matrix (dimensions = m, n)
            Training matrix, with dimensions m = n_samples and
            n = n_features.

        y : vector (dimensions = m, 1)
            Target vector relative to X, with dimensions m = n_samples and 1.

        weights : list (lengths = layers - 1)
            The i-th element of the list corresponds to the matrix representing the
            transition from the i-th to the i + 1-th layer of the network

        Output
        ---
        gradients : list (lengths = layers - 1)
            The i-th element of the list corresponds to the gradient with which the
            i-th element in the weights list should be updated
        """
        # Initialize necessary arrays
        deltas = [0] * len(weights)
        gradients = [0] * len(weights)
        errors = [0] * (len(weights) + 1)
        # Iterate over all samples
        for i in range(X.shape[0]):
            # forward propagate to obtain the current hypothesis of the network
            activations = self.forward_propagate(X[i, :], weights)
            # backward propagate iteratively over the layers
            for j in reversed(range(len(activations))):
                if j == max(range(len(activations))):
                    errors[j] = activations[j] - y[i]
                    sliced_e = errors[j][np.newaxis]
                    sliced_a = bias(activations[j - 1], self.bias)[np.newaxis]
                    deltas[j - 1] = deltas[j - 1] + sliced_a.T @ sliced_e
                elif j != 0:
                    errors[j] = (errors[j + 1] @ weights[j].T[:, 1:]
                                 * derivative(activations[j], self.type))
                    sliced_e = errors[j][np.newaxis]
                    sliced_a = bias(activations[j - 1], self.bias)[np.newaxis]
                    deltas[j - 1] = deltas[j - 1] + sliced_a.T @ sliced_e
        # update and return the gradients
        for l in range(len(weights)):
            gradients[l] = deltas[l].copy()
            gradients[l][:1] = deltas[l][:1] / X.shape[0]
            gradients[l][1:] = deltas[l][1:] / X.shape[0] + self.l2 * weights[l][1:]
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
        self.weights = make_topology(X, y, self.layers, self.neurons, self.type,
                                     self.random_state)
        X, _, _ = standardize_data(X)
        batchsize = min(self.batchsize, X.shape[0])
        for i in range(self.n_iter):
            if self.verbose:
                acc = np.round(accuracy_score(self.predict(X), y_test), 3)
                print("\r%s/%s, [%s]   "%(i, self.n_iter, acc), end ='')
            X_sh, y_sh = shuffle(X, y)
            if (X.shape[0] % batchsize) != 0:
                n_batches = range(int(np.floor(X.shape[0] / batchsize) + 1))
            else:
                n_batches = range(int(np.floor(X.shape[0] / batchsize)))
            for n in n_batches:

                batchstart = n * batchsize
                batchend = min((n + 1) * batchsize, X.shape[0])
                batch = slice(batchstart, batchend)
                gradients = self.backward_propagate(X_sh[batch], y_sh[batch],
                                                    self.weights)
                for l in range(len(self.weights)):
                    self.weights[l] = self.weights[l] - self.rate * gradients[l]
        if self.verbose:
            print("\n")

    def predict(self, X_val):
        weights = self.weights
        X = X_val.copy()
        y_pred = self.forward_propagate(X, weights)
        return self.classes[np.argmax(y_pred[-1].T, axis = 0)]
