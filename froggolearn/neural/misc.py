# Not used at the moment, code artifact

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
            h_pos = _forward_propagate(X, weights = pos_weights)[-1]
            h_neg = _forward_propagate(X, weights = neg_weights)[-1]
            j_pos = cost(h_pos, y, pos_weights)
            j_neg = cost(h_neg, y, neg_weights)
            n_vec[element] = (j_pos - j_neg) / (2 * eps)
        numerical_gradients[k] = n_vec.reshape(weights[k].shape)
    return numerical_gradients
