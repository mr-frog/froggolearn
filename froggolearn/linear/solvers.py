import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


__all__ = ['normalequation', 'gradientdescent', 'lbfgsb']

### Function minimizers:

def lbfgsb(cost_func, x0, args=(), delta_func = 0):
    """
    Solves for theta using scipys implementation of L-BFGS-B
    """

    return minimize(fun = cost_func, x0 = x0, args = args,
                    method = 'L-BFGS-B', jac = delta_func).x

def gradientdescent(cost_func, theta, args=(), delta_func = 0):
    """
    Minimizes a cost_function using an iterative gradient descent approach.
    """

    step = 1
    old_cost = 0
    while True:
        theta_old = theta.copy()
        cost = cost_func(theta, *args)
        delta = delta_func(theta, *args)
        theta = theta - step * delta
        if cost > old_cost and old_cost != 0:
            step = step*0.7
        if np.allclose(theta_old, theta):
            break
        old_cost = cost
    return theta

### Numerical solvers:

def normalequation(X_values, y_values, l_val = 0):
    """
    Solves a linear regression problem for theta using the normal equations approach.
    """

    l_matrix = np.eye(N = X_values.shape[1])
    l_matrix[0][0] = 0
    l_matrix = l_matrix * l_val
    return np.linalg.pinv(X_values.T @ X_values + l_matrix) @ X_values.T @ y_values
