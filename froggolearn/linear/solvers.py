import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


__all__ = ['normalequation', 'gradientdescent', 'lbfgsb']

### Function minimizers:

def lbfgsb(cost_func, x0, args=(), delta_func = ''):
    """Solves for theta using scipys implementation of BFGS"""
    return minimize(fun = cost_func, x0 = x0, args = args,
                    method = 'L-BFGS-B', jac = delta_func).x

def gradientdescent(X_values, y_values, cost_func, delta_func, lval = 0):
    """Minimizes a cost_function using an iterative gradient descent approach."""
    elements, features = X_values.shape
    theta = np.zeros(shape=(features))
    step = 1
    old_cost = 0
    while True:
        theta_old = theta.copy()
        penal_array = np.ones_like(theta)
        penal_array[1:] = (1 - step * lval/len(y_values))
        cost = cost_func(theta, X_values, y_values) + lval * (1 + sum(np.power(theta[1:],2)))
        delta = delta_func(theta, X_values, y_values)
        theta = theta * penal_array - step * delta
        if cost > old_cost and old_cost != 0:
            step = step*0.7
        if np.allclose(theta_old, theta):
            break
        old_cost = cost
    return theta

### Numerical solvers:

def normalequation(X_values, y_values, lval = 0):
    """Solves a linear regression problem for theta using the normal equations approach."""
    l_matrix = np.eye(N = X_values.shape[1])
    l_matrix[0][0] = 0
    return np.linalg.pinv(X_values.T @ X_values + lval * l_matrix) @ X_values.T @ y_values
