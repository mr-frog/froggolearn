import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


__all__ = ['normalequation', 'gradientdescent', 'bfgs']


def bfgs(func, x0, args=(), jac = ''):
    """Solves for theta using scipys implementation of BFGS"""
    return minimize(fun = func, x0 = x0, args = args, method = 'BFGS', jac = jac).x

def normalequation(X_values, y_values):
    """Solves for theta using the normal equations approach."""

    return np.linalg.pinv(X_values.T @ X_values) @ X_values.T @ y_values

def gradientdescent(X_values, y_values, cost_func, delta_func):
    """Approximates Theta using an iterative gradient descent approach."""
    elements, features = X_values.shape
    theta = np.zeros(shape=(features))
    step = 1
    old_cost = 0
    while True:
        theta_old = theta.copy()
        cost = cost_func(theta, X_values, y_values)
        delta = delta_func(theta, X_values, y_values)
        theta = theta - step * delta
        if cost > old_cost and old_cost != 0:
            step = step*0.7
        if np.allclose(theta_old, theta):
            break
        old_cost = cost
    return theta
