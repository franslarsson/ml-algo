"""This module implements activation functions."""

import numpy as np


class Sigmoid(object):
    """Sigmoid activation function."""
    def __init__(self):
        pass

    def evaluate(self, z):
        """Evaluate the sigmoid function at the value `z`.

        Parameters
        ----------
        z : float or ndarray
        A value for evaluating the sigmoid function at.

        Returns
        -------
        a : float or ndarray
            The sigmoid function evaluated at `z`. The type will be float if
            `z` is a float or ndarray if `z` is a ndarray.
        """
        return 1.0 / (1 + np.exp(-z))

    def derivative(self, z):
        """ Evaluate the derivative of the sigmoid function at the value `z`.

        Parameters
        ----------
        z : float or ndarray
            A value for evaluating the derivative of the sigmoid function at.

        Returns
        -------
        da : float or ndarray
            The derivative of the sigmoid function evaluated at `z`. The type
            will be float if `z` is a float or ndarray if `z` is a ndarray.
        """
        (1 - self.evaluate(z)) * self.evaluate(z)
