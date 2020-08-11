"""This module implements loss functions."""

import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss function.

    Parameters
    ----------
    y_true : ndarray
        A column vector with the true labels.

    y_pred : ndarray
        A column vector with the predicted labels.

    Returns
    -------
    J : float
        The average binary cross-entropy loss.
    """
    return -np.mean(y_true.T * np.log(y_pred) + (1 - y_true).T * np.log(1 - y_pred))


