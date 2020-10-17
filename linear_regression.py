
import numpy as np


class LinearRegression(object):
    """Linear regression.

    Parameters
    ----------
    intercept : bool
        If True then the model will be fitted with an intercept.

    Attributes
    ----------
    beta : ndarray
      The beta coefficients. Will be of shape (m,) where m is the number of
      features in the provided training data.
    """

    def __init__(self, intercept=True):
        self.beta = None
        self.intercept = intercept

    def fit(self, X, y):
        if self.intercept:
            ones = np.ones(shape=(X.shape[0], 1))
            X = np.hstack([ones, X])
        self.beta = np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(y).reshape(-1, 1)

    def predict(self, X):
        if self.intercept:
            ones = np.ones(shape=(X.shape[0], 1))
            X = np.hstack([ones, X])
        return X.dot(self.beta)


class RidgeRegression(LinearRegression):

    """Ridge regression.

    Parameters
    ----------
    alpha : float
    intercept : bool

    """
    def __init__(self, alpha, intercept=True):
        super().__init__(intercept=intercept)
        self.alpha = alpha

    def fit(self, X, y):
        if self.intercept:
            ones = np.ones(shape=(X.shape[0], 1))
            X = np.hstack([ones, X])
        inv = np.linalg.pinv((X.T.dot(X) + self.alpha*np.eye(X.shape[1])))
        self.beta = inv.dot(X.T).dot(y).reshape(-1, 1)
