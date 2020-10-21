
import numpy as np

from scipy.stats import multivariate_normal


class BayesianLinearRegression(object):

    def __init__(self, w_prior, s):
        self.w_prior = w_prior
        self.s = s

    def fit(self, X, y):
        pass

    def predict(self):
        pass