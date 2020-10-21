
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
    
    def calc_posterior(self, X):
        s_prior_inv = np.linalg.pinv(self.w_prior.cov)
        s_post = np.linalg.pinv(s_prior_inv + X.T.dot(X) / self.s)
        mu_post = s_post.dot(s_prior_inv.dot(self.w_prior.mean) + X.T.dot(y) / self.s)
        return multivariate_normal(mean=mu_post, cov=s_post)

    def update_prior(self, posterior):
        self.w_prior = posterior