import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BlockStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardise the x-block and the y-block of the target separately.

    Target layout: [x(t_0..t_{n_t-1}), y(t_0..t_{n_t-1})]
    One mean and one std (scalars) per block, so that x and y end up on the
    same scale while the shape of each trajectory is left untouched.
    """

    def __init__(self, n_t):
        self.n_t = n_t

    def fit(self, Y, y=None):
        Y = np.asarray(Y)
        self.mean_ = np.array([Y[:, :self.n_t].mean(), Y[:, self.n_t:].mean()])
        self.std_ = np.array([Y[:, :self.n_t].std(), Y[:, self.n_t:].std()])
        return self

    def _expand(self, values):
        # (2,) -> (2*n_t,): same value repeated over each block
        return np.concatenate([np.full(self.n_t, values[0]), np.full(self.n_t, values[1])])

    def transform(self, Y):
        return (np.asarray(Y) - self._expand(self.mean_)) / self._expand(self.std_)

    def inverse_transform(self, Y):
        return np.asarray(Y) * self._expand(self.std_) + self._expand(self.mean_)