import numpy as np


class GenericEmbedder:
    def __init__(self, **kwargs) -> None:
        # has inverse transform
        self.has_inv = False

    def fit(self, X: np.ndarray, **kwargs):
        pass

    def transform(self, X: np.ndarray, **kwargs):
        return X

    def fit_transform(self, X: np.ndarray, **kwargs):
        # fit
        self.fit(X)

        # transform
        X_new = self.transform(X)

        return X_new

    def inverse_transform(self, X: np.ndarray, **kwargs):
        return X


class GenericNorm:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X
