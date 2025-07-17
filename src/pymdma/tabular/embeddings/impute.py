import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

from ..embeddings.scale import GlobalNorm


class NNImputer:
    def __init__(self, n_neighbors: int = 2, weights: str = "uniform"):
        # imputer
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors, 
            weights=weights
        )

    def fit(self, X: np.ndarray):
        # fit
        self.imputer.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.imputer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.imputer.fit_transform(X)


class MeanImputer:
    def __init__(self):
        self.col_means = None
        self.inds = None
    
    def fit(self, X: np.ndarray):
        # compute mean of each column, skipping NaNs
        self.col_means = np.nanmean(X, axis=0)

        # replace NaNs with corresponding column means
        self.inds = np.where(np.isnan(X))

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.col_means is None or self.inds is None:
            raise ValueError("Must call `fit` first.")

        # replace NaNs with corresponding column means
        inds = self.inds

        # take column means
        X[inds] = np.take(self.col_means, inds[1])

        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # copy
        X_aux = X.copy()

        # fit estimator
        self.fit(X_aux)

        # compute column means
        col_means = self.col_means

        # replace NaNs with corresponding column means
        inds = self.inds

        # take column means
        X_aux[inds] = np.take(col_means, inds[1])

        return X_aux


class MedianImputer:
    def __init__(self):
        self.col_medians = None
        self.inds = None

    def fit(self, X: np.ndarray):
        # compute median of each column, skipping NaNs
        self.col_medians = np.nanmedian(X, axis=0)

        # replace NaNs with corresponding column medians
        self.inds = np.where(np.isnan(X))

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.col_medians is None or self.inds is None:
            raise ValueError("Must call `fit` first.")

        # replace NaNs with corresponding column medians
        inds = self.inds

        # take column medians
        X[inds] = np.take(self.col_medians, inds[1])

        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # copy
        X_aux = X.copy()

        # fit estimator
        self.fit(X_aux)

        # compute column medians
        col_medians = self.col_medians

        # replace NaNs with corresponding column medians
        inds = self.inds

        # take column medians
        X_aux[inds] = np.take(col_medians, inds[1])

        return X_aux


class GlobalImputer:
    def __init__(self, n_type: str = "knn", norm_type: str = "standard", **kwargs):
        # imputer tag
        imp_type = (n_type or "knn").lower()
        # norm tag
        n_type = (norm_type or "standard").lower()

        # KNN
        if imp_type == "knn":
            self.imputer = NNImputer(**kwargs)
        # Mean
        elif imp_type == "mean":
            self.imputer = MeanImputer(**kwargs)
        # Median
        elif imp_type == "median":
            self.imputer = MedianImputer(**kwargs)
        # Error
        else:
            raise ValueError("'imp_type' must be one of 'knn', 'mean', or 'median'")
        
        # normalization
        self.scaler = GlobalNorm(norm_type, **kwargs)

    def _get_col_types(self, data):
        # copy data
        aux_data_cp = data.astype(float).copy()
        aux_data_cp[np.isnan(aux_data_cp)] = 0

        # copy (round to nearest integer)
        aux_data_rnd = aux_data_cp.round(0)

        # check which columns are different (rounded != original)
        check = (aux_data_cp != aux_data_rnd).any(0)

        # columns where data is integer 
        # (imputations will be rounded to nearest integer)
        col_inds = np.where(~check)[0]

        return col_inds

    def fit_transform(self, X):
        # get column indices where data is integer
        int_cols = self._get_col_types(X)

        # impute the data
        X_imp = self.imputer.fit_transform(self.scaler.fit_transform(X))

        # inverse transform
        X_inv = self.scaler.inverse_transform(X_imp)

        # round integer columns to nearest integer
        X_inv[:, int_cols] = X_inv[:, int_cols].round(0)

        return X_inv
