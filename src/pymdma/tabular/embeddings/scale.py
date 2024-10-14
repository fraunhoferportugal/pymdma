from typing import Optional

from loguru import logger
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from ..embeddings import GenericNorm


class StandardNorm(GenericNorm):
    """Standard normalization class using StandardScaler to scale data to have
    a mean of 0 and a standard deviation of 1. Inherits from the GenericNorm
    base class.

    Example
    -------
    >>> scaler = StandardNorm()
    >>> X = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler.fit_transform(X)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = StandardScaler(**kwargs)
        except Exception as e:
            logger.debug(f"Error: {e}")
            self.scaler_obj = StandardScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X):
        return self.scaler_obj.inverse_transform(X)


class RobustNorm(GenericNorm):
    """Robust normalization class using RobustScaler to scale data by removing
    the median and scaling according to the interquartile range. Inherits from
    GenericNorm.

    Example
    -------
    >>> scaler = RobustNorm()
    >>> X = [[1, 2], [2, 4], [4, 8], [5, 10]]
    >>> scaler.fit_transform(X)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = RobustScaler(**kwargs)
        except Exception as e:
            logger.debug(f"Error: {e}")
            self.scaler_obj = RobustScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X):
        return self.scaler_obj.inverse_transform(X)


class MinMaxNorm(GenericNorm):
    """Min-max normalization class using MinMaxScaler to scale data to a given
    range (default between 0 and 1). Inherits from GenericNorm.

    Example
    -------
    >>> scaler = MinMaxNorm()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    >>> scaler.fit_transform(X)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = MinMaxScaler(**kwargs)
        except Exception as e:
            logger.debug(f"Error: {e}")
            self.scaler_obj = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X):
        return self.scaler_obj.inverse_transform(X)


class MaxAbsNorm(GenericNorm):
    """Max-abs normalization class using MaxAbsScaler to scale data based on
    the absolute maximum value for each feature. Inherits from GenericNorm.

    Example
    -------
    >>> scaler = MaxAbsNorm()
    >>> X = [[1, -1], [2, -2], [3, -3], [4, -4]]
    >>> scaler.fit_transform(X)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.scaler_obj = MaxAbsScaler(**kwargs)
        except Exception as e:
            logger.debug(f"Error: {e}")
            self.scaler_obj = MaxAbsScaler()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X):
        return self.scaler_obj.inverse_transform(X)


class GlobalNorm:
    """A wrapper class that applies the specified normalization method based on
    the provided normalization type.

    Parameters
    ----------
    n_type : Optional[str]
        The type of normalization method to use.
        Must be one of: "umap", "pca", "lle", "isomap", "spacy", "bert", "identity".

    Example
    -------
    >>> scaler = GlobalNorm(n_type='minmax')
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    >>> scaler.fit_transform(X)
    """

    N_MAP = {
        "maxabs": MaxAbsNorm,
        "minmax": MinMaxNorm,
        "standard": StandardNorm,
        "robust": RobustNorm,
        "identity": GenericNorm,
    }

    def __init__(self, n_type: Optional[str] = "standard", **kwargs):
        if isinstance(n_type, str) and n_type in self.N_MAP:
            # tag
            norm_tag = n_type.lower()

            # norm class
            cls_ = self.N_MAP.get(norm_tag, GenericNorm)

            # norm instance
            self.scaler_obj = cls_(**kwargs)
        else:
            self.scaler_obj = GenericNorm()

    def fit(self, X, y=None):
        self.scaler_obj.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler_obj.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler_obj.fit_transform(X, y)

    def inverse_transform(self, X):
        return self.scaler_obj.inverse_transform(X)
