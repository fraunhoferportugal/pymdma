from typing import Literal

import numpy as np

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType
from pymdma.general.functional.distance import cos_sim_2d, fast_mmd_linear, mmd_kernel, wasserstein, mk_mmd
from pymdma.general.functional.ratio import dispersion_ratio, distance_ratio
from pymdma.general.utils.util import features_splitting


class WassersteinDistance(FeatureMetric):
    """Calculate the Wasserstein distance between two sets of samples.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    compute_ratios : bool, optional, default=True
        If True, the diversity and dispersion ratios will be computed.
        If False, ratio computation is skipped. Default is True.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    Examples
    --------
    >>> wasserstein_distance = WassersteinDistance()
    >>> real_features = np.random.rand(64, 48) # (n_samples, num_features)
    >>> fake_features = np.random.rand(64, 48) # (n_samples, num_features)
    >>> result: MetricResult = wasserstein_distance.compute(x_feat, y_feat)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        compute_ratios: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_ratios = compute_ratios

    def _compute_ratios(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs):
        """Calculate the diversity and dispersion ratios.

        Parameters
        ----------
        real_features : array-like of shape (n_samples, n_features)
            2D array with features of the original samples.
        fake_features : array-like of shape (n_samples, n_features)
            2D array with features of the fake samples.

        Returns
        -------
        dispersion_ratio : float
            The dispersion ratio between the real features distributions and the fake features_distribution.
            Ideal value is 1 (dispersion between fake samples is equal to the dispersion between real samples).
        distance_ratio : float
            The distance ratio between t real features distributions and the fake features_distribution.
            Ideal value is 1 (distance between real and fake samples is equal to the distance between real samples).
        """

        state = kwargs.get("context", {})

        if any(key not in state for key in {"x_split_1", "x_split_2", "y_split_1", "y_split_2"}):
            state["x_split_1"], state["x_split_2"] = features_splitting(real_features, seed=0)
            state["y_split_1"], state["y_split_2"] = features_splitting(fake_features, seed=0)

        return {
            "dispersion_ratio": dispersion_ratio(
                wasserstein,
                state["x_split_1"],
                state["x_split_2"],
                state["y_split_1"],
                state["y_split_2"],
            ),
            "distance_ratio": distance_ratio(
                wasserstein,
                state["x_split_1"],
                state["x_split_2"],
                state["y_split_1"],
                state["y_split_2"],
            ),
        }

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Calculate the Wasserstein distance between two sets of samples. If
        compute_distance is True, the diversity and dispersion ratios are also
        computed using the wasserstein distance.

        Parameters
        ----------
        real_features : array-like of shape (n_samples, n_features)
            2D array with features of the original samples.
        fake_features : array-like of shape (n_samples, n_features)
            2D array with features of the fake samples.
        **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).


        Returns
        -------
        MetricResult
            Dataset-level results for the Wasserstein distance.
        """

        wasserstein_distance = wasserstein(real_features, fake_features)

        ratios = None
        if self.compute_ratios:
            ratios = self._compute_ratios(real_features, fake_features, **kwargs)

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": wasserstein_distance,
                "stats": ratios,
            },
        )


class MMD(FeatureMetric):
    """Calculate the Maximum Mean Discrepancy (MMD) using a specified kernel
    function.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    compute_ratios : bool, optional, default=True
        If True, the diversity and dispersion ratios will be computed.
        If False, ratio computation is skipped. Default is True.
    kernel : str, optional, default='linear'
        The kernel function to use for calculating MMD. Options include:
        'multi_gaussian', 'additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.
        
    Notes
    -----
    When using gaussian kernel, the number of samples in both datasets must be the same

    References
    ----------
    Gretton, A. et al. "A Kernel Method for the Two-Sample Problem" (2006)
    https://arxiv.org/pdf/0805.2368

    Gretton, A. et al, Optimal kernel choice for large-scale two-sample tests. (NIPS'12)
    https://proceedings.neurips.cc/paper_files/paper/2012/file/dbe272bab69f8e13f14b405e038deb64-Paper.pdf

    Examples
    --------
    >>> mmd = MMD(kernel = 'linear')
    >>> real_features = np.random.rand(64, 48) # (n_samples, num_features)
    >>> fake_features = np.random.rand(64, 48) # (n_samples, num_features)
    >>> result: MetricResult = mmd.compute(x_feat, y_feat)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        kernel: Literal[
            "multi_gaussian",
            "gaussian",
            "additive_chi2",
            "chi2",
            "linear",
            "poly",
            "polynomial",
            "rbf",
            "laplacian",
            "sigmoid",
            "cosine",
        ] = "linear",
        compute_ratios: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.compute_ratios = compute_ratios

        if kernel == "linear":
            self.kernel_fn = fast_mmd_linear
        elif kernel == "multi_gaussian":
            self.kernel_fn = mk_mmd
        else:
            self.kernel_fn = mmd_kernel

    def _compute_ratios(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs):
        """Calculate the diversity and dispersion ratios.

        Parameters
        ----------
        real_features : array-like of shape (n_samples, n_features)
            2D array with features of the original samples.
        fake_features : array-like of shape (n_samples, n_features)
            2D array with features of the fake samples.
        **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).


        Returns
        -------
        dispersion_ratio : float
            The dispersion ratio between the real features distributions and the fake features_distribution.
            Ideal value is 1 (dispersion between fake samples is equal to the dispersion between real samples).
        distance_ratio : float
            The distance ratio between t real features distributions and the fake features_distribution.
            Ideal value is 1 (distance between real and fake samples is equal to the distance between real samples).
        """

        state = kwargs.get("context", {})

        if any(key not in state for key in {"x_split_1", "x_split_2", "y_split_1", "y_split_2"}):
            state["x_split_1"], state["x_split_2"] = features_splitting(real_features, seed=0)
            state["y_split_1"], state["y_split_2"] = features_splitting(fake_features, seed=0)

        return {
            "dispersion_ratio": dispersion_ratio(
                self.kernel_fn,
                state["x_split_1"],
                state["x_split_2"],
                state["y_split_1"],
                state["y_split_2"],
            ),
            "distance_ratio": distance_ratio(
                self.kernel_fn,
                state["x_split_1"],
                state["x_split_2"],
                state["y_split_1"],
                state["y_split_2"],
            ),
        }

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Calculate the Maximum Mean Discrepancy (MMD) using a specified
        kernel function.

        Parameters
        ----------
        real_features : array-like of shape [n_samples_x, n_features]
            2D array containing features from samples of the original distribution.
        fake_features : array-like of shape [n_samples_y, n_features]
            2D array containing features from samples of the fake distribution.
        **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

        Returns
        -------
        MetricResult
                Dataset-level results for the MMD distance.
        """
        assert self.kernel != "gaussian" or len(real_features) == len(
            fake_features,
        ), "Gaussian kernel requires the same number of samples in each dataset."

        mmd = self.kernel_fn(real_features, fake_features, kernel=self.kernel)

        ratios = None
        if self.compute_ratios:
            ratios = self._compute_ratios(real_features, fake_features, **kwargs)

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": mmd,
                "stats": ratios,
            },
        )


class CosineSimilarity(FeatureMetric):
    """Calculate the cosine similarity between two sets of feature vectors.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Manning, C. D., Raghavan, P., & Schütze, H., An Introduction to Information Retrieval (2008).
    https://www.cambridge.org/highereducation/books/introduction-to-information-retrieval/669D108D20F556C5C30957D63B5AB65C#overview


    Examples
    --------
    >>> cossine_sim = MMD()
    >>> real_features = np.random.rand(64, 48) # (n_samples, num_features)
    >>> fake_features = np.random.rand(64, 48) # (n_samples, num_features)
    >>> result: MetricResult = cossine_sim.compute(x_feat, y_feat)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Calculate the cosine similarity between two sets of feature vectors.

        Parameters
        ----------
        real_features : array-like of shape [n_samples_x, n_features]
            2D array with features of the original samples.
        fake_features : array-like of shape [n_samples_y, n_features]
            2D array with features of the fake samples.

        Returns
        -------
        MetricResult
            Dataset-level results for the cosine similarity.
        """

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": cos_sim_2d(real_features, fake_features),
            },
        )


__all__ = ["CosineSimilarity", "MMD", "WassersteinDistance"]
