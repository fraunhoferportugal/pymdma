from typing import Literal

import numpy as np
from sklearn.mixture import GaussianMixture

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType
from pymdma.general.utils.util import min_max_scaling


class GIQA(FeatureMetric):
    """Generated Image Quality Assessment (GIQA) metric based on Gaussian
    Mixture Model (GMM). By default, computes the Quality score (QS) as
    reported in the paper.

    To compute the Diversity score (DS), exchange the real and synthetic features.
    The instance level result will indicate, for each real sample, how well it
    is represented in the synthetic distribution. The dataset level result will
    indicate the overal diversity score as defined in the paper.

    **Objective**: Quality, Diversity

    Parameters
    ----------
    n_components : int, optional
        Number of components in the GMM. Defaults to 7.
    covariance_type : str, optional
        Type of covariance. Defaults to "full".
    cache_model : bool, optional
        If set to true the GMM model will only be fitted once and then cached. Defaults to False.
        Only set to True if the reference features are constant across all calls to the compute method.
    random_state : int, optional
        Random seed. Defaults to 0.
    **kwargs : dict, optional
        Additional keyword arguments to be used by the GMM model.

    References
    ----------
    Gu et al., GIQA: Generated Image Quality Assessment (2020).
    https://arxiv.org/abs/2003.08932

    GIQA, GIQA: Generated Image Quality Assessment
    https://github.com/cientgu/GIQA

    Examples
    --------
    >>> giqa = GIQA()
    >>> x_feats = np.random.rand(100, 100)
    >>> y_feats = np.random.rand(100, 100)
    >>> quality_score: MetricResult = giqa.compute(x_feats, y_feats)
    >>> diversity_score: MetricResult = giqa.compute(y_feats, x_feats)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        n_components: int = 7,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        cache_model: bool = False,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.cache_model = cache_model

        self._mixture_model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            verbose=0,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Compute the GIQA metric.

        Parameters
        ----------
        real_features : np.ndarray
            ndarray of shape (n_samples, n_features) containing features of real samples.

        fake_features : np.ndarray
            ndarray of shape (n_samples, n_features) containing features of fake/generated samples.

        Returns
        -------
        result : MetricResult
            dataset-level mean of the scores and instance-level scores
        """
        # fit GMM model on real features
        if not self._fitted or (self._fitted and not self.cache_model):
            self._mixture_model.fit(real_features)
            self._fitted = True
        # compute scores for fake features
        scores = self._mixture_model.score_samples(fake_features)
        scores = min_max_scaling(scores)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": scores.mean()},
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores.tolist()},
        )


__all__ = ["GIQA"]
