import numpy as np

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ...utils_syn import _get_nn_distances, _get_nn_pdf, _get_pp_metrics


class NNDRPrivacy(FeatureMetric):
    """Compute the NNDR (Nearest Neighbor Distance Ratio) privacy score. A more
    negative score indicates better privacy assurance, while a less negative
    score suggests higher risk.

    **Objective**: Authenticity

    Parameters
    ----------
    distance_type : str, optional, default="euclidean"
        The distance metric to use for calculating distances between embeddings.
        Default is "euclidean".
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Liu et al., Scaling while privacy preserving: A comprehensive synthetic tabular data generation and evaluation in learning analytics (2024).
    https://doi.org/10.1145/3636555.3636921

    Returns
    -------
    MetricResult
        A MetricResult object containing the DCR and NNDR privacy scores.

    Examples
    --------
    >>> import numpy as np
    >>> real_embeddings = np.random.rand(100, 10)  # 100 samples, 10 features
    >>> synthetic_embeddings = np.random.rand(100, 10)
    >>> nndr_privacy_score = NNDRPrivacy(distance_type="euclidean")
    >>> result: MetricResult = nndr_privacy_score.compute(real_embeddings, synthetic_embeddings)
    >>> dataset_level, _ = result.value  # Output: privacy metrics including DCR and NNDR scores
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.PRIVACY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 100.0

    def __init__(
        self,
        distance_type: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # distance type with which metrics are calculated
        self.distance_type = distance_type

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the NNDR privacy score comparing synthetic and real dataset
        embeddings.

        Parameters
        ----------
        real_data : np.ndarray
            The target dataset embeddings.
        syn_data : np.ndarray
            The synthetic dataset embeddings.
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the privacy metrics including DCR and NNDR scores.
        """

        # get sample-sample distances
        _, nndr_map = _get_nn_distances(real_data, syn_data, self.distance_type)

        # get NNDR distributions
        tgt_pdf, syn_pdf, bins = _get_nn_pdf(
            nndr_map["tgt"],
            nndr_map["syn"],
        )

        # get NNDR scores
        metric_d = _get_pp_metrics(
            tgt_pdf,
            syn_pdf,
            bins,
            tag="NNDR",
            low_perc=25,
            high_perc=75,
        )

        # privacy metrics
        priv_d = {
            "privacy": 100 - metric_d.get("%_leak"),
            "level": metric_d.get("%_ctd_disp"),
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": priv_d,
            },
        )


class DCRPrivacy(FeatureMetric):
    """Compute the DCR (Distance to Closest Record) privacy score. A more
    negative score indicates better privacy assurance, while a less negative
    score suggests higher risk.

    **Objective**: Authenticity

    Parameters
    ----------
    distance_type : str, optional, default="euclidean"
        The distance metric to use for calculating distances between embeddings.
        Default is "euclidean". Available from: `link <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics>`_
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Liu et al., Scaling while privacy preserving: A comprehensive synthetic tabular data generation and evaluation in learning analytics (2024).
    https://doi.org/10.1145/3636555.3636921

    Returns
    -------
    MetricResult
        A MetricResult object containing the DCR and NNDR privacy scores.

    Examples
    --------
    >>> import numpy as np
    >>> real_emb = np.random.rand(100, 10)  # 100 samples, 10 features
    >>> synthetic_emb = np.random.rand(100, 10)
    >>> dcr_privacy_score = DCRPrivacy(distance_type="euclidean")
    >>> result: MetricResult = dcr_privacy_score.compute(real_emb, synthetic_emb)
    >>> dataset_level, _ = result.value  # Output: privacy metrics including DCR and NNDR scores
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.PRIVACY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 100.0

    def __init__(
        self,
        distance_type: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_type = distance_type

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the DCR privacy score comparing synthetic and real dataset
        embeddings.

        Parameters
        ----------
        real_data : np.ndarray
            The target dataset embeddings.
        syn_data : np.ndarray
            The synthetic dataset embeddings.
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the privacy metrics including DCR and NNDR scores.
        """

        # get sample-sample distances
        dcr_map, _ = _get_nn_distances(real_data, syn_data, self.distance_type)

        # get DCR distributions
        tgt_pdf, syn_pdf, bins = _get_nn_pdf(
            dcr_map["tgt"],
            dcr_map["syn"],
        )

        # get DCR scores
        metric_d = _get_pp_metrics(
            tgt_pdf,
            syn_pdf,
            bins,
            tag="DCR",
            low_perc=25,
            high_perc=75,
        )

        # privacy metrics
        priv_d = {
            "privacy": 100 - metric_d.get("%_leak"),
            "level": metric_d.get("%_ctd_disp"),
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": priv_d,
            },
        )


__all__ = ["NNDRPrivacy", "DCRPrivacy"]
