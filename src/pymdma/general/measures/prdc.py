import numpy as np

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ..utils.util import compute_nearest_neighbour_distances, compute_pairwise_distance


class ImprovedPrecision(FeatureMetric):
    """Improved Precision Metric for accessing fidelity of generative models.

    **Objective**: Fidelity

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors to consider in the hypersphere estimation. Defaults to 5.
    metric : str, optional, default="euclidean"
        The metric to use when calculating distance between instances.
        For the available metrics, see the documentation of `sklearn.metrics.pairwise_distances`.
    **kwargs
        Additional keyword arguments for compatiblilty.

    References
    ----------
    Kynkaanniemi et al., Improved Precision and Recall Metric for Assessing Generative Models (2019).
    https://arxiv.org/abs/1904.06991

    Code adapted from:
    improved-precision-and-recall-metric: Improved Precision and Recall Metric for Assessing Generative Models — Official TensorFlow Implementation.
    https://github.com/kynkaat/improved-precision-and-recall-metric

    Hypersphere estimation code was taken from:
    generative-evaluation-prdc, Reliable Fidelity and Diversity Metrics for Generative Models.
    https://github.com/clovaai/generative-evaluation-prdc

    Examples
    --------
    >>> improved_precision = ImprovedPrecision()
    >>> real_features = np.random.rand(100, 100)
    >>> fake_features = np.random.rand(100, 100)
    >>> result: MetricResult = improved_precision.compute(real_features, fake_features)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Compute the Improved Precision metric.

        Parameters
        ----------
        real_features : np.ndarray
            Array of shape (n_samples, n_features) containing the real features.
        fake_features : np.ndarray
            Array of shape (n_samples, n_features) containing the fake features.

        Notes
        -----
        Intermediate computations can be stored in the `context` dictionary of the `kwargs` parameter.
        Usefull when calculating multiple metrics that share the same intermediate computations.

        Returns
        -------
        result: MetricResult
            Dataset-level and instance-level results for the precision metric.
        """
        state = kwargs.get("context", {})
        if "real_nn_distances" not in state:
            state["real_nn_distances"] = compute_nearest_neighbour_distances(
                real_features,
                nearest_k=self.k,
                metric=self.metric,
            )

        if "real_fake_distances" not in state:
            state["real_fake_distances"] = compute_pairwise_distance(real_features, fake_features, metric=self.metric)

        precision = (
            np.logical_or(
                (state["real_fake_distances"] < np.expand_dims(state["real_nn_distances"], axis=1)),
                np.isclose(state["real_fake_distances"], np.expand_dims(state["real_nn_distances"], axis=1)),
            )
            .any(axis=0)
            .astype(int)
        )

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": precision.mean()},
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "int", "value": precision.tolist()},
        )


class ImprovedRecall(FeatureMetric):
    """Improved Recall Metric for accessing diversity of generative models.

    **Objective**: Diversity

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors to consider in the hypersphere estimation. Defaults to 5.
    metric : str, optional, default="euclidean"
        The metric to use when calculating distance between instances.
        For the available metrics, see the documentation of `sklearn.metrics.pairwise_distances`.
    **kwargs
        Additional keyword arguments for compatiblilty.

    References
    ----------
    Kynkaanniemi et al., Improved Precision and Recall Metric for Assessing Generative Models (2019).
    https://arxiv.org/abs/1904.06991

    Code adapted from:
    improved-precision-and-recall-metric: Improved Precision and Recall Metric for Assessing Generative Models — Official TensorFlow Implementation.
    https://github.com/kynkaat/improved-precision-and-recall-metric

    Hypersphere estimation code was taken from:
    generative-evaluation-prdc, Reliable Fidelity and Diversity Metrics for Generative Models.
    https://github.com/clovaai/generative-evaluation-prdc

    Examples
    --------
    >>> improved_recall = ImprovedRecall()
    >>> real_features = np.random.rand(100, 100)
    >>> fake_features = np.random.rand(100, 100)
    >>> result: MetricResult = improved_recall.compute(real_features, fake_features)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Compute the Improved Recall metric.

        Parameters
        ----------
        real_features : np.ndarray
            Array of shape (n_samples, n_features) containing the real features.
        fake_features : np.ndarray
            Array of shape (n_samples, n_features) containing the fake features.

        Notes
        -----
        Intermediate computations can be stored in the `context` dictionary of the `kwargs` parameter.
        Usefull when calculating multiple metrics that share the same intermediate computations.

        Returns
        -------
        result: MetricResult
            Dataset-level and instance-level results for the recall metric.
        """
        state = kwargs.get("context", {})
        if "fake_nn_distances" not in state:
            state["fake_nn_distances"] = compute_nearest_neighbour_distances(
                fake_features,
                nearest_k=self.k,
                metric=self.metric,
            )

        if "real_fake_distances" not in state:
            state["real_fake_distances"] = compute_pairwise_distance(real_features, fake_features, metric=self.metric)

        recall_mask = np.logical_or(
            state["real_fake_distances"] < np.expand_dims(state["fake_nn_distances"], axis=0),
            np.isclose(state["real_fake_distances"], np.expand_dims(state["fake_nn_distances"], axis=0)),
        )
        recall = recall_mask.any(axis=1).astype(int)

        # matrix with (R, F) shape -> .any() -> matrix with (F,) shape
        # an array that indicates for each F sample how many real samples are within its manifold
        recall_counts = recall_mask.sum(axis=0)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": recall.mean()},
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "int", "value": recall_counts.tolist()},
        )


class Density(FeatureMetric):
    """Density Metric for accessing fidelity of the generated samples. Unlike
    Improved Precision, it is robust towards outliers in the real/reference
    data.

    **Objective**: Fidelity

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors to consider in the hypersphere estimation. Defaults to 5.
    metric : str, optional, default="euclidean"
        The metric to use when calculating distance between instances.
        For the available metrics, see the documentation of `sklearn.metrics.pairwise_distances`.
    **kwargs
        Additional keyword arguments for compatibility.

    References
    ----------
    Naeem et al., Reliable Fidelity and Diversity Metrics for Generative Models (2020).
    https://arxiv.org/abs/2002.09797

    Code was adapted from:
    generative-evaluation-prdc, Reliable Fidelity and Diversity Metrics for Generative Models.
    https://github.com/clovaai/generative-evaluation-prdc

    Examples
    --------
    >>> density = Density()
    >>> real_features = np.random.rand(100, 100)
    >>> fake_features = np.random.rand(100, 100)
    >>> result: MetricResult = density.compute(real_features, fake_features)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Compute the Density metric.

        Parameters
        ----------
        real_features : np.ndarray
            Array of shape (n_samples, n_features) containing the real features.
        fake_features : np.ndarray
            Array of shape (n_samples, n_features) containing the fake features.

        Notes
        -----
        Intermediate computations can be stored in the `context` dictionary of the `kwargs` parameter.
        Usefull when calculating multiple metrics that share the same intermediate computations.

        Returns
        -------
        result: MetricResult
            Dataset-level and instance-level results for the density metric.
        """
        state = kwargs.get("context", {})
        if "real_nn_distances" not in state:
            state["real_nn_distances"] = compute_nearest_neighbour_distances(
                real_features,
                nearest_k=self.k,
                metric=self.metric,
            )

        if "real_fake_distances" not in state:
            state["real_fake_distances"] = compute_pairwise_distance(real_features, fake_features, metric=self.metric)

        density = np.logical_or(
            (state["real_fake_distances"] < np.expand_dims(state["real_nn_distances"], axis=1)),
            np.isclose(state["real_fake_distances"], np.expand_dims(state["real_nn_distances"], axis=1)),
        )
        density = (1.0 / float(self.k)) * density.sum(axis=0)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": density.mean()},
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": density.tolist()},
        )


class Coverage(FeatureMetric):
    """Coverage Metric for accessing diversity of the generated samples. Unlike
    Improved Recall, it is robust towards outliers in the real/reference data.

    **Objective**: Diversity

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors to consider in the hypersphere estimation. Defaults to 5.
    metric : str, optional, default="euclidean"
        The metric to use when calculating distance between instances.
        For the available metrics, see the documentation of `sklearn.metrics.pairwise_distances`.
    **kwargs
        Additional keyword arguments for compatibility.

    References
    ----------
    Naeem et al., Reliable Fidelity and Diversity Metrics for Generative Models (2020).
    https://arxiv.org/abs/2002.09797

    Code was adapted from:
    generative-evaluation-prdc, Reliable Fidelity and Diversity Metrics for Generative Models.
    https://github.com/clovaai/generative-evaluation-prdc

    Examples
    --------
    >>> coverage = Coverage()
    >>> real_features = np.random.rand(100, 100)
    >>> fake_features = np.random.rand(100, 100)
    >>> result: MetricResult = coverage.compute(real_features, fake_features)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Compute the Coverage metric.

        Parameters
        ----------
        real_features : np.ndarray
            Array of shape (n_samples, n_features) containing the real features.
        fake_features : np.ndarray
            Array of shape (n_samples, n_features) containing the fake features.

        Notes
        -----
        Intermediate computations can be stored in the `context` dictionary of the `kwargs` parameter.
        Usefull when calculating multiple metrics that share the same intermediate computations.

        Returns
        -------
        result: MetricResult
            Dataset-level and instance-level results for the coverage metric.
        """
        state = kwargs.get("context", {})
        if "real_nn_distances" not in state:
            state["real_nn_distances"] = compute_nearest_neighbour_distances(
                real_features,
                nearest_k=self.k,
                metric=self.metric,
            )

        if "real_fake_distances" not in state:
            state["real_fake_distances"] = compute_pairwise_distance(real_features, fake_features, metric=self.metric)

        coverage = np.logical_or(
            state["real_fake_distances"].min(axis=1) < state["real_nn_distances"],
            np.isclose(state["real_fake_distances"].min(axis=1), state["real_nn_distances"]),
        )

        # matrix with (R, F) shape -> .any() -> matrix with (F,) shape
        # an array that indicates for each F in how many real manifolds it is contained in
        coverage_counts = np.logical_or(
            state["real_fake_distances"] < np.expand_dims(state["real_nn_distances"], axis=1),
            np.isclose(state["real_fake_distances"], np.expand_dims(state["real_nn_distances"], axis=1)),
        ).sum(axis=0)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": coverage.mean()},
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "int", "value": coverage_counts.tolist()},
        )


class Authenticity(FeatureMetric):
    """Authenticity Metric for assessing the authenticity of the generated
    samples. A synthetic sample is considered authentic if it is signficantly
    distinct from any real sample.

    **Objective**: Privacy

    Parameters
    ----------
    metric : str, optional, default="euclidean"
        The metric to use when calculating distance between instances.
        For the available metrics, see the documentation of `sklearn.metrics.pairwise_distances`.
    **kwargs
        Additional keyword arguments for compatibility.

    Notes
    -----
    The authenticity metric is computed by checking if any fake sample is closer to a real sample than the real sample is to any other real sample.

    References
    ----------
    Naeem et al., Reliable Fidelity and Diversity Metrics for Generative Models (2020).
    https://arxiv.org/abs/2002.09797

    Kynkaanniemi et al., Improved Precision and Recall Metric for Assessing Generative Models (2019).
    https://arxiv.org/abs/1904.06991

    Hypersphere estimation code was adapted from:
    generative-evaluation-prdc, Reliable Fidelity and Diversity Metrics for Generative Models.
    https://github.com/clovaai/generative-evaluation-prdc

    Examples
    --------
    >>> authenticity = Authenticity()
    >>> real_features = np.random.rand(100, 100)
    >>> fake_features = np.random.rand(100, 100)
    >>> result: MetricResult = authenticity.compute(real_features, fake_features)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.PRIVACY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric = metric

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Compute the Authenticity metric.

        Parameters
        ----------
        real_features : np.ndarray
            Array of shape (n_samples, n_features) containing the real features.
        fake_features : np.ndarray
            Array of shape (n_samples, n_features) containing the fake features.

        Notes
        -----
        Intermediate computations can be stored in the `context` dictionary of the `kwargs` parameter.
        Usefull when calculating multiple metrics that share the same intermediate computations.

        Returns
        -------
        result: MetricResult
            Dataset-level and instance-level results for the authenticity metric
        """
        state = kwargs.get("context", {})

        if "real_fake_distances" not in state:
            state["real_fake_distances"] = compute_pairwise_distance(real_features, fake_features, metric=self.metric)

        # compute distance to closest real samples
        state["real_closest_real_distances"] = compute_nearest_neighbour_distances(
            real_features,
            nearest_k=1,
            metric=self.metric,
        )

        # check if any fake sample is closer to Ri than Ri is to any other Rj
        authenticity = np.logical_or(
            state["real_fake_distances"] < np.expand_dims(state["real_closest_real_distances"], axis=1),
            np.isclose(state["real_fake_distances"], np.expand_dims(state["real_closest_real_distances"], axis=1)),
        )

        # mask of the values that are considered authentic in the fake dataset
        authenticity_mask = ~authenticity.any(axis=0)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": authenticity_mask.mean()},
            instance_level={
                "dtype": OutputsTypes.ARRAY,
                "subtype": "int",
                "value": authenticity_mask.astype(int).tolist(),
            },
        )


__all__ = ["ImprovedPrecision", "ImprovedRecall", "Density", "Coverage", "Authenticity"]
