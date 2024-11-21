import numpy as np

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ..utils.util import cluster_into_bins


def _compute_prd(eval_dist, ref_dist, num_angles: int = 1001, epsilon: float = 1e-10):
    """Computes the PRD curve for discrete distributions.

    This function computes the PRD curve for the discrete distribution eval_dist
    with respect to the reference distribution ref_dist. This implements the
    algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
    equiangular grid of num_angles values between [0, pi/2].

    Parameters
    ----------
    eval_dist : np.ndarray or list of float
        1D array or list of floats with the probabilities of the different states
        under the distribution to be evaluated.
    ref_dist : np.ndarray or list of float
        1D array or list of floats with the probabilities of the different states
        under the reference distribution.
    num_angles : int, optional
        Number of angles for which to compute PRD. Must be in [3, 1e6].
        The default value is 1001.
    epsilon : float, optional
        Angle for PRD computation in the edge cases 0 and pi/2. The PRD
        will be computed for epsilon and pi/2-epsilon, respectively.
        The default value is 1e-10.

    Returns
    -------
    precision : numpy.ndarray, shape (num_angles,)
        Precision for the different ratios.
    recall : numpy.ndarray, shape (num_angles,)
        Recall for the different ratios.

    Raises
    ------
    AssertionError
        If `epsilon` is not in the range (0, 0.1].
    AssertionError
        If `num_angles` is not in the range [3, 1e6].


    References
    ---------
    Sajjadi, Mehdi SM, et al. Assessing generative models via precision and recall (2018).
    https://proceedings.neurips.cc/paper_files/paper/2018/file/f7696a9b362ac5a51c3dc8f098b73923-Paper.pdf

    Code adapted from:
    https://github.com/vanderschaarlab/evaluating-generative-models/blob/main/metrics/prd_score.py
    """

    assert epsilon > 0 and epsilon <= 0.1, "epsilon must be in (0, 0.1]."
    assert num_angles >= 3 and num_angles <= 1e6, "num_angles must be in [3, 1e6]."

    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    # Broadcast slopes so that second dimension will be states of the distribution
    slopes_2d = np.expand_dims(slopes, 1)

    # Broadcast distributions so that first dimension represents the angles
    ref_dist_2d = np.expand_dims(ref_dist, 0)
    eval_dist_2d = np.expand_dims(eval_dist, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    # handle numerical instabilities leaing to precision/recall just above 1
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)
    return precision, recall


def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
    """Computes F_beta scores for the given precision/recall values.

    The F_beta scores for all precision/recall pairs will be computed and
    returned.

    For precision p and recall r, the F_beta score is defined as:
    F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)

    Parameters
    ----------
    precision : np.ndarray
        1D NumPy array of precision values in [0, 1].
    recall : np.ndarray
        1D NumPy array of recall values in [0, 1].
    beta : float, optional
        Beta parameter. Must be positive. The default value is 1.
    epsilon : float, optional
        Small constant to avoid numerical instability caused by division by 0
        when precision and recall are close to zero.

    Returns
    -------
    np.ndarray
        NumPy array of same shape as precision and recall with the F_beta scores
        for each pair of precision/recall.

    Raises
    ------
    AssertionError
        If `beta` is not positive.

    References
    ---------
    Sajjadi, Mehdi SM, et al. Assessing generative models via precision and recall (2018).
    https://proceedings.neurips.cc/paper_files/paper/2018/file/f7696a9b362ac5a51c3dc8f098b73923-Paper.pdf

    Code adapted from:
    https://github.com/vanderschaarlab/evaluating-generative-models/blob/main/metrics/prd_score.py
    """
    assert beta > 0, "Given parameter beta must be positive."

    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + epsilon)


class PrecisionRecallDistribution(FeatureMetric):
    """Computes PRD data from sample embeddings and the maximum F_beta scores
    for the given precision/recall values.

    The points from both distributions are mixed and then clustered. This leads
    to a pair of histograms of discrete distributions over the cluster centers
    on which the PRD algorithm is executed.

    For PRD, it is recommended that number of points in eval_data and ref_data are equal since
    unbalanced distributions bias the clustering towards the larger dataset. The
    check for this condition can be performed by setting the enforce_balance flag to True
    (recommended).

    Regarding the maximum F_beta scores, the maximum F_beta score over all pairs of precision/recall
    values is useful to compress a PRD plot into a single value which correlate with recall.
    Whereas, the max_f_beta_inv score over all pairs of precision/recall values compresses
    the PRD plot into a single value that correlates with precision.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    num_clusters: int, optional
        Number of cluster centers to fit. The default value is 2.
    num_angles: int, optional
        Number of angles for which to compute PRD. Must be in [3, 1e6]. The default value is 1001.
    num_runs: int, optional
        Number of independent runs over which to average the PRD data. The default value is 10.
    beta: int, optional
        Beta parameter for F_beta score. Must be positive. The default value is 8.
    epsilon: float, optional
        Small constant to avoid numerical instability caused by division
        by 0 when precision and recall are close to zero. The default value is 1e-10.
    compute_stats: bool, optional
        If True, F_beta scores for all precision/recall pairs will be computed.
        If False, F_beta scores computation is skipped. Default is True.

    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    References
    ---------
    Sajjadi, Mehdi SM, et al. Assessing generative models via precision and recall (2018).
    https://proceedings.neurips.cc/paper_files/paper/2018/file/f7696a9b362ac5a51c3dc8f098b73923-Paper.pdf

    Code adapted from:
    https://github.com/vanderschaarlab/evaluating-generative-models/blob/main/metrics/prd_score.py


    Examples
    --------
    >>> prd = PrecisionRecallDistribution()
    >>> x_feats = np.random.rand(64, 48)
    >>> y_feats = np.random.rand(64, 48)
    >>> result: MetricResult = prd.compute(x_feats, y_feats)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    def __init__(
        self,
        num_clusters: int = 2,
        num_angles: int = 1001,
        num_runs: int = 10,
        epsilon: float = 1e-10,
        beta: int = 8,
        compute_stats: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.num_angles = num_angles
        self.num_runs = num_runs
        self.epsilon = epsilon
        self.compute_stats = compute_stats

        assert beta > 0, "Given parameter beta must be positive."
        self.beta = beta

    def _prd_from_embedding(self, target: np.ndarray, reference: np.ndarray):
        eval_data = np.array(target, dtype=np.float64)
        ref_data = np.array(reference, dtype=np.float64)
        precisions = []
        recalls = []
        for _ in range(self.num_runs):
            eval_dist, ref_dist = cluster_into_bins(eval_data, ref_data, self.num_clusters)
            precision, recall = _compute_prd(eval_dist, ref_dist, self.num_angles, self.epsilon)
            precisions.append(precision)
            recalls.append(recall)
        precision = np.mean(precisions, axis=0)
        recall = np.mean(recalls, axis=0)
        return precision, recall

    def compute(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> MetricResult:
        """Computes PRD data from sample embeddings. Using the PRD, the maximum
        F_beta and F_beta_inv score are also computed. These scores for the
        given precision/recall values correlate, respectively, with recall and
        precision.

        Returns
        -------
        MetricResult
            Instance-level PRD values.
            Dataset-leve max_f_beta and max_f_beta_inv scores.
        """
        warning = None
        if len(fake_features) != len(real_features):
            warning = (
                "The number of points in eval_data %d is not equal to the number of "
                "points in ref_data %d. To disable this exception, set enforce_balance "
                "to False (not recommended)." % (len(fake_features), len(real_features))
            )

        precision, recall = self._prd_from_embedding(fake_features, real_features)

        stats = None
        if self.compute_stats:
            max_f_beta = np.max(_prd_to_f_beta(precision, recall, self.beta, self.epsilon))
            max_f_beta_inv = np.max(_prd_to_f_beta(recall, precision, 1 / self.beta, self.epsilon))
            stats = {
                "max_f_beta": max_f_beta,
                "max_f_beta_inv": max_f_beta_inv,
            }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_ARRAY,
                "subtype": "float",
                "value": {
                    "precision_values": precision,
                    "recall_values": recall,
                },
                "plot_params": {
                    "x_label": "Recall",
                    "y_label": "Precision",
                    "kind": "line",
                    "x_key": "recall_values",
                    "y_key": "precision_values",
                },
                "stats": stats,
            },
            errors=[warning] if warning else None,
        )


__all__ = ["PrecisionRecallDistribution"]
