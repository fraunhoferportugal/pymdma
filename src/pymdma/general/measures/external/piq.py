from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from piq import FID as _FID
from piq import GS as _GS
from piq import MSID as _MSID
from torch import Tensor

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ...functional.ratio import dispersion_ratio, distance_ratio
from ...utils.util import features_splitting, to_tensor


class FrechetDistance(FeatureMetric):
    """Frechet Distance (FD) metric wrapper from the PIQ implementation of FID.
    Allows the computation of the dispersion and distance ratios for the
    metric.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    compute_ratios : bool, optional
        If set to True, the dispersion and distance ratios will be computed. Defaults to True.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    Notes
    -----
    This implementation is based on the PIQ library. The base extractor model is InceptionV3, but this implementation allows
    for the use of other embedding models (useful when the synthetic data is not compatible with Inception models).

    See Also
    --------
    general.functional.ratio.dispersion_ratio : Compute the dispersion ratio for the Frechet Distance metric.
    general.functional.ratio.distance_ratio : Compute the distance ratio for the Frechet Distance metric.

    References
    ----------
    Kastryulin et al., PyTorch Image Quality: Metrics for Image Quality Assessment (2022).
    https://arxiv.org/abs/2208.14818

    piq, PyTorch Image Quality: Metrics and Measure for Image Quality Assessment,
    https://github.com/photosynthesis-team/piq

    Examples
    --------
    >>> fid = FrechetDistance()
    >>> x_feats = np.random.rand(100, 100)
    >>> y_feats = np.random.rand(100, 100)
    >>> result: MetricResult = fid.compute(x_feats, y_feats)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    extractor_model_name: str = "inception_v3"

    def __init__(
        self,
        compute_ratios: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_ratios = compute_ratios
        self._fid_inst = _FID()

    def _compute_ratios(self, real_features: np.ndarray, fake_features: np.ndarray, **kwargs) -> Dict[str, float]:
        """Compute the dispersion and distance ratios for the Frechet Distance
        metric.

        Parameters
        ----------
        real_features : np.ndarray
            Array of shape (n_samples, n_features) containing features of real samples.
        fake_features : np.ndarray
            Array of shape (n_samples, n_features) containing features of fake/generated samples.

        Returns
        -------
        ratios: dict
            Dictionary containing the dispersion and distance ratio values.
        """
        state = kwargs.get("context", {})

        if any(key not in state for key in {"x_split_1", "x_split_2", "y_split_1", "y_split_2"}):
            state["x_split_1"], state["x_split_2"] = features_splitting(real_features, seed=0)
            state["y_split_1"], state["y_split_2"] = features_splitting(fake_features, seed=0)

        return {
            "dispersion_ratio": dispersion_ratio(
                self._fid_inst.compute_metric,
                to_tensor(state["x_split_1"]),
                to_tensor(state["x_split_2"]),
                to_tensor(state["y_split_1"]),
                to_tensor(state["y_split_2"]),
            ),
            "distance_ratio": distance_ratio(
                self._fid_inst.compute_metric,
                to_tensor(state["x_split_1"]),
                to_tensor(state["x_split_2"]),
                to_tensor(state["y_split_1"]),
                to_tensor(state["y_split_2"]),
            ),
        }

    def compute(
        self,
        real_features: Union[Tensor, np.ndarray],
        fake_features: Union[Tensor, np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Compute the Frechet Distance metric.

        Parameters
        ----------
        real_features : Union[Tensor, np.ndarray]
            Array-like of shape (n_samples, n_features) containing features of real samples.
        fake_features : Union[Tensor, np.ndarray]
            Array-like of shape (n_samples, n_features) containing features of fake/generated samples.

        Returns
        -------
        result : MetricResult
            Dataset-level FD score and the dispersion and distance ratios.
        """
        real_features = to_tensor(real_features)
        fake_features = to_tensor(fake_features)

        fid = self._fid_inst.compute_metric(real_features, fake_features)

        ratios = None
        if self.compute_ratios:
            ratios = self._compute_ratios(real_features, fake_features, **kwargs)

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": fid.detach().item(),
                "stats": ratios,
            },
        )


class GeometryScore(FeatureMetric):
    """Geometry Score (GS) metric wrapper from the PIQ implementation of GS.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    sample_size : int, optional
        Number of samples to use for the GS computation. Defaults to 128.
    num_iters : int, optional
        Number of iterations to use for the GS computation. Defaults to 1000.
    gamma : float, optional
        Gamma parameter for the GS computation. Defaults to None.
    i_max : int, optional
        Maximum number of iterations for the GS computation. Defaults to 10.
    num_workers : int, optional
        Number of workers to use for the GS computation. Defaults to 4.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    References
    ----------
    Kastryulin et al., PyTorch Image Quality: Metrics for Image Quality Assessment (2022).
    https://arxiv.org/abs/2208.14818

    piq, PyTorch Image Quality: Metrics and Measure for Image Quality Assessment,
    https://github.com/photosynthesis-team/piq

    Examples
    --------
    >>> gs = GeometryScore()
    >>> x_feats = np.random.rand(100, 100)
    >>> y_feats = np.random.rand(100, 100)
    >>> result: MetricResult = gs.compute(x_feats, y_feats)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        sample_size: int = 128,
        num_iters: int = 1000,
        gamma: Optional[float] = None,
        i_max: int = 10,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_size = sample_size
        self.num_iters = num_iters
        self.gamma = gamma
        self.i_max = i_max
        self.num_workers = num_workers

        self._gs = _GS(
            sample_size=self.sample_size,
            num_iters=self.num_iters,
            gamma=self.gamma,
            i_max=self.i_max,
            num_workers=self.num_workers,
        )

    def compute(
        self,
        real_features: Union[Tensor, np.ndarray],
        fake_features: Union[Tensor, np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Compute the Geometry Score metric.

        Parameters
        ----------
        real_features : Union[Tensor, np.ndarray]
            Array-like of shape (n_samples, n_features) containing features of real samples.
        fake_features : Union[Tensor, np.ndarray]
            Array-like of shape (n_samples, n_features) containing features of fake/generated samples.

        Returns
        -------
        result : MetricResult
            Dataset-level GS score.
        """
        real_features = to_tensor(real_features)
        fake_features = to_tensor(fake_features)

        score = self._gs.compute_metric(real_features, fake_features)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": score.detach().item()},
        )


class MultiScaleIntrinsicDistance(FeatureMetric):
    """Multi-Scale Intrinsic Distance (MSID) metric wrapper from the PIQ
    implementation of MSID.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    ts : Optional[Tensor], optional
        Tensor of shape (n_samples, n_features) containing the temperature values. Defaults to None.
    k_neighbours : int, optional
        Number of nearest neighbours to consider. Defaults to 5.
    m_steps : int, optional
        Number of steps for the MSID computation. Defaults to 10.
    niters : int, optional
        Number of iterations for the MSID computation. Defaults to 100.
    rademacher : bool, optional
        Whether to use Rademacher distribution for the MSID computation. Defaults to False.
        When not active will use standard normal for random vectors in Hutchinson.
    normalized_laplacian : bool, optional
        Whether to normalize the laplacian for the MSID computation. Defaults to True.
    normalize : Literal["empty", "complete", "er", "none"], optional
        Normalization strategy for the laplacian. Defaults to "empty".
    msid_mode : Literal["l2", "max"], optional
        Mode for the MSID computation. Defaults to "max".
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    Notes
    -----
    The results of this metric are based on random approximations, so they are not deterministic.
    In some datasets the results can be unstable. This can be mitigated by increasing the
    number of iterations with the `niters` parameter.


    References
    ----------
    Kastryulin et al., PyTorch Image Quality: Metrics for Image Quality Assessment (2022).
    https://arxiv.org/abs/2208.14818

    piq, PyTorch Image Quality: Metrics and Measure for Image Quality Assessment,
    https://github.com/photosynthesis-team/piq

    Examples
    --------
    >>> msid = MultiScaleIntrinsicDistance()
    >>> x_feats = np.random.rand(100, 100)
    >>> y_feats = np.random.rand(100, 100)
    >>> result: MetricResult = msid.compute(x_feats, y_feats)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    extractor_model_name: str = "inception_v3"

    def __init__(
        self,
        ts: Optional[Union[Tensor, np.ndarray]] = None,
        k_neighbours: int = 5,
        m_steps: int = 10,
        niters: int = 100,
        rademacher: bool = False,
        normalized_laplacian: bool = True,
        normalize: Literal["empty", "complete", "er", "none"] = "empty",
        msid_mode: Literal["l2", "max"] = "max",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ts = torch.from_numpy(ts) if isinstance(ts, np.ndarray) else ts
        self.k_neighbours = k_neighbours
        self.m_steps = m_steps
        self.niters = niters
        self.rademacher = rademacher
        self.normalized_laplacian = normalized_laplacian
        self.normalize = normalize
        self.msid_mode = msid_mode

        self._msid = _MSID(
            ts=self.ts,
            k=self.k_neighbours,
            m=self.m_steps,
            niters=self.niters,
            rademacher=self.rademacher,
            normalized_laplacian=self.normalized_laplacian,
            normalize=self.normalize,
            msid_mode=self.msid_mode,
        )

    def compute(
        self,
        real_features: Union[Tensor, np.ndarray],
        fake_features: Union[Tensor, np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Compute the Multi-Scale Intrinsic Distance metric.

        Parameters
        ----------
        real_features : Union[Tensor, np.ndarray]
            Array-like of shape (n_samples, n_features) containing features of real samples.
        fake_features : Union[Tensor, np.ndarray]
            Array-like of shape (n_samples, n_features) containing features of fake/generated samples.

        Returns
        -------
        result : MetricResult
            Dataset-level MSID score.
        """
        real_features = to_tensor(real_features)
        fake_features = to_tensor(fake_features)
        score = self._msid.compute_metric(real_features, fake_features)
        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": score.detach().item()},
        )


__all__ = ["FrechetDistance", "GeometryScore", "MultiScaleIntrinsicDistance"]
