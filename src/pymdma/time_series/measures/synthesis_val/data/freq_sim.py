from typing import Optional

import numpy as np
import ot
from scipy.integrate import simps
from scipy.signal import coherence, welch

from pymdma.common.definitions import FeatureMetric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


class SpectralCoherence(FeatureMetric):
    """Compute the mean coherence between real and synthetic signals. Returns
    the average (median-based) coherence across signals.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    fs: int, optional, default=2048
        The sampling frequency of the signal.
    valid_freq_range: tuple, optional, default=(-np.inf, np.inf)
        The range of valid frequencies for the signal.
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
        fs: int = 2048,
        nperseg: Optional[int] = None,
        valid_freq_range: tuple = (-np.inf, np.inf),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fs = fs
        self.valid_freq_range = valid_freq_range
        self.nperseg = nperseg

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Compute the metric.

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

        coherence_values = []
        # Wrap single 1D signals as a list
        if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
            real_data = [real_data]
        if isinstance(syn_data, np.ndarray) and syn_data.ndim == 1:
            syn_data = [syn_data]

        # Pair up signals one-to-one
        for real_sig, synth_sig in zip(real_data, syn_data):
            win = min(len(real_sig) // 8, self.fs // 2) if self.nperseg is None else self.nperseg
            f, Cxy = coherence(real_sig, synth_sig, fs=self.fs, nperseg=win, window="boxcar")
            valid_freqs = (f >= self.valid_freq_range[0]) & (f <= self.valid_freq_range[1])
            coherence_values.append(np.median(Cxy[valid_freqs]))

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": np.mean(coherence_values),
            },
        )


class SpectralWassersteinDistance(FeatureMetric):
    """Compute a Wasserstein distance in the frequency domain based on PSD
    differences.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    fs: int, optional, default=2048
        The sampling frequency of the signal.
    valid_freq_range: tuple, optional, default=(-np.inf, np.inf)
        The range of valid frequencies for the signal.
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
        fs: int = 2048,
        nperseg: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fs = fs
        self.nperseg = nperseg

    def _compute_power_spectral_density(self, data) -> np.ndarray:
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = [data]

        psd = []
        for sig in data:
            # Use a segment length (nperseg) to compute PSD:
            win = min(len(sig), 4 * self.fs) if self.nperseg is None else self.nperseg
            psd.append(welch(sig, self.fs, nperseg=win, window="boxcar")[1])
        return np.array(psd)

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Compute the metric."""
        real_psd = self._compute_power_spectral_density(real_data)
        syn_psd = self._compute_power_spectral_density(syn_data)

        # Align number of frequency bins
        min_len = min(real_psd.shape[1], syn_psd.shape[1])
        real_psd = real_psd[:, :min_len]
        syn_psd = syn_psd[:, :min_len]

        real_psd = real_psd.reshape(-1, min_len)
        syn_psd = syn_psd.reshape(-1, min_len)

        # Cost matrix in PSD space (Euclidean)
        cost_matrix_psd = ot.dist(real_psd, syn_psd, metric="euclidean")
        # Uniform distributions
        a = np.ones(real_psd.shape[0]) / real_psd.shape[0]
        b = np.ones(syn_psd.shape[0]) / syn_psd.shape[0]

        # Sinkhorn regularized Wasserstein distance
        wd_psd = ot.sinkhorn2(a, b, cost_matrix_psd, reg=0.01)

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": wd_psd,
            },
        )


__all__ = ["SpectralCoherence", "SpectralWassersteinDistance"]
