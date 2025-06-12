from typing import List, Optional, Union

import numpy as np
import ot
from scipy.signal import coherence, welch

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


class SpectralCoherence(Metric):
    """Compute the mean coherence between real and synthetic signals. Returns
    the average (median-based) coherence across signals.

    **Objective**: Similarity

    Parameters
    ----------
    fs: int, optional, default=2048
        The sampling frequency of the signal.
    nperseg: int, optional, default=None
        Length of each segment used to compute the power spectral density.
    valid_freq_range: tuple, optional, default=(-np.inf, np.inf)
        The range of valid frequencies for the signal.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    F. P. Carrle, Y. Hollenbenders, and A. Reichenbach, Generation of synthetic EEG data for training algorithms supporting the diagnosis of major depressive disorder (2023).
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1219133/full.

    J. Vetter, J. H. Macke, and R. Gao, Generating realistic neurophysiological time series with denoising diffusion probabilistic models (2024)
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11573898/

    A. Zancanaro, I. Zoppis, S. Manzoni, and G. Cisotto, vEEGNet: A New Deep Learning Model to Classify and Generate EEG (2023)
    https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0011990800003476.

    Examples
    --------
    >>> coeherence = SpectralCoherence()
    >>> real_data = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> fake_data = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> result: MetricResult = coherence.compute(real_data, fake_data)
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

    def _preprocess(self, data: Union[np.ndarray, List[np.ndarray]]):
        """Preprocess data for computation.

        Parameters
        ----------
        data : Union[np.ndarray, List[np.ndarray]]
            Input data, either a 1D, 2D, or 3D array.

        Returns
        -------
        data : np.ndarray
            Preprocessed data, a 2D array.
        """
        data = np.array(data)
        # 1D signals need to be converted to 2D
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        elif data.ndim > 2:
            # Convert to 2D
            data = data.reshape(data.shape[0], -1)
        return data

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Compute the metric.

        Parameters
        ----------
        real_features : array-like of shape (N, L) or (N, L, C)
            Array with the original samples.
        fake_features : array-like of shape (N, L) or (N, L, C)
            Array with the fake samples.
        **kwargs : dict, optional
            Additional keyword arguments for compatibility (unused).

        Returns
        -------
        MetricResult
            Dataset-level results for the Spectral Coherence.
        """

        coherence_values = []

        real_data = self._preprocess(real_data)
        syn_data = self._preprocess(syn_data)

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


class SpectralWassersteinDistance(Metric):
    """Compute a Wasserstein distance in the frequency domain based on Power
    Spectral Density (PSD) differences.

    **Objective**: Similarity

    Parameters
    ----------
    fs: int, optional, default=2048
        The sampling frequency of the signal.
    nperseg: int, optional, default=None
        Length of each segment used to compute the power spectral density.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    F. P. Carrle, Y. Hollenbenders, and A. Reichenbach, Generation of synthetic EEG data for training algorithms supporting the diagnosis of major depressive disorder (2023).
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1219133/full.

    Examples
    --------
    >>> spectral_wd = SpectralWassersteinDistance()
    >>> real_data = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> fake_data = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> result: MetricResult = spectral_wd.compute(real_data, fake_data)
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
        """Compute the power spectral density of the given data.

        Parameters
        ----------
        data: array-like of shape (N,) or (N, L) or (N, L, C)
            The input data.

        Returns
        -------
        psd: array-like of shape (N, L) or (N, L, C)
            The power spectral density of the input data.
        """
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = [data]

        psd = []
        for sig in data:
            # Use a segment length (nperseg) to compute PSD:
            win = min(len(sig), 4 * self.fs) if self.nperseg is None else self.nperseg
            psd.append(welch(sig, self.fs, nperseg=win, window="boxcar")[1])
        return np.array(psd)

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Compute the metric.

        Parameters
        ----------
        real_features : array-like of shape (N, L) or (N, L, C)
            Array with the original samples.
        fake_features : array-like of shape (N, L) or (N, L, C)
            Array with the fake samples.
        **kwargs : dict, optional
            Additional keyword arguments for compatibility (unused).

        Returns
        -------
        MetricResult
            Dataset-level results for the Spectral Coherence.
        """
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
