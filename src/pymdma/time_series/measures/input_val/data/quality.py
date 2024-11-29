from typing import Union

import numpy as np

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


class Uniqueness(Metric):
    """Computes the percentage of consecutive equal values in the input
    signals. For multidimensional input signals, it considers the average of
    the consecutive values across the signal dimensions (e.g., leads).

    **Objective**: Uniqueness

    Parameters
    ----------
    tolerance : float, optional, default=0.0001
        The tolerance level for considering consecutive values as equal.
        A smaller tolerance means stricter equality. Defaults to 0.0001.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    Examples
    --------
    >>> uniqueness = Uniqueness()
    >>> sigs = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> result: MetricResult = uniqueness.compute(sigs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 100

    def __init__(
        self,
        tolerance: float = 0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tolerance = tolerance

    def compute(self, signals: Union[np.ndarray, list], **kwargs) -> MetricResult:
        """Computes the ratio of consecutive equal values in the input signals.
        For multidimensional input signals, it considers the average of the
        consecutive values across the signal dimensions (e.g., leads).

        Parameters
        ----------
        signals : list or ndarray of shape (batch_size, signal_len, nr_dims)
            A batch of signals.

        Returns
        -------
        MetricResult
            A list of shape (batch_size) containing the percentage of consecutive values in each signal.
        """
        signals_np = np.array(signals)

        differences = np.abs(signals_np[:, :-1, :] - signals_np[:, 1:, :])

        within_tolerance = differences <= self.tolerance

        consecutive_equal_values = list(np.mean(within_tolerance, axis=(1, 2)))

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": consecutive_equal_values},
        )


class SNR(Metric):
    """The signal-to-noise ratio of the input data computed as the average of
    the mean to standard deviation ratio across signal dimensions.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Smith, S. W., The Scientist and Engineer's Guide to Digital Signal Processing (1997).
    https://dl.acm.org/doi/10.5555/281875

    Examples
    --------
    >>> snr = SNR()
    >>> sigs = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> result: MetricResult = snr.compute(sigs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(self, signals: Union[np.ndarray, list], **kwargs) -> MetricResult:
        """The signal-to-noise ratio of the input data computed as the average
        of the mean to standard deviation ratio across signal dimensions.

        Parameters
        ----------
        signals : list or ndarray of shape (batch_size, signal_len, nr_dims)
            A batch of signals.

        Returns
        -------
        MetricResult
            List of shape (batch_size) with the signal-to-noise ratio of each signal.
        """
        snrs = []
        for signal in signals:
            signal_mean = np.mean(signal, axis=0)
            signal_std = np.std(signal, axis=0)

            # Compute SNR using standard deviation method
            snr = np.mean(signal_mean / signal_std)
            snrs.append(snr)

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": snrs},
        )


__all__ = ["Uniqueness", "SNR"]
