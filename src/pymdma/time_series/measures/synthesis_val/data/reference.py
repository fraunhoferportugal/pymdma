from typing import List, Literal

import numpy as np
from fastdtw import fastdtw

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


class DTW(Metric):
    """Computes the Dynamic Time Warping (DTW) distance between two sets of
    time-series signals, evaluating the similarity between corresponding
    channels in the target and reference signals. The DTW distance is computed
    by comparing each target signal with every reference signal, with lower DTW
    values indicating greater similarity. For each signal pair, the DTW
    distance is calculated across all channels, with the mean of the distances
    being taken across all channels and instances. This process yields both
    instance-level and dataset-level DTW metrics.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility with the Metric framework.

    References
    ----------
    Salvador, S., & Chan, P., FastDTW: Toward Accurate Dynamic Time (2004).
    https://dl.acm.org/doi/10.5555/1367985.1367993

    Examples
    --------
    >>> dtw = DTW()
    >>> reference_sigs = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> target_imgs = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> result: MetricResult = dtw.compute(reference_sigs, target_sigs)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(
        self,
        reference_sigs: List[np.ndarray],
        target_sigs: List[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Computes Dinamic Time Wrapping.

        Parameters
        ----------
        reference_sigs: (N, L, C) ndarray
            Signals to use as reference.
            List of arrays representing a signal of shape (L, C).
        target_sigs : (N, L, C) ndarray
            Signals compare with reference.
            List of arrays representing a signal of shape (L, C).

        Returns
        -------
        result : MetricResult
            Instance-level dtw.
            Dataset-level dtw.
        """
        instance_dtw = []
        for targ in target_sigs:
            for ref in reference_sigs:
                dtw_values_by_chan = []

                # Compute the dtw for each channel
                for targ_channel, ref_channel in zip(targ.T, ref.T):
                    channel_dtw, _ = fastdtw(targ_channel, ref_channel)

                    dtw_values_by_chan.append(channel_dtw)

                # Compute mean dtw across channels
                mean_dtw = np.mean(dtw_values_by_chan)

            # Correlations by signal
            instance_dtw.append(mean_dtw)

        # Average correlation across signals
        final_dtw = np.mean(instance_dtw)

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": instance_dtw},
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": final_dtw},
        )


class CrossCorrelation(Metric):
    """Computes the Cross-Correlation between two sets of signals.

    This function calculates the cross-correlation to analyze the relationship between
    corresponding channels in the target and reference signals. The computation is performed
    for each signal in the target set against every signal in the reference set, using a
    specified overlap mode. The computed cross-correlation for each channel can be summarized
    using one of two reduction methods: 'mean'and 'max'.

    For each signal pair, the function calculates the cross-correlation values for each channel
    using the specified reduction method. It then computes the mean of these values across all
    channels to provide an instance-level metric and averages these results across all signal
    pairs to obtain the dataset-level metric.

    **Objective**: Fidelity, Diversity

    Parameters
    ----------
    mode : {'full', 'same', 'valid'}, optional
        Defines how the cross-correlation is computed. Default is 'full'.
        - 'full': Computes the convolution at every point of overlap, producing
          an output of size (N + M - 1). Boundary effects may be present.
        - 'same': Produces an output of length max(M, N), centered on the signals.
        - 'valid': Produces an output of length max(M, N) - min(M, N) + 1,
          considering only complete overlaps.
    reduction : {'mean', 'max'}, optional
        Determines how the cross-correlation is summarized for each channel. Default is 'max'.
        - 'mean': The average of the cross-correlation values for the channel.
        - 'max': The maximum cross-correlation value for the channel.
    **kwargs : dict, optional
        Additional keyword arguments for customization.

    References
    ----------
    Proakis and Manolakis, Digital Signal Processing: Principles, Algorithms, and Applications (1996).
    https://dl.acm.org/doi/10.5555/227373

    Examples
    --------
    >>> cc = CrossCorrelation()
    >>> reference_sigs = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> target_sigs = np.random.rand(64, 1000, 12) # (N, L, C)
    >>> result: MetricResult = cc.compute(reference_sigs, target_sigs)
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = [EvaluationLevel.INSTANCE, EvaluationLevel.DATASET]
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = -np.inf
    max_value: float = np.inf

    def __init__(
        self,
        mode: Literal["full", "same", "valid"] = "full",
        reduction: Literal["mean", "max"] = "max",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert mode in ["full", "same", "valid"], f"Unsupported mode for Cross Correlation: {mode}"
        self.mode = mode
        assert reduction in ["mean", "max"], f"Unsupported criteria for relative tenengrad: {reduction}"
        self.reduction = reduction

    def compute(
        self,
        reference_sigs: List[np.ndarray],
        target_sigs: List[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Computes Dinamic Time Wrapping.

        Parameters
        ----------
        reference_sigs: (N, L1, C) ndarray
            Signals to use as reference.
            List of arrays representing a signal of shape (L1, C).
        target_sigs :(N, L2, C) ndarray
            Signals compare with reference.
            List of arrays representing a signal of shape (L2, C).

        Returns
        -------
        result : MetricResult
            Instance-level maximum cross-correlation.
            Dataset-level maximum cross-correlation.
        """
        instance_cross_corr = []

        for targ in target_sigs:
            for ref in reference_sigs:
                reduct_corr_values_by_chan = []

                # Compute the cross-correlation for each channel with different reductions
                for targ_channel, ref_channel in zip(targ.T, ref.T):
                    cross_corr = np.correlate(targ_channel, ref_channel, mode=self.mode)

                    if self.reduction == "max":
                        max_corr_idx = np.argmax(cross_corr)
                        max_corr_value = cross_corr[max_corr_idx]
                        reduct_corr_values_by_chan.append(max_corr_value)

                    else:
                        mean_corr_value = np.mean(cross_corr)
                        reduct_corr_values_by_chan.append(mean_corr_value)

                # Compute the mean of the correlation across channels
                cross_corr = np.mean(reduct_corr_values_by_chan)

            # Correlations by signal
            instance_cross_corr.append(cross_corr)

        # Average correlation across signals
        final_cross_corr = np.mean(instance_cross_corr)

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": instance_cross_corr},
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "float", "value": final_cross_corr},
        )


__all__ = ["DTW", "CrossCorrelation"]
