from typing import Sequence

import numpy as np
from loguru import logger

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

# from https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py#L736
_yiq_from_rgb = np.array(
    [
        [0.299, 0.587, 0.114],
        [0.59590059, -0.27455667, -0.32134392],
        [0.21153661, -0.52273617, 0.31119955],
    ],
).T.astype(np.float32)


class PSNR(Metric):
    """Computes Peak Signal to Noise Ratio (PSNR) between two images.

    **Objective**: Signal-to-noise ratio

    Parameters
    ----------
    data_range : int, optional, default=255
        Maximum value of the image data (e.g., 255 for 8-bit images).
    convert_to_grayscale : bool, optional, default=False
        Whether to convert the images to grayscale before computing PSNR.
    allow_nan : bool, optional, default=False
        Whether to allow NaN values in the returned scores.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unsused).

    References
    ----------
    wikipedia.org: Peak signal-to-noise ratio
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Examples
    --------
    >>> psnr = PSNR()
    >>> reference_imgs = np.random.rand(20, 256, 256, 3) # (N, H, W, C)
    >>> target_imgs = np.random.rand(20, 256, 256, 3) # (N, H, W, C)
    >>> result: MetricResult = psnr.compute(reference_imgs, target_imgs)
    """

    reference_type = ReferenceType.INSTANCE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(
        self,
        data_range: int = 255,
        convert_to_grayscale: bool = False,
        allow_nan: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_range = data_range
        self.convert_to_grayscale = convert_to_grayscale
        self.allow_nan = allow_nan
        self._eps = 1e-8 if not allow_nan else 0

    def compute(
        self,
        reference_imgs: Sequence[np.ndarray],
        target_imgs: Sequence[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Computes PSNR score between two image sets.

        Parameters
        ----------
        reference_imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            Images to use as reference.
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.
        target_imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            Corresponding images to compare with reference.
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result : MetricResult
            Instance-level PSNR scores.
            May contain warnings if PSNR is infinite.
        """
        assert len(reference_imgs) == len(target_imgs), "Reference and target images must have the same length"
        assert all(
            ref.shape == targ.shape for ref, targ in zip(reference_imgs, target_imgs)
        ), "Reference and target images must have the same shape"

        # convert images to YIQ and extract luminance
        if self.convert_to_grayscale:
            reference_imgs = [np.sum(ref @ _yiq_from_rgb, axis=-1) for ref in reference_imgs]
            target_imgs = [np.sum(targ @ _yiq_from_rgb, axis=-1) for targ in target_imgs]

        mses = np.array(
            [
                np.mean((ref.astype(np.float32) - targ.astype(np.float32)) ** 2)
                for ref, targ in zip(reference_imgs, target_imgs)
            ],
        )
        psnrs = 20 * np.log10(self.data_range / (np.sqrt(mses) + self._eps))

        warnings = set()
        if np.isinf(psnrs).any():
            warnings.add("Infinite PSNR value. Check if reference image is the same as the target image.")
            logger.warning("Infinite PSNR value. Check if reference image is the same as the target image.")

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": psnrs.tolist()},
            errors=list(warnings) if len(warnings) > 0 else None,
        )


__all__ = ["PSNR"]
