import math
from typing import List

import numpy as np
import torch
from loguru import logger
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

# TODO review documentations and attributes


class PSNR(Metric):
    """Computes Peak Signal to Noise Ratio (PSNR) between two images.

    **Objective**: Signal-to-noise ratio

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unsused).

    References
    ----------
    geeksforgeeks.org: Python | Peak Signal to Noise Ratio (PSNR)
    https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

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
    max_value: float = 1.0

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(
        self,
        reference_imgs: List[np.ndarray],
        target_imgs: List[np.ndarray],
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
        psnrs = []
        warnings = set()
        for ref, targ in zip(reference_imgs, target_imgs):
            mse = np.mean((ref - targ) ** 2)
            if mse == 0:  # MSE is zero means no noise is present in the signal (same image)
                psnrs.append(float("inf"))
                warnings.add("Infinite PSNR value. Check if reference image is the same as the target image.")
                logger.warning("Infinite PSNR value. Check if reference image is the same as the target image.")
                continue

            max_pixel = 255.0
            psnrs.append(20 * math.log10(max_pixel / math.sqrt(mse)))

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": psnrs},
            errors=list(warnings) if len(warnings) > 0 else None,
        )


class SSIM(Metric):
    """Computes the Structural Similarity Index (SSIM) between two images.
    Wrapper of the torchmetrics.image.StructuralSimilarityIndexMeasure.

    **Objective**: Similarity

    Parameters
    ----------
    data_range : int
        Maximum value of the image data (e.g., 255 for 8-bit images).
    device : str
        Device to use for computation.
    same_size : bool
        If True, all images must have the same size.
        Images will be stacked and computed in a single batch.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unsused).

    References
    ----------
    This class is a wrapper of the implementation from:
    torchmetrics, TorchMetrics - Measuring Reproducibility in PyTorch.
    https://github.com/Lightning-AI/torchmetrics

    Examples
    --------
    >>> ssim = SSIM()
    >>> reference_imgs = np.random.rand(20, 256, 256, 3) # (N, H, W, C)
    >>> target_imgs = np.random.rand(20, 256, 256, 3) # (N, H, W, C)
    >>> result: MetricResult = ssim.compute(reference_imgs, target_imgs)
    """

    reference_type = ReferenceType.INSTANCE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        data_range: int = 255,
        device: str = "cpu",
        same_size: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction="none", **kwargs)
        self._batch_calculation = same_size
        self.device = device

    def _preprocess_image(self, img):
        # (H, W, C) -> (C, H, W)
        assert img.ndim == 3, "Image must have 3 dimensions (H, W, C)"
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

    def compute(
        self,
        reference_imgs: List[np.ndarray],
        target_imgs: List[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Computes SSIM score.

        Parameters
        ----------
        reference_imgs : (N, H, W, C) ndarray
            Images to use as reference.
            List of arrays representing a RGB image of shape (H, W, C).
        target_imgs : (N, H, W, C) ndarray
            Corresponding images to compare with reference.
            List of arrays representing a RGB image of shape (H, W, C).

        Returns
        -------
        result : MetricResult
            Instance-level SSIM scores.
        """
        reference_imgs = [self._preprocess_image(img) for img in reference_imgs]
        target_imgs = [self._preprocess_image(img) for img in target_imgs]

        if self._batch_calculation:
            reference_imgs = torch.stack(reference_imgs).to(self.device)
            target_imgs = torch.stack(target_imgs).to(self.device)
            ssims = self._ssim(reference_imgs, target_imgs).cpu().tolist()
        else:
            ssims = []
            for ref, targ in zip(reference_imgs, target_imgs):
                ssims.append(self._ssim(ref.unsqueeze(0).to(self.device), targ.unsqueeze(0).to(self.device)).item())

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": ssims},
        )


class MSSIM(Metric):
    """Computes the Multiscale Structural Similarity Index (MSSIM) between two
    images. Wrapper of the
    torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure.

    **Objective**: Similarity

    Parameters
    ----------
    data_range : int
        Maximum value of the image data (e.g., 255 for 8-bit images).
    device : str
        Device to use for computation.
    same_size : bool
        If True, all images must have the same size.
        Images will be stacked and computed in a single batch.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    This class is a wrapper of the implementation from:
    torchmetrics, TorchMetrics - Measuring Reproducibility in PyTorch.
    https://github.com/Lightning-AI/torchmetrics

    Examples
    --------
    >>> mssim = MSSIM()
    >>> reference_imgs = np.random.rand(20, 256, 256, 3) # (N, H, W, C)
    >>> target_imgs = np.random.rand(20, 256, 256, 3) # (N, H, W, C)
    >>> result: MetricResult = mssim.compute(reference_imgs, target_imgs)
    """

    reference_type = ReferenceType.INSTANCE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        data_range: int = 255,
        device: str = "cpu",
        same_size: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range, reduction="none", **kwargs)
        self._batch_calculation = same_size
        self.device = device

    def _preprocess_image(self, img):
        # (H, W, C) -> (C, H, W)
        assert img.ndim == 3, "Image must have 3 dimensions (H, W, C)"
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

    def compute(
        self,
        reference_imgs: List[np.ndarray],
        target_imgs: List[np.ndarray],
        **kwargs,
    ) -> MetricResult:
        """Computes MSSIM score.

        Parameters
        ----------
        reference_imgs : (N, H, W, C) ndarray
            Images to use as reference.
            List of arrays representing a RGB image of shape (H, W, C).
        target_imgs : (N, H, W, C) ndarray
            Corresponding images to compare with reference.
            List of arrays representing a RGB image of shape (H, W, C).

        Returns
        -------
        result : MetricResult
            Instance-level MSSIM scores.
        """

        reference_imgs = [self._preprocess_image(img) for img in reference_imgs]
        target_imgs = [self._preprocess_image(img) for img in target_imgs]

        if self._batch_calculation:
            reference_imgs = torch.stack(reference_imgs).to(self.device)
            target_imgs = torch.stack(target_imgs).to(self.device)

            ssims = self._ssim(reference_imgs, target_imgs).cpu().tolist()
        else:
            ssims = []
            for ref, targ in zip(reference_imgs, target_imgs):
                ssims.append(self._ssim(ref.unsqueeze(0).to(self.device), targ.unsqueeze(0).to(self.device)).item())

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": ssims},
        )


__all__ = ["PSNR", "SSIM", "MSSIM"]
