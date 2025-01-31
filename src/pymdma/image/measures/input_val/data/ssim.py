from typing import Optional, Sequence, Union

import numpy as np
import torch
from piq import MultiScaleSSIMLoss, SSIMLoss
from torch import Tensor

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


def _process_image(img: np.ndarray, channel_last_dim: bool) -> Tensor:
    """Process the input image to match the requirements of the SSIM metric.

    Args:
        img: The input image to process.
        channel_last_dim: Whether to permute the image to have the color channel as the last dimension.

    Returns:
        The processed image as a tensor.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if channel_last_dim:
        img = img.permute(2, 0, 1)
    return img.float()


def _validate_inputs(
    refs: Sequence[Union[Tensor, np.ndarray]],
    targets: Sequence[Union[Tensor, np.ndarray]],
    channel_last_dim: bool,
) -> Tensor:
    assert len(refs) == len(targets), "Number of reference and target images must be equal"
    assert all(
        ref.shape == targ.shape for ref, targ in zip(refs, targets)
    ), "Reference and target images must have the same shape"

    refs = [_process_image(ref, channel_last_dim) for ref in refs]
    targets = [_process_image(target, channel_last_dim) for target in targets]
    return refs, targets


class SSIM(Metric):
    """Computes the Structural Similarity Index (SSIM) between two images.
    Wrapper of the PIQ SSIMLoss implementation.

    **Objective**: Similarity

    Parameters
    ----------
    kernel_size : int
        The side-length of the sliding window used in comparison. Must be an odd value.
    sigma : float
        Sigma of normal distribution.
    k1 : float (default: 0.01)
        Algorithm parameter, K1 (small constant).
    k2 : float (default: 0.03)
        Algorithm parameter, K2 (small constant).
    downsample : bool (default: False)
        Whether to downsample the images to speed up the computation.
    data_range : int
        Maximum value of the image data (e.g., 255 for 8-bit images).
    channel_last_dim : bool (default: True)
        Indicates if the input images have the color channel as the last dimension.
        Needed for compatibility with torch operations.
    device : str
        Device to use for computation.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unsused).

    References
    ----------
    This class is a wrapper of the implementation from:
    piq, PyTorch Image Quality: Metrics and Measure for Image Quality Assessment,
    https://github.com/photosynthesis-team/piq

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
        kernel_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        downsample: bool = False,
        data_range: int = 255,
        channel_last_dim: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ssim = SSIMLoss(
            kernel_size=kernel_size,
            kernel_sigma=sigma,
            k1=k1,
            k2=k2,
            downsample=downsample,
            data_range=data_range,
            reduction="none",
        ).to(device)
        self.channel_last_dim = channel_last_dim
        self.device = device

    def compute(
        self,
        reference_imgs: Sequence[Union[np.ndarray, Tensor]],
        target_imgs: Sequence[Union[np.ndarray, Tensor]],
        **kwargs,
    ) -> MetricResult:
        """Computes SSIM score.

        Parameters
        ----------
        reference_imgs : (N, H, W, C) ndarray or tensor
            Images to use as reference.
            List of arrays representing a RGB image of shape (H, W, C).
            For (C, H, W) images, set channel_last_dim to False.
        target_imgs : (N, H, W, C) ndarray or tensor
            Corresponding images to compare with reference.
            List of arrays representing a RGB image of shape (H, W, C).
            For (C, H, W) images, set channel_last_dim to False.

        Returns
        -------
        result : MetricResult
            Instance-level SSIM scores.
        """
        reference_imgs, target_imgs = _validate_inputs(reference_imgs, target_imgs, self.channel_last_dim)

        # do batch calculation on all images (if same shape)
        _batch_calculation = len({ref.shape for ref in reference_imgs}) == 1
        if _batch_calculation:
            reference_imgs = torch.stack(reference_imgs).to(self.device)
            target_imgs = torch.stack(target_imgs).to(self.device)
            ssims = self._ssim(reference_imgs, target_imgs).detach().cpu()
            ssims = (1 - ssims).tolist()
        else:
            ssims = []
            for ref, targ in zip(reference_imgs, target_imgs):
                ssims.append(1 - self._ssim(ref.unsqueeze(0).to(self.device), targ.unsqueeze(0).to(self.device)).item())

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": ssims},
        )


class MSSSIM(Metric):
    """Computes the Multiscale Structural Similarity Index (MSSSIM) between two
    images. Wrapper of the PIQ MultiScaleSSIMLoss implementation.

    **Objective**: Similarity

    Parameters
    ----------
    kernel_size : int
        The side-length of the sliding window used in comparison. Must be an odd value.
    sigma : float
        Sigma of normal distribution.
    k1 : float (default: 0.01)
        Algorithm parameter, K1 (small constant).
    k2 : float (default: 0.03)
        Algorithm parameter, K2 (small constant).
    scale_weights : list, optional
        Weights for different scales.
    data_range : int
        Maximum value of the image data (e.g., 255 for 8-bit images).
    channel_last_dim : bool (default: True)
        Indicates if the input images have the color channel as the last dimension.
        Needed for compatibility with torch operations.
    device : str
        Device to use for computation.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    This class is a wrapper of the implementation from:
    piq, PyTorch Image Quality: Metrics and Measure for Image Quality Assessment,
    https://github.com/photosynthesis-team/piq

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
        kernel_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        scale_weights: Optional[Sequence[float]] = None,
        data_range: int = 255,
        channel_last_dim: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if scale_weights is not None:
            scale_weights = torch.tensor(scale_weights)

        self._mssim = MultiScaleSSIMLoss(
            kernel_size=kernel_size,
            kernel_sigma=sigma,
            k1=k1,
            k2=k2,
            scale_weights=scale_weights,
            data_range=data_range,
            reduction="none",
        ).to(device)
        self.channel_last_dim = channel_last_dim
        self.device = device

    def compute(
        self,
        reference_imgs: Sequence[Union[np.ndarray, Tensor]],
        target_imgs: Sequence[Union[np.ndarray, Tensor]],
        **kwargs,
    ) -> MetricResult:
        """Computes MSSIM score.

        Parameters
        ----------
        reference_imgs : (N, H, W, C) ndarray
            Images to use as reference.
            List of arrays representing a RGB image of shape (H, W, C).
            For (C, H, W) images, set channel_last_dim to False.
        target_imgs : (N, H, W, C) ndarray
            Corresponding images to compare with reference.
            List of arrays representing a RGB image of shape (H, W, C).
            For (C, H, W) images, set channel_last_dim to False.

        Returns
        -------
        result : MetricResult
            Instance-level MSSIM scores.
        """
        reference_imgs, target_imgs = _validate_inputs(reference_imgs, target_imgs, self.channel_last_dim)

        _batch_calculation = len({ref.shape for ref in reference_imgs}) == 1
        if _batch_calculation:
            reference_imgs = torch.stack(reference_imgs).to(self.device)
            target_imgs = torch.stack(target_imgs).to(self.device)
            mssims = self._mssim(reference_imgs, target_imgs).detach().cpu()
            mssims = (1 - mssims).tolist()
        else:
            mssims = []
            for ref, targ in zip(reference_imgs, target_imgs):
                mssims.append(
                    1 - self._mssim(ref.unsqueeze(0).to(self.device), targ.unsqueeze(0).to(self.device)).item(),
                )

        return MetricResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": mssims},
        )


__all__ = ["SSIM", "MSSSIM"]
