from typing import Literal, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from piq.brisque import brisque
from piq.clip_iqa import CLIPIQA as _clip_iqa

from pymdma.common.definitions import Metric
from pymdma.common.output import DistributionResult, MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ....utils.processing import image_resize


class Tenengrad(Metric):
    """Computes Tenengrad score for an image. Sharpness measure based on the
    gradient magnitude.

    **Objective**: Sharpness

    Parameters
    ----------
    kernel_size : int, optional, default=3
        Size of the Sobel kernel.
    threshold : int, optional, default=0
        Threshold for valid gradient pixels. Used to supress noise, smooth the curve and impose sensitivity.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Groen et al., A Comparison of Different Focus Functions for Use in Autofocus Algorithms (1984)
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/cyto.990060202

    More information on Tenengrad:
    Her et al., Research of Image Sharpness Assessment Algorithm for Autofocus (2019)
    https://ieeexplore.ieee.org/abstract/document/8980980

    Examples
    --------
    >>> tenengrad = Tenengrad()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = tenengrad.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        kernel_size: int = 3,
        threshold: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.ksize = kernel_size

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes Tenengrad score for a list of images.

        Parameters
        ----------
        imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result: MetricResult
            Tenengrad score for each image.
        """
        scores = []
        for img in imgs:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.ksize)
            gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.ksize)

            x_y_gradients = np.sqrt(gradient_x**2 + gradient_y**2)

            x_y_gradients[x_y_gradients < self.threshold] = 0

            scores.append(np.mean(x_y_gradients))

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores},
        )


class TenengradRelative(Metric):
    """Computes Tenengrad score for an image in relation to a blurred instance
    of itself. Sharpness measure based on the gradient magnitude.

    **Objective**: Sharpness

    Parameters
    ----------
    kernel_size : int, optional, default=3
        Size of the Sobel kernel.
    threshold : int, optional, default=0
        Threshold for valid gradient pixels. Used to supress noise, smooth the curve and impose sensitivity.
    criteria : str, optional, default="ratio"
        Criteria for relative tenengrad score. Can be `ratio` or `diff`.
    blur_factor : float, optional, default=0.0
        Degree of blurring to apply to the image. The lower the value, the more blurred the image.
        Must be in the range [0., 1.].
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Groen et al., A Comparison of Different Focus Functions for Use in Autofocus Algorithms (1984)
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/cyto.990060202

    More information on Tenengrad:
    Her et al., Research of Image Sharpness Assessment Algorithm for Autofocus (2019)
    https://ieeexplore.ieee.org/abstract/document/8980980

    Examples
    --------
    >>> tenengrad = TenengradRelative()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = tenengrad.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        kernel_size: int = 3,
        threshold: int = 0,
        criteria: Literal["ratio", "diff"] = "ratio",
        blur_factor: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ksize = kernel_size
        self.trheshold = threshold
        assert criteria in ["ratio", "diff"], f"Unsupported criteria for relative tenengrad: {criteria}"
        self.criteria = criteria

        assert 0.0 <= blur_factor <= 1.0, "Blur factor must be in the range [0., 1.]"
        self.blur_factor = blur_factor

        self.tenengrad = Tenengrad(ksize=kernel_size, threshold=threshold)

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes TenengradRelative score for a list of images.

        Parameters
        ----------
        imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result: MetricResult
            Tenengrad score for each image.
        """
        relative_metrics = []
        for img in imgs:
            img_blur = np.asarray(ImageEnhance.Sharpness(Image.fromarray(img)).enhance(self.blur_factor))

            tenengrad_metric_img = self.tenengrad.compute([img]).instance_level.value[0]
            tenengrad_metric_img_blur = self.tenengrad.compute([img_blur]).instance_level.value[0]

            relative_metrics.append(
                (
                    tenengrad_metric_img_blur / tenengrad_metric_img
                    if self.criteria == "ratio"
                    else tenengrad_metric_img_blur - tenengrad_metric_img
                ),
            )

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": relative_metrics},
        )


class EME(Metric):
    """
    Computes the Measure of Enhancement (EME) score.
    Quantifies the enhancement of an image by measuring the contrast ratio of the image.
    Adapted from: https://www.researchgate.net/publication/244268659_A_New_Measure_of_Image_Enhancement

    **Objective**: Contrast

    Parameters
    ----------
    blocks_size : tuple of int, optional, default=(100, 100)
        Size of the blocks to divide the image into.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Agaian et al., A New Measure of Image Enhancement (2000).
    https://www.researchgate.net/publication/244268659_A_New_Measure_of_Image_Enhancement

    Examples
    --------
    >>> eme = EME()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = eme.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        blocks_size: Tuple[int, int] = (100, 100),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = blocks_size

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes the Measure of Enhancement (EME) score for a list of
        images.

        Parameters
        ----------
        imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result: MetricResult
            EME score for each image.
        """
        emes = []
        for img in imgs:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_width = img.shape[0]
            img_height = img.shape[1]
            n_steps_width = int(img_width / self.block_size[0])
            n_steps_height = int(img_height / self.block_size[1])
            total_contrast_ratio = 0
            total_blocks = 0
            for w_step in range(0, n_steps_width):
                w_cords = (w_step * self.block_size[0], (w_step + 1) * self.block_size[0])
                for h_step in range(1, n_steps_height):
                    h_cords = (h_step * self.block_size[1], (h_step + 1) * self.block_size[1])
                    cur_block = img[w_cords[0] : w_cords[1], h_cords[0] : h_cords[1]]
                    if np.min(cur_block) < 0.1:
                        min_val = 0.1
                    else:
                        min_val = np.min(cur_block)
                    if np.max(cur_block) < 0.1:
                        max_val = 0.1
                    else:
                        max_val = np.max(cur_block)
                    contrast_ratio_block = 20 * np.log(max_val / min_val)
                    total_contrast_ratio += contrast_ratio_block
                    total_blocks += 1

            emes.append(total_contrast_ratio / total_blocks)

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": emes},
        )


# TODO documentation
class ExposureBrightness(Metric):
    """Computes Exposure and Brightness level Metric. Values higher than 1
    indicate overexposure, while values closer to 0 indicate underexposure.

    **Objective**: Exposure and Brightness

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    Examples
    --------
    >>> exposure_brightness = ExposureBrightness()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = exposure_brightness.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes exposure level for a list of images.

        Parameters
        ----------
        imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result: MetricResult
            Exposure score for each image.
        """
        exposure_levels = []
        for img in imgs:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            mean, std = cv2.meanStdDev(img)
            exposure_levels.append((mean[0][0] + 2 * std[0][0]) / 255)

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": exposure_levels},
        )


class Brightness(Metric):
    """Computes brightness level of an image.

    **Objective**: Brightness

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Darel Rex Finley, HSP Color Model — Alternative to HSV (HSB) and HSL (2006).
    http://alienryderflex.com/hsp.html

    Marian Stefanescu, Measuring and enhancing image quality attributes (2021).
    https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10

    Examples
    --------
    >>> brightness = Brightness()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = brightness.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes brightness level for a list of images.

        Parameters
        ----------
        imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result: MetricResult
            Brightness score for each image.
        """
        scores = []
        for img in imgs:
            assert img.shape[-1] == 3, "Image should be in RGB format"
            nr_of_pixels = len(img) * len(img[0])

            img = img.astype(np.float32)
            r_vals = 0.299 * img[:, :, 0] ** 2
            g_vals = 0.587 * img[:, :, 1] ** 2
            b_vals = 0.114 * img[:, :, 2] ** 2
            total_brightness = np.sqrt(r_vals + g_vals + b_vals).sum()

            scores.append(total_brightness / nr_of_pixels)

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores},
        )


class Colorfulness(Metric):
    """Computes colorfulness level of an image.

    **Objective**: Colorfulness

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Hasler et al., Measuring colourfulness in natural images (2003).
    https://infoscience.epfl.ch/server/api/core/bitstreams/77f5adab-e825-4995-92db-c9ff4cd8bf5a/content

    Code was adapted from:
    Adrian Rosebrock, Computing image “colorfulness” with OpenCV and Python (2017).
    https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/

    Examples
    --------
    >>> colorfulness = Colorfulness()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = colorfulness.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes colorfulness level of list of images.

        Parameters
        ----------
        imgs : (N, H, W, C) ndarray
            List of arrays representing RGB image of shape (H, W, C).

        Returns
        -------
        result: MetricResult
            Colorfulness score for each image.
        """
        scores = []
        for img in imgs:
            assert len(img.shape) == 3, "Image should be in RGB format"

            img = img.astype(np.float32)
            (r_channel, g_channel, b_channel) = img[:, :, 0], img[:, :, 1], img[:, :, 2]

            rg = np.absolute(r_channel - g_channel)
            yb = np.absolute(0.5 * (r_channel + g_channel) - b_channel)

            (rb_mean, rb_std) = (np.mean(rg), np.std(rg))
            (yb_mean, yb_std) = (np.mean(yb), np.std(yb))

            std_root = np.sqrt((rb_std**2) + (yb_std**2))
            mean_root = np.sqrt((rb_mean**2) + (yb_mean**2))

            scores.append(std_root + (0.3 * mean_root))
        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores},
        )


class CLIPIQA(Metric):
    """Compute the CLIP-based IQA score. Wrapper of the PIQ CLIP-IQA metric.
    Evaluates perceptual quality of an image using a CLIP model.

    **Objective**: General Image Quality

    Parameters
    ----------
    img_size : {tuple of int, int}, optional, default=(512, 512)
        Size for each image. If a single integer is provided, the image will be thumbnail resized.
        In thumbnail resizing, the aspect ratio is maintained and batch calculation is not allowed (slower computation).
    interpolation : int, optional, default=cv2.INTER_LINEAR
        Interpolation method for resizing.
    data_range : float, optional, default=255
        The range of the data. By default, it is assumed to be [0, 255].
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Wang et al., Exploring CLIP for Assessing the Look and Feel of Images (2022).
    https://arxiv.org/abs/2207.12396

    This is a wrapper class for the implementation in:
    piq, PyTorch Image Quality: Metrics for Image Quality Assessment.
    https://github.com/photosynthesis-team/piq

    Examples
    --------
    >>> clip_iqa = CLIPIQA()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = clip_iqa.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        img_size: Union[Tuple[int, int], int] = (512, 512),
        interpolation: int = cv2.INTER_LINEAR,
        data_range: float = 255,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.interpolation = interpolation
        self._clip = _clip_iqa(data_range=data_range).to(device)
        self.device = device

        if isinstance(img_size, tuple):
            self._height, self._width = img_size, img_size
            self._batch_calculation = True
        else:
            # thumbnail resize (different image sizes)
            self._height, self._width = img_size, None
            self._batch_calculation = False

        self._height, self._width = img_size if isinstance(img_size, tuple) else (img_size, None)

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to the required size and convert to tensor."""
        img = image_resize(img, height=self._height, width=self._width, inter=self.interpolation)
        return torch.tensor(img).permute(2, 0, 1)

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes CLIPIQA level of a list of images.

        Parameters
        ----------
        imgs : (N, H, W, C) ndarray
            List of arrays representing RGB image of shape (H, W, C).

        Returns
        -------
        result: MetricResult
            CLIPIQA score for each image.
        """
        imgs = [self._process_image(img) for img in imgs]

        if self._batch_calculation:
            imgs = torch.stack(imgs).to(self.device)
            scores = self._clip(imgs).detach().cpu().squeeze(1).tolist()
        else:
            scores = [self._clip(img.unsqueeze(0)).detach().cpu().item() for img in imgs]

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores},
        )


class BRISQUE(Metric):
    """Computes  Blind/referenceless Image Spatial Quality Evaluator (BRISQUE)
    score. Wrapper of the PIQ BRISQUE metric implementation.

    **Objective**: General Image Quality

    Parameters
    ----------
    kernel_size : int, optional, default=7
        The size of the Gaussian kernel.
    kernel_sigma : float, optional, default=7 / 6
        The standard deviation of the Gaussian kernel.
    data_range : float, optional, default=255
        The range of the data. By default, it is assumed to be [0, 255].
    device : str, optional, default="cpu"
        Device to run the computation.
    same_size : bool, optional, default=False
        If True, all provided images must have the same size (faster computation).
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Mittal et al., No-Reference Image Quality Assessment in the Spatial Domain (2012).
    https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

    This is a wrapper class for the implementation in:
    piq, PyTorch Image Quality: Metrics for Image Quality Assessment.
    https://github.com/photosynthesis-team/piq

    Examples
    --------
    >>> brisque = BRISQUE()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = brisque.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        kernel_size: int = 7,
        kernel_sigma: float = 7 / 6,
        data_range: float = 255,
        device: str = "cpu",
        same_size: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.data_range = data_range
        self._brisque = brisque
        self._batch_calculation = same_size
        self.device = device

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to the required size and convert to tensor."""
        return torch.tensor(img).permute(2, 0, 1)

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes BRISQUE.

        Parameters
        ----------
        imgs : (N, H, W, C) ndarray
            List of arrays representing RGB image of shape (H, W, C).

        Returns
        -------
        result: MetricResult
            BRISQUE score for each image.
        """
        imgs = [self._process_image(img) for img in imgs]

        if self._batch_calculation:
            imgs = torch.stack(imgs).to(self.device)
            scores = (
                self._brisque(
                    imgs,
                    kernel_size=self.kernel_size,
                    kernel_sigma=self.kernel_sigma,
                    data_range=self.data_range,
                    reduction="none",
                )
                .detach()
                .cpu()
                .tolist()
            )
        else:
            scores = [
                self._brisque(
                    img.unsqueeze(0),
                    kernel_size=self.kernel_size,
                    kernel_sigma=self.kernel_sigma,
                    data_range=self.data_range,
                    reduction="none",
                )
                .detach()
                .cpu()
                .item()
                for img in imgs
            ]

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores},
        )


__all__ = [
    "Tenengrad",
    "TenengradRelative",
    "EME",
    "ExposureBrightness",
    "Brightness",
    "Colorfulness",
    "CLIPIQA",
    "BRISQUE",
]
