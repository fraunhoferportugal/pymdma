"""Module for DOM sharpness.

Adapted from:
pydom, Sharpness Estimation for Document and Scene Images.
    https://github.com/umang-singhal/pydom


Original paper:
Kumar et al., Sharpness estimation for document and scene images (2012).
    https://ieeexplore.ieee.org/document/6460868
"""

import cv2
import numpy as np
from loguru import logger

from pymdma.common.definitions import Metric
from pymdma.common.output import DistributionResult, MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


def _dom(median_blurred: np.ndarray):
    """Find DOM at each pixel.

    Parameters
    ----------
    median_blurred : np.ndarray
        Median filtered image.

    Returns
    -------
    domx : np.ndarray
        Diff of diff on x axis.
    domy : np.ndarray
        Diff of diff on y axis.
    """
    median_shift_up = np.pad(median_blurred, ((0, 2), (0, 0)), "constant")[2:, :]
    median_shift_down = np.pad(median_blurred, ((2, 0), (0, 0)), "constant")[:-2, :]
    domx = np.abs(median_shift_up - 2 * median_blurred + median_shift_down)

    median_shift_left = np.pad(median_blurred, ((0, 0), (0, 2)), "constant")[:, 2:]
    median_shift_right = np.pad(median_blurred, ((0, 0), (2, 0)), "constant")[:, :-2]
    domy = np.abs(median_shift_left - 2 * median_blurred + median_shift_right)
    return domx, domy


def _sharpness_matrix(
    median_blurred: np.ndarray,
    edgex: np.ndarray,
    edgey: np.ndarray,
    width: int = 2,
):
    """Find sharpness value at each pixel.

    Parameters
    ----------
    median_blurred : np.ndarray
        Median filtered grayscale image.
    edgex : np.ndarray
        Edge pixels in x-axis.
    edgey : np.ndarray
        Edge pixels in y-axis.
    width : int, optional
        Edge width, by default 2.
    debug : bool, optional
        To show intermediate results, by default False.

    Returns
    -------
    Sx : np.ndarray
        Sharpness value matrix computed in x-axis.
    Sy : np.ndarray
        Sharpness value matrix computed in y-axis.
    """
    # Compute dod measure on both axis
    domx, domy = _dom(median_blurred)

    # Contrast on x and y axis
    Cx = np.abs(median_blurred - np.pad(median_blurred, ((1, 0), (0, 0)), "constant")[:-1, :])
    Cy = np.abs(median_blurred - np.pad(median_blurred, ((0, 0), (1, 0)), "constant")[:, :-1])

    # Filter out sharpness at pixels other than edges
    Cx = np.multiply(Cx, edgex)
    Cy = np.multiply(Cy, edgey)

    # initialize sharpness matriz with 0's
    Sx = np.zeros(domx.shape)
    Sy = np.zeros(domy.shape)

    # Compute Sx
    for i in range(width, domx.shape[0] - width):
        num = np.abs(domx[i - width : i + width, :]).sum(axis=0)
        dn = Cx[i - width : i + width, :].sum(axis=0)
        Sx[i] = [(num[k] / dn[k] if dn[k] > 1e-3 else 0) for k in range(Sx.shape[1])]

    # Compute Sy
    for j in range(width, domy.shape[1] - width):
        num = np.abs(domy[:, j - width : j + width]).sum(axis=1)
        dn = Cy[:, j - width : j + width].sum(axis=1)
        Sy[:, j] = [(num[k] / dn[k] if dn[k] > 1e-3 else 0) for k in range(Sy.shape[0])]
    return Sx, Sy


def _sharpness_measure(
    median_blurred: np.ndarray,
    edgex: np.ndarray,
    edgey: np.ndarray,
    width: int,
    sharpness_threshold: float,
    epsilon: float = 1e-8,
):
    """Final Sharpness Value.

    Parameters
    ----------
    median_blurred : np.ndarray
        Median filtered grayscale image.
    width : int
        Edge width.
    sharpness_threshold : float
        Thresold to consider if a pixel is sharp.
    epsilon : float, optional
        Small value to defer div by zero, by default 1e-8.

    Returns
    -------
    S : float
        Sharpness measure(0<S<sqrt(2)).
    """
    Sx, Sy = _sharpness_matrix(median_blurred, edgex, edgey, width=width)
    Sx = np.multiply(Sx, edgex)
    Sy = np.multiply(Sy, edgey)

    Rx = np.sum(Sx >= sharpness_threshold) / (np.sum(edgex) + epsilon)
    Ry = np.sum(Sy >= sharpness_threshold) / (np.sum(edgey) + epsilon)
    return np.sqrt(Rx**2 + Ry**2)


def _smoothen_image(image: np.ndarray, transpose: bool = False, epsilon: float = 1e-8):
    """Smmoth image with ([0.5, 0, -0.5]) 1D filter.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    transpose : bool, optional
        To apply filter on vertical axis, by default False.
    epsilon : float, optional
        Small value to defer div by zero, by default 1e-8.

    Returns
    -------
    image_smoothed : np.ndarray
        Smoothened image.
    """
    fil = np.array([0.5, 0, -0.5])  # Smoothing Filter

    # change image axis for column convolution
    if transpose:
        image = image.T

    # Convolve grayscale image with smoothing filter
    image_smoothed = np.array([np.convolve(image[i], fil, mode="same") for i in range(image.shape[0])])

    # change image axis after column convolution
    if transpose:
        image_smoothed = image_smoothed.T

    # Normalize smoothened grayscale image
    return np.abs(image_smoothed) / (np.max(image_smoothed) + epsilon)


def _get_sharpness(
    img,
    width=2,
    sharpness_threshold=2,
    edge_threshold=0.0001,
    blur: bool = False,
    blur_size: tuple = (5, 5),
):
    """Image Sharpness Assessment.

    Parameters
    ----------
    img : str or np.ndarray
        Image source or image matrix.
    width : int, optional
        Text edge width, by default 2.
    sharpness_threshold : float, optional
        Thresold to consider if a pixel is sharp, by default 2.
    edge_threshold : float, optional
        Thresold to consider if a pixel is an edge pixel, by default 0.0001.
    debug : bool, optional
        To show intermediate results, by default False.

    Returns
    -------
    sharpness : float
        Image sharpness measure(0<S<sqrt(2)).
    """
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        image = img
    else:
        raise ValueError("Image is not in correct shape")

    # Add Gaussian Blur
    if blur:
        image = cv2.GaussianBlur(image, blur_size)

    # Perform median blur for removing Noise
    median_blurred = cv2.medianBlur(image, 3, cv2.CV_64F).astype("double") / 255.0

    # Smoothen image in x and y axis
    smoothx = _smoothen_image(image, transpose=True)
    smoothy = _smoothen_image(image)
    edgex = smoothx > edge_threshold
    edgey = smoothy > edge_threshold

    return _sharpness_measure(median_blurred, edgex, edgey, width=width, sharpness_threshold=sharpness_threshold)


class DOM(Metric):
    """Computes DOM sharpness score for an image. It is effective in detecting
    motion-blur, de-focused images or inherent properties of imaging system.

    **Objective**: Sharpness

    Parameters
    ----------
    width : int, optional, default=2
        Width of the edge filter.
    sharpness_threshold : int, optional, default=2
        Threshold for considering if a pixel is sharp or not.
    edge_threshold : float, optional, default=0.0001
        Threshold for edge.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Kumar et al., Sharpness estimation for document and scene images (2012).
    https://ieeexplore.ieee.org/document/6460868

    Code was adapted from:
    pydom, Sharpness Estimation for Document and Scene Images.
    https://github.com/umang-singhal/pydom

    Examples
    --------
    >>> dom = DOM()
    >>> imgs = np.random.rand(20, 100, 100, 3) # (N, H, W, C)
    >>> result: MetricResult = dom.compute(imgs)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        width: int = 2,
        sharpness_threshold: int = 2,
        edge_threshold: float = 0.0001,
        blur: bool = False,
        blur_size: tuple = (5, 5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = width
        self.sharpness_threshold = sharpness_threshold
        self.edge_threshold = edge_threshold
        self.blur = blur
        self.blur_size = blur_size
        if blur:
            logger.warning("Applying Gaussian Blur to the images may lead to non-deterministic results.")

    def compute(
        self,
        imgs: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes DOM score for an image.

        Parameters
        ----------
        imgs : {(N, H, W, C) ndarray, (N, H, W) ndarray}
            List of arrays representing RGB or grayscale image of shape (H, W, C) or (H, W), respectively.

        Returns
        -------
        result: MetricResult
            DOM score for each image.
        """
        scores = [
            _get_sharpness(
                img,
                self.width,
                self.sharpness_threshold,
                self.edge_threshold,
                self.blur,
                self.blur_size,
            )
            for img in imgs
        ]

        return DistributionResult(
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "float", "value": scores},
        )


__all__ = ["DOM"]
