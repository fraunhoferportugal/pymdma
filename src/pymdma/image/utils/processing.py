from typing import List, Tuple

import cv2
import numpy as np


def downsample_largest(im_1, im_2):
    size_1 = im_1.shape[0] * im_1.shape[1]
    size_2 = im_2.shape[0] * im_2.shape[1]

    if size_1 > size_2:
        return cv2.resize(im_1, (im_2.shape[1], im_2.shape[0]), interpolation=cv2.INTER_LANCZOS4), im_2
    return im_1, cv2.resize(im_2, (im_1.shape[1], im_1.shape[0]), interpolation=cv2.INTER_LANCZOS4)


def batch_downsample_to_largest(imgs: List[np.ndarray], imgs2: List[np.ndarray]) -> Tuple[List[np.ndarray]]:
    """Downsamples the largest image in the image pair.

    Args:
        imgs (List[np.ndarray]): left images
        imgs2 (List[np.ndarray]): right images

    Returns:
        Tuple[np.ndarray]: tuple with the image lists
    """
    resized = [downsample_largest(img, img2) for img, img2 in zip(imgs, imgs2)]
    return [img for img, _ in resized], [img2 for _, img2 in resized]


def image_resize(
    image: np.ndarray,
    height: int = None,
    width: int = None,
    inter: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """Resize an image to a given width or height.

    Aspect ratio is kept if only one dimmension is given.
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    if width is not None and height is not None:
        return cv2.resize(image, (width, height), interpolation=inter)

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
