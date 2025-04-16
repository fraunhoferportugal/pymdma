from typing import Generator, Optional

import numpy as np
from sklearn import get_config
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.utils import gen_batches


def get_chunk_n_rows(row_bytes, *, max_n_rows=None, working_memory=None):
    """Calculate how many rows can be processed within `working_memory`.

    Parameters
    ----------
    row_bytes : int
        The expected number of bytes of memory that will be consumed
        during the processing of each row.
    max_n_rows : int, default=None
        The maximum return value.
    working_memory : int or float, default=None
        The number of rows to fit inside this number of MiB will be
        returned. When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    int
        The number of rows which can be processed within `working_memory`.

    Warns
    -----
    Issues a UserWarning if `row_bytes exceeds `working_memory` MiB.
    """

    if working_memory is None:
        working_memory = get_config()["working_memory"]

    chunk_n_rows = int(working_memory * (2**20) // row_bytes)
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    if chunk_n_rows < 1:
        chunk_n_rows = 1
    return chunk_n_rows


# ======= DISTANCE FUNCTIONS =======
# from: https://github.com/clovaai/generative-evaluation-prdc
def compute_pairwise_distance(
    data_x: np.ndarray,
    data_y: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    n_workers: int = 4,
) -> np.ndarray:
    """Computes the pairwise distance between the rows of data_x and data_y.

    Parameters
    ----------
        data_x : np.ndarray
            The first set of data points.
        data_y : np.ndarray, optional
            The second set of data points. If None, data_x is used.
        metric : str, optional
            The distance metric to use. Defaults to "euclidean".

    Returns
    -------
        dists : np.ndarray
            The pairwise distances between the rows of data_x and data_y.
    """
    if data_y is None:
        data_y = data_x

    dists = np.zeros((len(data_x), len(data_y)), dtype=np.float32)
    gen = pairwise_distances_chunked(
        data_x,
        data_y,
        metric=metric,
        n_jobs=n_workers,
    )

    pos = 0
    for chunk in gen:
        dists[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
    return dists


def get_kth_value_chunked(
    unsorted: np.ndarray,
    kth: int,
    axis: Optional[int] = -1,
) -> Generator[np.ndarray, None, None]:
    """Yields the kth smallest values in unsorted along the specified axis.

    Parameters
    ----------
        unsorted : np.ndarray
            The input array.
        kth : int
            The kth smallest value to retrieve.
        axis : int, optional
            The axis over which to compute the kth smallest value. Defaults to -1.

    Yields
    ------
        kth_values : np.ndarray
            The kth smallest values in the input array.
    """
    chunk_n_rows = get_chunk_n_rows(
        row_bytes=8 * len(unsorted),
        max_n_rows=len(unsorted),
    )
    slices = gen_batches(len(unsorted), chunk_n_rows)

    # TODO parallel processing?
    for slice_ in slices:
        chunk = unsorted[slice_]
        indices = np.argpartition(chunk, kth, axis=axis)[..., :kth]
        k_smallests = np.take_along_axis(chunk, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        yield kth_values


def get_kth_value(unsorted: np.ndarray, kth: int, axis: Optional[int] = -1) -> np.ndarray:
    """Computes the kth smallest values in unsorted along the specified axis.

    Parameters
    ----------
        unsorted : np.ndarray
            The input array.
        kth : int
            The kth smallest value to retrieve.
        axis : int, optional
            The axis over which to compute the kth smallest value. Defaults to -1.

    Returns
    -------
        kth_values : np.ndarray
            The kth smallest values in the input array.
    """

    kth_values = np.zeros(len(unsorted), dtype=np.float32)
    pos = 0
    for chunk in get_kth_value_chunked(unsorted, kth, axis):
        kth_values[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
    return kth_values


def compute_nearest_neighbour_distances(
    input_features: np.ndarray,
    nearest_k: int,
    metric: str = "euclidean",
    n_workers: int = 4,
) -> np.ndarray:
    """Compute the distances to the kth nearest neighbor for each point in the
    input features.

    Parameters
    ----------
    input_features : np.ndarray
        A 2D array of shape (N, feature_dim) containing the input feature vectors.
    nearest_k : int
        The number of nearest neighbors to consider.
    metric : str, optional
        The distance metric to use for computing pairwise distances. Defaults to "euclidean".

    Returns
    -------
    np.ndarray
        A 1D array of shape (N,) containing the distances to the kth nearest neighbor for each input feature.
    """
    distances = compute_pairwise_distance(input_features, metric=metric, n_workers=n_workers)
    radii = get_kth_value(distances, kth=nearest_k + 1, axis=-1)
    return radii
