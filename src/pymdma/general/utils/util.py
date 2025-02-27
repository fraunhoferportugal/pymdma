from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor, tensor


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def set_seed(seed: int = 42):
    """Set random seed for numpy.

    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    """
    rng = np.random.default_rng(seed)
    return rng


def check_input_shape(dataset, ndim=2):
    if not isinstance(dataset, np.ndarray) or dataset.ndim != ndim:
        raise ValueError(
            "Input variable is not a NumPy array with the required dimensions",
        )


# ======= General Functions =======
def min_max_scaling(array: np.ndarray) -> np.ndarray:
    return (array - array.min()) / (array.max() - array.min())


def features_splitting(feat, seed: Optional[int] = None):
    """Split the input array into two subsets with equal lengths. If the number
    of elements in the input array is odd, the function discards one element to
    ensure both subsets have the same length.

    Parameters
    ----------
    feat : np.ndarray (n_samples, n_features)
        Input array to be split. The first dimension should be the number of elements.

    seed : int, optional
        Seed for the random number generator, ensuring reproducibility of the split.
        If None (default), the random number generator uses the current random state.

    Returns
    -------
    tuple
        A tuple of two np.ndarray objects, each representing a subset of the input array.
    """
    rng = set_seed(seed)
    # Randomly sample two subsets
    subset_indices1 = rng.choice(feat.shape[0], size=len(feat) // 2, replace=False)
    remaining_indices = np.setdiff1d(np.arange(len(feat)), subset_indices1)
    subset_indices2 = rng.choice(remaining_indices, size=len(feat) // 2, replace=False)

    # Use the sampled indices to extract subsets from the original array
    return feat[subset_indices1], feat[subset_indices2]


def cluster_into_bins(eval_data, ref_data, num_clusters):
    """Clusters the union of the data points and returns the cluster
    distribution.

    Clusters the union of eval_data and ref_data into num_clusters using minibatch
    k-means. Then, for each cluster, it computes the number of points from

    Parameters
    ----------
    eval_data : np.ndarray
        Data points from the distribution to be evaluated.
    ref_data : np.ndarray
        Data points from the reference distribution.
    num_clusters : int
        Number of cluster centers to fit.

    Returns
    -------
    eval_bins : np.ndarray
        Array where the i-th entry represents the number of points from eval_data assigned to the i-th cluster.
    ref_bins : np.ndarray
        Array where the i-th entry represents the number of points from ref_data assigned to the i-th cluster.

    References
    ---------
    Sajjadi, Mehdi SM, et al. Assessing generative models via precision and recall (2018).
    https://proceedings.neurips.cc/paper_files/paper/2018/file/f7696a9b362ac5a51c3dc8f098b73923-Paper.pdf

    Code taken from:
    https://github.com/vanderschaarlab/evaluating-generative-models/blob/main/metrics/prd_score.py
    """

    cluster_data = np.vstack([eval_data, ref_data])
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[: len(eval_data)]
    ref_labels = labels[len(eval_data) :]

    eval_bins = np.histogram(
        eval_labels,
        bins=num_clusters,
        range=[0, num_clusters],
        density=True,
    )[0]
    ref_bins = np.histogram(
        ref_labels,
        bins=num_clusters,
        range=[0, num_clusters],
        density=True,
    )[0]
    return eval_bins, ref_bins


def to_tensor(data: Union[Tensor, np.ndarray]) -> Tensor:
    return data if isinstance(data, Tensor) else tensor(data)


# EOF
