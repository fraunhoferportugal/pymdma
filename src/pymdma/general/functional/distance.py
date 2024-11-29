import numpy as np
import torch
import ot
from sklearn.metrics import pairwise_kernels

from ..models.kernels import GaussianKernel, MultipleKernelMaximumMeanDiscrepancy


def wasserstein(x_feat, y_feat):
    """Calculates the Wasserstein distance between two sets of sample features.

    Parameters
    ----------
    x_feat : array-like of shape (nx_samples, n_features)
        2D array with features of the original samples.
    y_feat : array-like of shape (ny_samples, n_features)
        2D array with features of the fake samples.

    Returns
    -------
    wasserstein_distance : float
        Wasserstein distance between the distributions of x_feat and y_feat.

    Notes
    -----
    The function uses the POT library (https://pythonot.github.io/) to compute the pairwise distance matrix and the Wasserstein distance.
    """
    x_feat = np.array(x_feat)
    y_feat = np.array(y_feat)

    # Compute pairwise distance matrix
    dist_matrix_ot = ot.dist(x_feat, y_feat)

    # Compute Wasserstein distance using POT
    wasserstein_distance = ot.emd2([], [], dist_matrix_ot)

    return wasserstein_distance


def fast_mmd_linear(x_feat, y_feat, **kwargs):
    """Calculate the fast version of Maximum Mean Discrepancy (MMD) using a
    linear kernel(i.e., k(x,y) = <x,y>).

    Parameters
    ----------
    x_feat : array-like of shape [n_samples_x, n_features]
        2D array with features of the original samples.
    y_feat : array-like of shape [n_samples_y, n_features]
        2D array with features of the fake samples.

    Returns
    -------
    float
        The MMD between the two sets of samples.

    References
    ----------
    Gretton, A. et al. "A Kernel Method for the Two-Sample Problem" (2006)
    https://arxiv.org/pdf/0805.2368

    Code adapted from:
    https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    """

    # Calculate the mean difference between X and Y
    delta = x_feat.mean(0) - y_feat.mean(0)

    # Calculate the MMD using the linear kernel formula
    fast_mmd = delta.dot(delta.T)

    return fast_mmd


def mmd_kernel(x_feat, y_feat, kernel="sigmoid"):
    """Calculate the Maximum Mean Discrepancy (MMD) using a specified kernel
    function.

    Parameters
    ----------
    x_feat : array-like of shape (n_samples_x, n_features)
        2D array containing features from samples of the original distribution.
    y_feat : array-like of shape (n_samples_y, n_features)
        2D array containing features from samples of the fake distribution.
    kernel : str, optional, default: 'sigmoid'
        The kernel function to use for calculating MMD. Options include:
        'additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'

    Returns
    -------
    float
        The MMD between the two sets of samples calculated using the specified kernel.

    References
    ----------
    Gretton, A. et al.  "A Kernel Method for the Two-Sample Problem" (2006)
    https://arxiv.org/pdf/0805.2368
    """

    n_x = x_feat.shape[0]
    n_y = y_feat.shape[0]

    # Calculate the MMD components
    k_xx = pairwise_kernels(x_feat, x_feat, metric=kernel)
    k_yy = pairwise_kernels(y_feat, y_feat, metric=kernel)
    k_xy = pairwise_kernels(x_feat, y_feat, metric=kernel)

    intra_x = np.sum(k_xx) / (n_x * n_x)
    intra_y = np.sum(k_yy) / (n_y * n_y)
    extra = np.sum(k_xy) / (n_x * n_y)

    # Calculate the MMD by subtracting the inter-distribution similarity from the sum of intra-distribution similarities
    mmd = intra_x + intra_y - 2 * extra

    return mmd


def mk_mmd(x_feat, y_feat, **kwargs):
    """Compute Multiple Kernel Maximum Mean Discrepancy (MK-MMD) between two
    sets of samples using a set of Gaussian kernels.

    Parameters
    ----------
    x_feat : array-like of shape [n_samples_x, n_features]
        2D array containing features from samples of the original distribution.
    y_feat : array-like of shape [n_samples_y, n_features]
        2D array containing features from samples of the fake distribution.
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    Returns
    -------
    torch.Tensor
        MK-MMD value.

    References
    ----------
    Gretton, A. et al, Optimal kernel choice for large-scale two-sample tests. (NIPS'12)
    https://proceedings.neurips.cc/paper_files/paper/2012/file/dbe272bab69f8e13f14b405e038deb64-Paper.pdf

    Code adapted from:
    Transfer Learning Library by the THUML Group.
    https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/dan.py
    """
    mk_mmd = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2**k) for k in range(-3, 2)],
        linear=False,
    )

    return mk_mmd(torch.from_numpy(x_feat), torch.from_numpy(y_feat)).item()


def cos_sim_2d(x_feat, y_feat):
    """Calculate the cosine similarity between two sets of feature vectors.

    Parameters
    ----------
    x_feat : array-like of shape [n_samples_x, n_features]
        2D array with features of the original samples.
    y_feat : array-like of shape [n_samples_y, n_features]
        2D array with features of the fake samples.

    Returns
    -------
    float
        The cosine similarity between the two sets of feature vectors.

    References
    ----------
    Manning, C. D., Raghavan, P., & Schütze, H., Introduction to Information Retrieval (2008).
    https://www.cambridge.org/highereducation/books/introduction-to-information-retrieval/669D108D20F556C5C30957D63B5AB65C#overview
    """
    # Normalize the vectors in both sets to have unit length
    norm_x = x_feat / np.linalg.norm(x_feat, axis=1, keepdims=True)
    norm_y = y_feat / np.linalg.norm(y_feat, axis=1, keepdims=True)

    # Compute the cosine similarity by taking the dot product of normalized vectors
    cosine_similarity_matrix = np.matmul(norm_x, norm_y.T)

    # Calculate the mean cosine similarity across the same pais of vectors in each feat_vec
    similarity = np.mean(np.diagonal(cosine_similarity_matrix))

    return similarity
