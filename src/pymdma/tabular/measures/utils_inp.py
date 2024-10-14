from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stat
from loguru import logger
from pycanon import anonymity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.outliers_influence import variance_inflation_factor

# # DESCRIPTIVE STATISTICS
# def n_sample(data: np.ndarray):
#     """Computes the number of samples in the dataset.

#     Args:
#         data (np.ndarray): Input dataset.

#     Returns:
#         int: Number of samples.
#     """
#     return len(data)


# def mean_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the mean along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the mean is computed. Defaults to 0.

#     Returns:
#         np.ndarray: Mean values.
#     """
#     return np.mean(data, axis=axis)


# def stdev_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the standard deviation along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the standard deviation is computed. Defaults to 0.

#     Returns:
#         np.ndarray: Standard deviation values.
#     """
#     return np.std(data, axis=axis)


def percentile_(data: np.ndarray, percentile: int = 75, **kwargs):
    """Computes the specified percentile of the data.

    Args:
        data (np.ndarray): Input data.
        percentile (int, optional): Percentile to compute. Defaults to 75.

    Returns:
        np.ndarray: Percentile values.
    """
    return np.percentile(data, q=percentile)


# def max_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the maximum value along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the maximum value is computed. Defaults to 0.

#     Returns:
#         np.ndarray: Maximum values.
#     """
#     return np.max(data, axis=axis)


# def min_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the minimum value along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the minimum value is computed. Defaults to 0.

#     Returns:
#         np.ndarray: Minimum values.
#     """
#     return np.min(data, axis=axis)


# def skewness_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the skewness along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the skewness is computed. Defaults to 0.

#     Returns:
#         float: Skewness values.
#     """
#     return stat.skewness(data, axis=axis)


# def kurtosis_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the kurtosis along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the kurtosis is computed. Defaults to 0.

#     Returns:
#         float: Kurtosis values.
#     """
#     return stat.kurtosis(data, axis=axis)


# def cv_(data: np.ndarray, axis: int = 0, **kwargs):
#     """Computes the coefficient of variation along the specified axis.

#     Args:
#         data (np.ndarray): Input data.
#         axis (int, optional): Axis along which the coefficient of variation is computed. Defaults to 0.

#     Returns:
#         np.ndarray: Coefficient of variation values.
#     """
#     return np.divide(stdev_(data, axis), mean_(data, axis))


def entropy_(data: np.ndarray, axis: int = 0, **kwargs):
    """Computes the entropy along the specified axis.

    Args:
        data (np.ndarray): Input data.
        axis (int, optional): Axis along which the entropy is computed. Defaults to 0.

    Returns:
        float: Entropy values.
    """
    # number of unique values
    unq = len(data)

    # max possible entropy
    max_val = np.log2(unq)

    # compute normalised entropy
    if unq == 1:
        entpy = 1.0
    else:
        entpy = stat.entropy(data, axis=axis, base=2) / max_val

    return entpy


def chi2_uniformity(data, **kwargs):
    if len(data) > 1:
        # expected distribution
        expected = np.full_like(data, fill_value=np.mean(data))

        # perform test
        chi2, p_value = stat.chisquare(data, f_exp=expected)

        # norm
        chi2_norm = chi2 / (len(data) - 1)
    else:
        # norm
        chi2_norm = 0.0

    return chi2_norm


def ks_uniformity(data, **kwargs):
    # observed data
    obs_data = data.copy()

    # expected data
    exp_data = np.random.uniform(
        low=min(obs_data),
        high=max(obs_data),
        size=len(obs_data),
    )

    # perform test
    ks_stat, p_value = stat.kstest(obs_data, exp_data)

    return ks_stat


def stat_test_uniformity(data: np.ndarray, var: np.ndarray, tag: str, **kwargs):
    # score
    if tag == "discrete":
        score = 1 - chi2_uniformity(var, **kwargs)
    elif tag == "continuous":
        score = 1 - ks_uniformity(data, **kwargs)
    else:
        score = -1

    return score


def iqr_(data: np.ndarray, **kwargs):
    """Computes the interquartile range (IQR) of the data.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Interquartile range (IQR) values.
    """
    return percentile_(data, 75) - percentile_(data, 25)


# STATISTICAL OUTLIERS
def z_score_outliers(data: np.ndarray, threshold: int = 3, **kwargs):
    """Counts the number of outliers found through the z-score metric.

    Args:
        data (np.ndarray): Input data.
        threshold (int, optional): Z-score threshold for identifying outliers. Defaults to 3.

    Returns:
        int: Number of outliers.
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    return np.sum(z_scores > threshold)


def iqr_outliers(data: np.ndarray, factor: float = 1.5, **kwargs):
    """Counts the number of outliers found through the interquartile range
    (IQR) method.

    Args:
        data (np.ndarray): Input data.
        factor (float, optional): IQR factor for identifying outliers. Defaults to 1.5.

    Returns:
        int: Number of outliers.
    """
    # Quartiles
    Q1 = percentile_(data, 25)
    Q3 = percentile_(data, 75)

    # Interquartile range
    IQR = Q3 - Q1

    # Outlier Boundaries
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    return np.sum((data < lower_bound) | (data > upper_bound))


# def descriptive_stat(data: np.ndarray, cols: list = None):
#     # columns
#     cols_ = [f"attr{x}" for x in range(data.shape[1])] if cols is None else cols

#     # total count
#     n_total = len(data)

#     # valid count
#     n_valid = len(np.isfinite(data.astype(float)))

#     # mean
#     mean_ = np.round(np.mean(data, axis=0), 4)

#     # std
#     std_ = np.round(np.std(data, axis=0), 4)

#     # min
#     min_ = np.round(np.min(data, axis=0), 4)

#     # 25 perc
#     p_25 = np.round(np.percentile(data, 25, axis=0), 4)

#     # 50 perc
#     p_50 = np.round(np.percentile(data, 50, axis=0), 4)

#     # 75 perc
#     p_75 = np.round(np.percentile(data, 75, axis=0), 4)

#     # max
#     max_ = np.round(np.max(data, axis=0), 4)

#     # metrics stacked
#     metrics = {
#         "n_total": [n_total] * len(cols_),
#         "n_valid": [n_valid] * len(cols_),
#         "mean": mean_,
#         "std": std_,
#         "min": min_,
#         "25%": p_25,
#         "50%": p_50,
#         "75%": p_75,
#         "max": max_,
#     }

#     # stats dict
#     stat = {col: {f"{k}": v[idx] for k, v in metrics.items()} for idx, col in enumerate(cols_)}

#     return stat


def corr_matrix(data: np.ndarray, **kwargs):
    """Computes the correlation matrix for the input data.

    Args:
        data (np.ndarray): Input dataset.
        col_map (dict): Column mapper for encoding categorical attributes.
        corr_type (str, optional): Type of correlation to compute ('pearson', 'kendall', 'spearman'). Defaults to 'pearson'.

    Returns:
        np.ndarray: Correlation matrix.
    """

    # unique values
    unq = np.unique(data)

    # compute linear correlation matrix
    if len(unq) > 1:
        corr_m = np.corrcoef(data.astype(float), rowvar=False)
    else:
        corr_m = np.ones((data.shape[-1], data.shape[-1])).astype(float)

    return corr_m


def corr_strong(corr_matrix: np.ndarray, cols: list, c_thresh: float = 0.5, **kwargs):
    """Computes correlation statistics for each column based on a correlation
    threshold.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        c_thresh (float, optional): Correlation threshold for identifying correlated columns. Defaults to 0.5.

    Returns:
        dict: Correlation statistics.
    """
    stats_d = {}

    # filter out weak relationships
    corr_ = np.array(corr_matrix.__abs__() > c_thresh)

    # numpy column array
    cols_ = np.array(cols)

    # check strong correlations per column
    for idx, col_ in enumerate(cols):
        # strong correlation indices
        corr_ind = np.where(corr_[:, idx])[0]
        corr_ind = corr_ind[corr_ind != idx]  # remove the target column

        # get strongly correlation columns and its values
        corr_cols = cols_[corr_ind]
        corr_vals = corr_matrix[idx, corr_ind]

        stats_d[col_] = {col: val for col, val in zip(corr_cols, corr_vals)}

    return stats_d


def uniformity_score_per_column(
    data_col: np.ndarray,
    is_continuous: bool,
    n_categories: int = None,
    **kwargs,
):
    """Detect target attribute imbalancing likelihood. This function computes
    the CV, Entropy and IQR metrics to assess an imbalancing score from the
    target attribute side. Rule of thumb: CV < [0.10, 0.15], IQR < 0.25*mean,
    entropy.

    Args:
        data (np.ndarray): _description_
        target_col (str): _description_

    Returns:
        _type_: _description_
    """

    # target attribute distribution
    if is_continuous:
        # create histogram for continuous data
        tag = "continuous"
        var, _ = np.histogram(data_col, bins="auto", density=True)
    else:
        # calculate counts for discrete data
        tag = "discrete"
        _, var = np.unique(data_col, return_counts=True)

    # normalise counts
    var = var / var.sum()

    # if specified, adjust for expected number of categories
    if isinstance(n_categories, int) and not is_continuous:
        n_diff = n_categories - len(var)
        var = np.append(var, [0] * n_diff)

    # calculate scores
    stat_score = stat_test_uniformity(data=data_col, var=var, tag=tag)
    entropy_score = round(entropy_(var), 4)

    # define flags based on heuristic thresholds
    stat_flag = stat_score > 0.80
    entropy_flag = entropy_score > 0.95

    # calculate imbalance level
    imb_level = round(100 * (1 - np.mean([stat_flag, entropy_flag])), 1)

    return stat_score, entropy_score, imb_level


# def dtype_inference(data: pd.DataFrame, **kwargs):
#     check_map = {"dtype": []}  # dtype validation map {col: check}

#     # Schema Validation
#     for col in data.columns:
#         check_map["dtype"].append(data[col].dtype.__str__())

#     return pd.DataFrame(check_map, index=data.columns).T


# def consistency_check(data: pd.DataFrame, categorical_map: dict, **kwargs):
#     # Consistency Check
#     for col, avail_categ in categorical_map.items():
#         unique_categories = data[col].unique()
#         predefined_categories = [unq in avail_categ for unq in unique_categories]
#         if not set(unique_categories).issubset(predefined_categories):
#             return False
#     return True


# def estimate_eps(embeddings: np.ndarray, k: int):
#     # Compute distances to k nearest neighbors
#     nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
#     distances, _ = nbrs.kneighbors(embeddings)

#     # Take the average of distances to k nearest neighbors
#     avg_distances = np.mean(distances[:, 1:], axis=1)

#     # Choose eps as a fraction of the average distance
#     eps = np.percentile(avg_distances, 90)  # Adjust the percentile as needed

#     return eps


# def proximity_score(embeddings: np.ndarray, n_neighbors: int = 4, **kwargs):
#     # estimate eps param for DBSCAN initialization
#     eps_ = estimate_eps(embeddings, k=n_neighbors)

#     # perform DBSCAN clustering
#     dbscan = DBSCAN(eps=eps_)
#     cluster_lbl = dbscan.fit_predict(embeddings)

#     # number of clusters found (-1 indicates noisy samples)
#     n_clusters = len(np.unique(cluster_lbl[cluster_lbl > -1]))

#     # compute silhouette score
#     if n_clusters > 1:
#         score = silhouette_score(
#             embeddings[cluster_lbl > -1],
#             cluster_lbl[cluster_lbl > -1],
#         )
#     else:
#         score = 1

#     return score


# GDPR privacy
def compute_k_anonymity(data: np.ndarray, column_names: List[str], qi_names: List[str] = None, **kwargs):
    # convert into dataframe
    aux_data = pd.DataFrame(data, columns=column_names)

    # compute k-anonimity
    if isinstance(qi_names, list):
        if len(qi_names) > 0:
            # get k-anom value
            k_anom = anonymity.k_anonymity(aux_data, qi_names)

            # convert into percentage (the higher the better privacy is preserved)
            k_anom = 100 * (1 - k_anom / len(aux_data))
        else:
            k_anom = -1
    else:
        k_anom = -1

    return float(k_anom)


# multicolinearity
def compute_vif(data: np.ndarray, column_names: List[str], **kwargs):
    vif_map, vif_val = {}, []

    for idx, col in enumerate(column_names):
        # column-wise vif
        aux_vif = variance_inflation_factor(data, idx)

        # assign vif
        vif_map[col] = aux_vif
        vif_val.append(aux_vif > 5)  # multicolinearity threshold

    # get percentage of columns above the threshold
    vif_perc = sum(vif_val) / len(vif_val)

    return vif_perc, vif_map
