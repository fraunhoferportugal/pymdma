from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from ..data.utils import is_categorical, is_numeric
from ..models.classification import ClassificationML
from ..models.regression import RegressionML

device = "cpu"  # matrices are too big for gpu
feature_metrics = {"categorical": "hamming", "numerical": "euclidean"}


def _get_cdf(data, nbins: int = 50):
    # histogram
    count, _ = np.histogram(data, bins=nbins)

    # probability distribution function
    pdf = count / sum(count)

    # cumulative distribution function
    cdf = np.cumsum(pdf)

    return cdf


def _get_change_perc(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        perc = (df1 - df2).divide(df2 + 1e-20) * 100
    except TypeError as e:
        logger.debug(f"Error:\n{e}")
        perc = None
    return perc


def _get_nn_model(data: np.ndarray, distance_type: str = "euclidean"):
    """Find nearest neighbors of test in train with first categoric_slice-many
    variables being categorical."""
    # get nn model
    nearest_neighbor_model = NearestNeighbors(
        metric=distance_type,
        algorithm="ball_tree",
        n_jobs=None,
    )

    # fit data
    nearest_neighbor_model.fit(data)

    return nearest_neighbor_model


def _get_nn_distances(
    tgt_emb: np.ndarray,
    syn_emb: np.ndarray,
    distance_type: dict = "euclidean",
    size: int = None,
) -> Tuple[np.ndarray]:
    # checkpoint
    assert tgt_emb.shape[1] == syn_emb.shape[1], "Train and Syn have mismatched columns"

    # split into tgt_train, tgt_query, and syn_query
    if size is None:
        tgt_size, syn_size = len(tgt_emb), len(syn_emb)
    else:
        tgt_size, syn_size = size, size

    # train and query from target
    tgt_query = tgt_emb[-int(tgt_size) :]

    # syn_train is not needed
    # if sample_size = synthetic_size, syn_query is all syn dataset
    syn_query = syn_emb[-int(syn_size) :]

    # training model
    nn_model = _get_nn_model(tgt_query, distance_type)

    # get nearest neighbors
    # target
    tgt_query_nn, _ = nn_model.kneighbors(tgt_query, n_neighbors=3)
    tgt_query_nn = tgt_query_nn[:, 1:]  # except the closest (itself)

    # synthetic
    syn_query_nn, _ = nn_model.kneighbors(syn_query, n_neighbors=2)

    # calculating DCR NNDR
    query_dict = {"tgt": tgt_query_nn, "syn": syn_query_nn}

    # compute privacy scores
    dcr, nndr = {}, {}
    for label, query in query_dict.items():
        # closest neighbor
        aux_dcr = query[:, 0]

        # normalized closest neighbor distances
        aux_nndr = aux_dcr / (query[:, 1] + 1e-10)

        # assign
        dcr[label] = aux_dcr
        nndr[label] = aux_nndr

    return dcr, nndr


def _get_gaussian_kde(data: np.ndarray, bins: np.ndarray):
    # get unique values
    unq = np.unique(data)

    # check condition
    if len(unq) > 1:
        # sample from distribution
        pdf_data = gaussian_kde(
            data,
        ).pdf(bins)

        # normalize
        pdf_data /= sum(pdf_data)

    else:
        # single value bin index
        arg_idx = np.argmin(abs(bins - unq[0]))

        # gen default distribution
        pdf_data = np.zeros_like(bins, dtype=float)

        # assign single value bin
        pdf_data[arg_idx] = 1.0

    return pdf_data


def _get_nn_pdf(
    tgt_dist: np.ndarray,
    syn_dist: np.ndarray,
) -> Tuple[np.ndarray]:

    # get distributions bins
    t_min, t_max = min(tgt_dist), max(tgt_dist)
    s_min, s_max = min(syn_dist), max(syn_dist)

    bins = np.linspace(
        min([t_min, s_min]),
        max([t_max, s_max]),
        300,
    )

    # get distributions
    # tgt pdf dists
    pdf_tgt = _get_gaussian_kde(tgt_dist, bins)

    # syn pdf dists
    pdf_syn = _get_gaussian_kde(syn_dist, bins)

    return pdf_tgt, pdf_syn, bins


def _get_pp_metrics(
    tgt_pdf: np.ndarray,
    syn_pdf: np.ndarray,
    bins: np.ndarray,
    low_perc: int = 10,
    high_perc: int = 90,
    tag: str = "dist",
):
    # overlap
    overlap = np.minimum(tgt_pdf, syn_pdf).sum()

    # percentiles real
    tgt_cumsum = np.cumsum(tgt_pdf)
    p_low = np.where(tgt_cumsum >= low_perc / 100)[0][0]
    p_high = np.where(tgt_cumsum >= high_perc / 100)[0][0]

    # percentiles synthetic
    syn_cumsum = np.cumsum(syn_pdf)
    p_low_s = np.where(syn_cumsum >= low_perc / 100)[0][0]
    p_high_s = np.where(syn_cumsum >= high_perc / 100)[0][0]

    # synthetic samples within percentiles
    syn_leak = syn_pdf[:p_low].sum()
    syn_nleak = syn_pdf[p_high:].sum()
    syn_fidel = 1 - syn_leak - syn_nleak
    syn_div = (p_high_s - p_low_s) / (max([p_high, p_high_s]) - min([p_low, p_low_s]))

    # centroid displacement
    tgt_ctd = np.dot(tgt_pdf, bins)
    syn_ctd = np.dot(syn_pdf, bins)
    ctd_disp = (syn_ctd - tgt_ctd) / (max(bins) - min(bins))
    ctd_disp = (ctd_disp + 1) / 2  # min-max norm (-1/1 to 0/1)

    # metrics dict
    metric_d = {
        "dist": tag,
        "%_overlap": round(overlap * 100, 1),
        "%_leak": round(syn_leak * 100, 1),
        "%_nfidel": round(syn_nleak * 100, 1),
        "%_fidel": round(syn_fidel * 100, 1),
        "%_divers": round(syn_div * 100, 1),
        "%_ctd_disp": round(ctd_disp * 100, 1),
        "bin_low": bins[p_low],
        "bin_high": bins[p_high],
    }

    # result string
    metric_s = " | ".join([f"{m.split('_')[-1]}:{val}%" for m, val in metric_d.items() if "syn" in m])

    # add to dict
    metric_d["prompt"] = metric_s

    return metric_d


def assign_utility_experiment(
    train_tgt: pd.DataFrame,
    train_syn: pd.DataFrame,
    test_tgt: pd.DataFrame,
    target_col: str = "",
    kind: str = "rr",
    task_class: callable = ClassificationML,
    **kwargs,
):
    # assign experiment
    if kind == "rr":
        train_, test_ = train_tgt, test_tgt
    elif kind == "sr":
        train_, test_ = train_syn, test_tgt
    elif kind == "srr":
        train_, test_ = pd.concat((train_tgt, train_syn), axis=0), test_tgt
    else:
        train_, test_ = None, None

    exp = task_class(train_, test_, target_col)

    return exp


def compute_utility_scores(
    real_data: np.ndarray,
    syn_data: np.ndarray,
    cols: list,
    target_col: str,
    train_cols: list,
    kind: List = ["rr", "sr", "srr"],
    seed: int = 84,
    **kwargs,
):

    # columns (target and train)
    job_cols = train_cols + [target_col]

    # real data
    X_real_total = pd.DataFrame(real_data, columns=cols)[job_cols]

    # synthetic data
    X_syn_total = pd.DataFrame(syn_data, columns=cols)[job_cols]

    # detect target variable type (continuous, categorical)
    is_categ = is_categorical(
        np.concatenate((X_real_total[target_col], X_syn_total[target_col])),
    )

    # split real data for train/test
    X_real_train, X_real_test, _, _ = train_test_split(
        X_real_total,
        X_real_total,
        random_state=seed,
        stratify=(X_real_total[target_col] if is_categ else None),
    )

    # reset indices
    X_train = X_real_train.reset_index(drop=True)
    X_test = X_real_test.reset_index(drop=True)

    # select the task type
    task_class = ClassificationML if is_categ else RegressionML

    # utility evaluation strategy
    score_map = {}
    for k in kind:
        # Utility experiment
        utility_exp = assign_utility_experiment(
            train_tgt=X_train,
            train_syn=X_syn_total,
            test_tgt=X_test,
            target_col=target_col,
            kind=k,
            task_class=task_class,
        )

        # run
        if is_categ:
            utility_exp.prepare_run_(
                normalize=True,
                fix_imbalance=True,
            )
        else:
            utility_exp.prepare_run_(
                normalize=True,
            )

        # compare_models
        run_ = utility_exp.do_run_(
            cross_validation=False,
        )

        # pick best model
        if is_categ:
            # get dataframe results
            df_scores = utility_exp.exp.pull().sort_values("Model").iloc[:, 1:-3]

            # classification (maximize score)
            df_scores = df_scores[df_scores["F1"] == df_scores["F1"].max()].iloc[:1]
        else:
            # get dataframe results
            df_scores = utility_exp.exp.pull().sort_values("Model").iloc[:, [1, 2, 3]]

            # regression (minimize score)
            df_scores = df_scores[df_scores["MAE"] == df_scores["MAE"].min()].iloc[:1]

        logger.debug(df_scores)

        # reset indices
        df_scores.reset_index(drop=True, inplace=True)

        # assign to mapper
        score_map[k] = df_scores

    return score_map


def _get_kl_divergence(real_pdf: np.ndarray, syn_pdf: np.ndarray):
    """Compute the Kullback-Leibler (KL) divergence between two probability
    distributions.

    Parameters:
    - real_pdf, syn_pdf: 1-D series representing the two probability distributions.

    Returns:
    - kl_score: KL divergence score.
    """

    eps = 1e-10

    # compute KL divergence
    kl_score = np.ma.masked_invalid(
        np.where(real_pdf != 0, real_pdf * np.log(real_pdf / (syn_pdf + eps)), 0),
    ).sum()

    return np.maximum(kl_score, 0)


def _get_js_divergence(real_pdf: np.ndarray, syn_pdf: np.ndarray):
    """Compute the Jensen-Shannon (JS) divergence between two probability
    distributions.

    Parameters:
    - real_pdf, syn_pdf: 1-D series representing the two probability distributions.

    Returns:
    - js_score: JS divergence score.
    """

    eps = 1e-10

    # Calculate midpoint distribution
    m = 0.5 * (real_pdf + syn_pdf)

    # Compute JS divergence
    kl_tgt = np.ma.masked_invalid(real_pdf * np.log2(real_pdf / (m + eps))).sum()
    kl_syn = np.ma.masked_invalid(syn_pdf * np.log2(syn_pdf / (m + eps))).sum()

    js_score = 0.5 * (kl_tgt + kl_syn)

    return np.maximum(js_score, 0)


def _get_ks_similarity(real_data: np.ndarray, syn_data: np.ndarray):
    """Compute attribute similarity for continuous attributes using Kolmogorov-
    Smirnov statistics.

    Parameters
    ----------
    real_col : np.ndarray
        The target dataset attribute.
    syn_col : np.ndarray
        The synthetic dataset attribute.

    Returns
    ----------
    float
        KS score
    """
    # define bin range
    _min = np.concatenate([real_data, syn_data]).min()  # min
    _max = np.concatenate([real_data, syn_data]).max()  # max

    bins = np.linspace(_min, _max, 100)  # histogram bins

    # original cdf
    real_cdf = _get_cdf(real_data, nbins=bins)

    # synthetic cdf
    syn_cdf = _get_cdf(syn_data, nbins=bins)

    # ks statistic
    ks_score = 1 - np.max(real_cdf - syn_cdf)

    return ks_score


def _get_tv_similarity(real_data: np.ndarray, syn_data: np.ndarray) -> float:
    """Compute attribute similarity for categorical attributes.

    Parameters
    ----------
    real_col : np.ndarray
        The target dataset attribute.
    syn_col : np.ndarray
        The synthetic dataset attribute.

    Returns
    -------
    float
        TV similarity score
    """

    # original cdf
    real_unq, real_counts = np.unique(real_data, return_counts=True)
    real_map = {unq: count / sum(real_counts) for unq, count in zip(real_unq, real_counts)}

    # synthetic cdf
    syn_unq, syn_counts = np.unique(syn_data, return_counts=True)
    syn_map = {unq: count / sum(syn_counts) for unq, count in zip(syn_unq, syn_counts)}

    # tv statistics
    scores = [abs(real_map.get(unq) - syn_map.get(unq, 0)) for unq in real_unq]

    # tv score
    tv_score = 1 - (np.sum(scores) / len(real_unq))

    return tv_score
