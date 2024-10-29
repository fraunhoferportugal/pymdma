from os.path import join
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from pymdma import config as cfg
from pymdma.tabular.data import load as load
from pymdma.tabular.embeddings import embed, scale
from pymdma.tabular.measures.input_val.data import privacy as priv_inp
from pymdma.tabular.measures.input_val.data import quality as qual_inp
from pymdma.tabular.measures.synthesis_val.data import similarity as sim_syn
# from pymdma.tabular.measures.synthesis_val.data import utility as util_syn
from pymdma.tabular.measures.synthesis_val.feature import privacy as priv_syn


def test_check():
    assert True


####################################################################################################
################################# INPUT VALIDATION TESTS ###########################################
####################################################################################################
@pytest.mark.parametrize(
    "n_cols, corr_thresh",
    [
        (10, 0.5),
        (10, 0.9),
        (50, 0.0),
        (50, 0.9),
        (10, 0.8),
    ],
)
def test_qual_input__corr(n_cols: int, corr_thresh: float, **kwargs):
    # column names
    col_names = [f"att_{idx}" for idx in range(n_cols)]

    # data shape
    data_shape = (200, len(col_names))

    # data
    # get random data (normal distribution)
    data_norm = np.random.default_rng(42).normal(0, 1, size=data_shape)

    # get random data (uniform distribution)
    data_unif = np.random.default_rng(42).uniform(-2, 2, size=data_shape)

    # get dataset with one single value
    data_eq = np.ones(shape=data_shape)

    # metric class
    corr_inp = qual_inp.CorrelationScore(col_names, corr_thresh, **kwargs)

    # get scores
    metric_norm = corr_inp.compute(data_norm)
    metric_unif = corr_inp.compute(data_unif)
    metric_eq = corr_inp.compute(data_eq)

    # values
    dataset_level_norm = metric_norm.dataset_level
    dataset_level_unif = metric_unif.dataset_level
    dataset_level_eq = metric_eq.dataset_level

    # checkpoints
    # 1
    assert dataset_level_norm is not None, "Dataset level is None"
    assert dataset_level_unif is not None, "Dataset level is None"
    assert dataset_level_eq is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level_norm.value, dict), "Dataset level is not of evaluation level type"
    assert isinstance(dataset_level_unif.value, dict), "Dataset level is not of evaluation level type"
    assert isinstance(dataset_level_eq.value, dict), "Dataset level is not of evaluation level type"

    # 3
    assert len(dataset_level_norm.value) == len(col_names), "Number of instances do not match"
    assert len(dataset_level_unif.value) == len(col_names), "Number of instances do not match"
    assert len(dataset_level_eq.value) == len(col_names), "Number of instances do not match"

    # 4
    assert 0 <= dataset_level_norm.stats.get("mean") <= 100, "Mean value should be between 0 and 1"
    assert 0 <= dataset_level_unif.stats.get("mean") <= 100, "Mean value should be between 0 and 1"
    assert dataset_level_eq.stats.get("mean") == 100, "Mean value should equal 100"
    assert dataset_level_eq.stats.get("std") == 0, "Stddev value should equal 0"


@pytest.mark.parametrize(
    "perc_dupl",
    [
        100,
        50,
        0,
    ],
)
def test_qual_input__uniq(perc_dupl: float, **kwargs):
    # data shape
    n_samp, n_cols = (500, 20)

    # data
    data_dupl = np.random.default_rng(42).normal(10, 3, size=(1, n_cols))[[0] * int(n_samp * perc_dupl / 100)]

    # get data from % duplicates requested
    if perc_dupl < 100:
        # get non-duplicated random portion
        data_rnd = np.random.default_rng(42).normal(0, 6, size=(n_samp - len(data_dupl), n_cols))

        # duplicated + non-duplicated
        data = np.concatenate((data_rnd, data_dupl))

        # shuffle rows
        np.random.default_rng(42).shuffle(data)

    elif perc_dupl == 100:
        # data is comprised of duplicated rows
        data = data_dupl.copy()

    else:
        raise ValueError("'perc_dupl' should behave between 0 and 100.")

    # metric class
    unq_inp = qual_inp.UniquenessScore(**kwargs)

    # get scores
    metric_norm = unq_inp.compute(data)

    # values
    dataset_level_norm = metric_norm.dataset_level

    # checkpoints
    # 1
    assert dataset_level_norm is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level_norm.value, float), "Dataset level is not of evaluation level type"

    # 3
    assert dataset_level_norm.value == perc_dupl, "Obtained % duplicates should match the requested value"


@pytest.mark.parametrize(
    "data, tag, exp_level",
    [
        (np.random.default_rng(42).uniform(size=(2000, 10)), "continuous", 0),
        (np.random.default_rng(42).normal(size=(2000, 10)), "continuous", 100),
        (np.ones((2000, 10)), "discrete", 0),
        (np.zeros((2000, 10)), "discrete", 0),
        (np.concatenate((np.ones((1990, 10)), np.zeros((10, 10)))), "discrete", 100),
    ],
)
def test_qual_input__unif(data: float, tag: str, exp_level: float, **kwargs):
    # columns and unq values
    col_names = [f"att_{idx}" for idx in range(data.shape[1])]
    unq_data = np.unique(data) if tag == "discrete" else []

    # column map
    col_map = {
        col: {
            "type": {
                "tag": tag,
                "opt": {unq: unq for unq in unq_data},
            },
        }
        for col in col_names
    }

    # metric class
    unif_inp = qual_inp.UniformityScore(col_names, col_map, **kwargs)

    # get scores
    metric = unif_inp.compute(data)

    # values
    dataset_level = metric.dataset_level

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.value, dict), "Dataset level result should be of 'dict' type"

    # 3
    assert list(dataset_level.value.keys()) == col_names, "Output and Input columns do not match"

    # 4
    assert np.isclose(
        dataset_level.stats.get("mean"),
        exp_level,
        atol=10,
    ), "Obtained metric does not match the expected value"


@pytest.mark.parametrize(
    "data, column_names, exp_out_perc",
    [
        (
            np.random.default_rng(42).uniform(-0.5, 0.5, (100, 3)),
            ["col1", "col2", "col3"],
            0.0,
        ),  # uniform data, expect ~0% outliers
        (
            np.concatenate(
                [np.random.default_rng(42).normal(0, 1, (90, 3)), np.random.default_rng(42).normal(10, 1, (10, 3))],
            ),
            ["col1", "col2", "col3"],
            10.0,
        ),  # 10% of data as outliers
        (
            np.random.default_rng(42).normal(0, 10, (100, 10)),
            ["col1", "col2", "col3"],
            0.0,
        ),  # normal distribution, typically few outliers
    ],
)
def test_outliers_score(data: np.ndarray, column_names: List[str], exp_out_perc: float):
    # initialize the metric
    outliers_score = qual_inp.OutlierScore(column_names)

    # compute the metric
    metric = outliers_score.compute(data)

    # extract dataset-level values
    dataset_level = metric.dataset_level

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.value, dict), "Dataset level result should be of 'dict' type"
    assert isinstance(dataset_level.stats, dict), "Dataset level stats should be of 'dict' type"

    # 3
    assert np.isclose(dataset_level.stats.get("mean"), exp_out_perc, atol=5.0), "% Outliers mismatch"


@pytest.mark.parametrize(
    "data, exp_miss_samp, exp_miss_col",
    [
        (np.random.default_rng(42).uniform(size=(100, 10)), 0.0, 0.0),
        (np.array([[np.nan] * 10] * 100), 100.0, 100.0),
        (np.hstack((np.random.default_rng(42).uniform(size=(100, 9)), np.full((100, 1), np.nan))), 10.0, 10.0),
    ],
)
def test_missing_score(data: np.ndarray, exp_miss_samp: float, exp_miss_col: float):
    # column names
    column_names = [f"att_{idx}" for idx in range(data.shape[-1])]

    # initialize the metric
    miss_score = qual_inp.MissingScore(column_names)

    # compute the metric
    metric = miss_score.compute(data)

    # extract dataset-level values
    dataset_level = metric.dataset_level

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.value, dict), "Dataset level result should be of 'list' type"

    # 3
    assert np.isclose(dataset_level.stats.get("sample"), exp_miss_samp, atol=0.01), "Sample missing rate mismatch"

    # 4
    assert np.isclose(dataset_level.stats.get("column"), exp_miss_col, atol=0.01), "Column missing rate mismatch"


@pytest.mark.parametrize(
    "data, exp_ratio",
    [
        (np.random.default_rng(42).uniform(size=(100, 10)), 10.0),
        (np.random.default_rng(42).uniform(size=(1000, 10)), 100.0),
        (np.random.default_rng(42).uniform(size=(50, 50)), 1.0),
    ],
)
def test_dim_curse_score(data: np.ndarray, exp_ratio: float):
    # initialize the metric
    dim_score = qual_inp.DimCurseScore()

    # compute the metric
    metric = dim_score.compute(data)

    # extract dataset-level values
    dataset_level = metric.dataset_level

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.value, float), "Dataset level result should be of 'float' type"

    # 3
    assert np.isclose(dataset_level.value, exp_ratio, atol=0.0), "Dimensionality curse ratio mismatch"


@pytest.mark.parametrize(
    "data, column_names, qi_names, exp_k",
    [
        (np.array([[1, 2], [1, 3], [1, 2], [1, 2]]), ["col1", "col2"], ["col1", "col2"], 75),
        (np.array([[1, 2]] * 4), ["col1", "col2"], ["col1", "col2"], 0),
        (np.array([[1, 2]] * 4), ["col1", "col2"], [], -1),
    ],
)
def test_k_anonymity(data: np.ndarray, column_names: List[str], qi_names: List[str], exp_k: int):
    # initialize the metric
    k_anon_score = priv_inp.KAnonymityScore(
        column_names=column_names,
        qi_names=qi_names,
    )

    # compute the metric
    metric = k_anon_score.compute(data)

    # extract dataset-level values
    dataset_level = metric.dataset_level

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.value, float), "Dataset level result should be of 'float' type"

    # 3
    assert dataset_level.value == exp_k, "k-Anonymity value mismatch"


@pytest.mark.parametrize(
    "data",
    [
        (np.random.default_rng(42).uniform(size=200)),
        (np.random.default_rng(42).normal(size=200)),
    ],
)
def test_vif_score(data: np.ndarray):
    # data (linear combinations)
    D1 = data
    D2 = data * 2
    D3 = D2 * 4
    D4 = D3 * 8

    # data (non-linear combinations)
    D5 = data
    D6 = np.random.normal(0, 1, size=len(data))
    D7 = np.random.normal(0, 1, size=len(data))
    D8 = np.random.normal(0, 1, size=len(data))

    # data sets (low / high VIF scores)
    data_high = np.column_stack((D1, D2, D3, D4))
    data_low = np.column_stack((D5, D6, D7, D8))

    # get column names
    column_names = [f"att_{idx}" for idx in range(data_low.shape[-1])]

    # initialize the metric
    vif_score = qual_inp.VIFactorScore(column_names=column_names)

    # compute the metric
    metric_low = vif_score.compute(data_low)
    metric_high = vif_score.compute(data_high)

    # extract dataset-level values
    dataset_level_low = metric_low.dataset_level
    dataset_level_high = metric_high.dataset_level

    # checkpoints
    # 1
    assert dataset_level_low is not None, "Dataset level is None"
    assert dataset_level_high is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level_low.value, float), "Dataset level result should be of 'float' type"
    assert isinstance(dataset_level_high.value, float), "Dataset level result should be of 'float' type"

    # 3
    assert dataset_level_low.value == 0.0, "VIF score value mismatch"
    assert dataset_level_high.value == 1.0, "VIF score value mismatch"


####################################################################################################
################################# SYNTHETIC METRICS TESTS ##########################################
####################################################################################################
@pytest.mark.parametrize(
    "real_data, syn_data, col_map",
    [
        # Continuous data with high similarity
        (
            np.random.default_rng(42).normal(size=(100, 3)),  # real data
            np.random.default_rng(42).normal(size=(100, 3)),  # synthetic data
            {
                "att_0": {"type": {"tag": "continuous"}},
                "att_1": {"type": {"tag": "continuous"}},
                "att_2": {"type": {"tag": "continuous"}},
            },
        ),
        # Discrete data with low similarity
        (
            np.random.default_rng(42).choice([0, 1], size=(100, 3)),  # real data
            np.random.default_rng(42).choice([0, 1], size=(100, 3)),  # synthetic data
            {
                "att_0": {"type": {"tag": "discrete"}},
                "att_1": {"type": {"tag": "discrete"}},
                "att_2": {"type": {"tag": "discrete"}},
            },
        ),
        # Mixed data types
        (
            np.column_stack(
                [
                    np.random.default_rng(42).normal(size=100),
                    np.random.default_rng(42).choice([0, 1], size=100),
                    np.random.default_rng(42).normal(size=100),
                ],
            ),  # real data
            np.column_stack(
                [
                    np.random.default_rng(42).normal(size=100),
                    np.random.default_rng(42).choice([0, 1], size=100),
                    np.random.default_rng(42).normal(size=100),
                ],
            ),  # synthetic data
            {
                "att_0": {"type": {"tag": "continuous"}},
                "att_1": {"type": {"tag": "discrete"}},
                "att_2": {"type": {"tag": "continuous"}},
            },
        ),
    ],
)
def test_statistical_sim_score(real_data, syn_data, col_map):
    # initialize the metric
    stat_sim_score = sim_syn.StatisticalSimScore(col_map=col_map)

    # compute the metric
    metric_result = stat_sim_score.compute(real_data, syn_data)

    # extract dataset-level values
    dataset_level = metric_result.dataset_level
    global_stats = dataset_level.stats

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.stats, dict), "Similarity scores should be a dictionary"

    # 3
    for score in dataset_level.value.values():
        assert isinstance(score, float), "Similarity score should be of 'float' type"
        assert 0.0 <= score <= 1.0, "Similarity score should be between 0.0 and 1.0"

    # 4
    assert isinstance(global_stats.get("mean"), float), "Global mean score should be of 'float' type"
    assert isinstance(global_stats.get("std"), float), "Global std score should be of 'float' type"


@pytest.mark.parametrize(
    "real_data, syn_data, col_map, score_type",
    [
        # KL divergence with normal distributions
        (
            np.random.default_rng(42).normal(size=(100, 3)),  # real data
            np.random.default_rng(42).normal(size=(100, 3)),  # synthetic data
            {
                "att_0": {"type": {"tag": "continuous"}},
                "att_1": {"type": {"tag": "continuous"}},
                "att_2": {"type": {"tag": "continuous"}},
            },
            "kl",
        ),
        # JS divergence with uniform distributions
        (
            np.random.default_rng(42).uniform(size=(100, 3)),  # real data
            np.random.default_rng(42).uniform(size=(100, 3)),  # synthetic data
            {
                "att_0": {"type": {"tag": "continuous"}},
                "att_1": {"type": {"tag": "continuous"}},
                "att_2": {"type": {"tag": "continuous"}},
            },
            "js",
        ),
        # All divergence types with mixed data
        (
            np.column_stack(
                [
                    np.random.default_rng(40).normal(size=100),
                    np.random.default_rng(50).uniform(size=100),
                    np.random.default_rng(60).normal(size=100),
                ],
            ),  # real data
            np.column_stack(
                [
                    np.random.default_rng(10).normal(size=100),
                    np.random.default_rng(20).uniform(size=100),
                    np.random.default_rng(30).normal(size=100),
                ],
            ),  # synthetic data
            {
                "att_0": {"type": {"tag": "continuous"}},
                "att_1": {"type": {"tag": "continuous"}},
                "att_2": {"type": {"tag": "continuous"}},
            },
            "all",
        ),
    ],
)
def test_statistical_divergence_score(real_data: np.ndarray, syn_data: np.ndarray, col_map: dict, score_type: str):
    # initialize the metric
    stat_div_score = sim_syn.StatisiticalDivergenceScore(col_map=col_map, score=score_type)

    # compute the metric
    metric_result = stat_div_score.compute(real_data, syn_data)

    # extract dataset-level values
    dataset_level = metric_result.dataset_level
    global_stats = dataset_level.stats

    # checkpoints
    # 1
    assert dataset_level is not None, "Dataset level is None"

    # 2
    assert isinstance(dataset_level.value, dict), "Divergence scores should be a dictionary"

    # 3
    for col_scores in dataset_level.value.values():
        assert isinstance(col_scores, list), "Divergence scores for each column should be a list"
        assert all(isinstance(score, float) for score in col_scores), "Each divergence score should be a float"

    # 4
    assert isinstance(
        global_stats.get(f"{score_type}_mean"),
        float,
    ), f"Global {score_type} mean score should be of 'float' type"
    assert isinstance(
        global_stats.get(f"{score_type}_std"),
        float,
    ), f"Global {score_type} std score should be of 'float' type"

    # 5
    assert global_stats.get(f"{score_type}_mean") >= 0, f"Global {score_type} mean score should be non-negative"
    assert global_stats.get(f"{score_type}_std") >= 0, f"Global {score_type} std score should be non-negative"


@pytest.mark.parametrize(
    "real_data, syn_data, weights, corr_type, exp_corr_coh",
    [
        # Pearson correlation with equal weights
        (
            np.random.default_rng(42).normal(size=(100, 4)),  # real data
            np.random.default_rng(84).normal(size=(100, 4)),  # synthetic data
            None,  # equal weights
            "pearson",
            1.0,  # expected average correlation coherence (high similarity)
        ),
        # Kendall correlation with non-equal weights
        (
            np.random.default_rng(42).uniform(size=(100, 4)),  # real data
            np.random.default_rng(84).uniform(size=(100, 4)),  # synthetic data
            [0.1, 0.2, 0.3, 0.4],  # non-equal weights
            "kendall",
            1.0,  # expected average correlation coherence (high similarity)
        ),
        # Spearman correlation with non-equal weights and different data
        (
            np.random.default_rng(42).binomial(n=10, p=0.5, size=(100, 4)),  # real data
            np.random.default_rng(84).binomial(n=10, p=0.5, size=(100, 4)),  # synthetic data
            [0.25, 0.25, 0.25, 0.25],  # equal weights
            "spearman",
            1.0,  # expected average correlation coherence (high similarity)
        ),
        # Different data where the average correlation coherence is expected to be lower
        (
            np.random.default_rng(42).normal(size=(10000, 4)),  # real data
            np.ones((10000, 4)),  # synthetic data
            [1, 1, 1, 1],  # non-equal weights
            "pearson",
            0.5,  # expected average correlation coherence (low similarity)
        ),
    ],
)
def test_coherence_score(
    real_data: np.ndarray,
    syn_data: np.ndarray,
    weights: List[float],
    corr_type: str,
    exp_corr_coh: float,
):
    # initialize the metric
    coherence_score = sim_syn.CoherenceScore(corr_type=corr_type, weights=weights)

    # compute the metric
    metric_result = coherence_score.compute(real_data, syn_data)

    # extract dataset-level values
    avg_corr_coh = metric_result.dataset_level.value

    # checkpoints
    # 1
    assert avg_corr_coh is not None, "Average correlation coherence score is None"

    # 2
    assert isinstance(avg_corr_coh, float), "Average correlation coherence score should be of 'float' type"

    # 3
    assert 0.0 <= avg_corr_coh <= 1.0, "Average correlation coherence score should be between 0.0 and 1.0"

    # 4
    assert np.isclose(
        avg_corr_coh,
        exp_corr_coh,
        atol=0.1,
    ), "Average correlation coherence score does not match the expected value"


@pytest.mark.parametrize(
    "real_data, syn_data, distance_type, norm_type, exp_priv, exp_lvl",
    [
        # Euclidean distance with normal distributions
        (
            np.random.default_rng(42).normal(size=(300, 5)),  # real data
            np.random.default_rng(84).normal(size=(300, 5)),  # synthetic data
            "euclidean",  # distance type
            "standard",  # normalization type
            75,  # expected % privacy preservation
            50,  # expected % level of distribution displacement
        ),
        # Manhattan distance with very distinct uniform distributions
        (
            np.random.default_rng(42).uniform(0, 1, size=(300, 5)),  # real data
            np.random.default_rng(84).uniform(20, 30, size=(300, 5)),  # synthetic data
            "manhattan",  # distance type
            "minmax",  # normalization type
            100.0,  # expected % privacy preservation
            70,  # expected % level of distribution displacement
        ),
        # Different data with the same distance metric
        (
            np.random.default_rng(42).binomial(n=10, p=0.5, size=(300, 5)),  # real data
            np.random.default_rng(84).binomial(n=10, p=0.5, size=(300, 5)),  # synthetic data
            "euclidean",  # distance type
            "robust",  # normalization type
            70.0,  # expected % privacy preservation
            50.0,  # expected % level of distribution displacement
        ),
    ],
)
def test_nndr_privacy(
    real_data: np.ndarray,
    syn_data: np.ndarray,
    distance_type: str,
    norm_type: str,
    exp_priv: float,
    exp_lvl: float,
):
    # scaler
    scaler = scale.GlobalNorm(n_type=norm_type)

    # normalize data
    real_norm = scaler.fit_transform(real_data)
    syn_norm = scaler.transform(syn_data)

    # initialize the metric
    nndr_privacy = priv_syn.NNDRPrivacy(distance_type=distance_type)

    # compute the metric
    metric_result = nndr_privacy.compute(real_norm, syn_norm)

    # extract dataset-level values
    priv_d = metric_result.dataset_level.value

    # checkpoints
    # 1
    assert priv_d is not None, "Privacy metrics dictionary is None"

    # 2
    assert isinstance(priv_d, dict), "Privacy metrics should be a dictionary"

    # 3
    assert "privacy" in priv_d, "Privacy metrics should contain 'privacy' key"
    assert "level" in priv_d, "Privacy metrics should contain 'level' key"

    # 4
    assert isinstance(priv_d["privacy"], float), "'privacy' should be of 'float' type"
    assert isinstance(priv_d["level"], float), "'level' should be of 'float' type"
    assert 0.0 <= priv_d["privacy"] <= 100.0, "'privacy' should be between 0.0 and 100.0"
    assert 0.0 <= priv_d["level"] <= 100.0, "'level' should be between 0.0 and 100.0"

    # 5
    assert np.isclose(priv_d["privacy"], exp_priv, atol=10.0), "Privacy score does not match the expected value"
    assert np.isclose(
        priv_d["level"],
        exp_lvl,
        atol=10.0,
    ), "Magnitude of displacement does not match the expected value"


@pytest.mark.parametrize(
    "real_data, syn_data, distance_type, norm_type, exp_priv, exp_lvl",
    [
        # Euclidean distance with normal distributions
        (
            np.random.default_rng(42).normal(size=(300, 5)),  # real data
            np.random.default_rng(84).normal(size=(300, 5)),  # synthetic data
            "euclidean",  # distance type
            "maxabs",  # normalization type
            75.0,  # expected % privacy preservation
            50.0,  # expected % level of distribution displacement
        ),
        # Manhattan distance with very distinct uniform distributions
        (
            np.random.default_rng(42).uniform(0, 1, size=(300, 5)),  # real data
            np.random.default_rng(84).uniform(20, 30, size=(300, 5)),  # synthetic data
            "manhattan",  # distance type
            "identity",  # normalization type
            100.0,  # expected % privacy preservation
            90.0,  # expected % level of distribution displacement
        ),
        # Different data with the same distance metric
        (
            np.random.default_rng(42).binomial(n=10, p=0.5, size=(300, 5)),  # real data
            np.random.default_rng(84).binomial(n=10, p=0.5, size=(300, 5)),  # synthetic data
            "euclidean",  # distance type
            "standard",  # normalization type
            75.0,  # expected % privacy preservation
            50.0,  # expected % level of distribution displacement
        ),
        # Same data with a little noise added
        (
            np.random.default_rng(42).uniform(20, 30, size=(300, 5)),  # real data
            None,  # synthetic data
            "euclidean",  # distance type
            "standard",  # normalization type
            0,  # expected % privacy preservation
            15.0,  # expected % level of distribution displacement
        ),
    ],
)
def test_dcr_privacy(
    real_data: np.ndarray,
    syn_data: np.ndarray,
    distance_type: str,
    norm_type: str,
    exp_priv: float,
    exp_lvl: float,
):
    # scaler
    scaler = scale.GlobalNorm(n_type=norm_type)

    # normalize data
    real_norm = scaler.fit_transform(real_data)
    syn_norm = scaler.transform(syn_data if syn_data is not None else real_data.copy())

    # initialize the metric
    dcr_privacy = priv_syn.DCRPrivacy(distance_type=distance_type)

    # compute the metric
    metric_result = dcr_privacy.compute(real_norm, syn_norm)

    # extract dataset-level values
    priv_d = metric_result.dataset_level.value

    # checkpoints
    # 1
    assert priv_d is not None, "Privacy metrics dictionary is None"

    # 2
    assert isinstance(priv_d, dict), "Privacy metrics should be a dictionary"

    # 3
    assert "privacy" in priv_d, "Privacy metrics should contain 'privacy'"
    assert "level" in priv_d, "Privacy metrics should contain 'level'"

    # 4
    assert isinstance(priv_d["privacy"], float), "'privacy' should be of 'float' type"
    assert isinstance(priv_d["level"], float), "'level' should be of 'float' type"
    assert 0.0 <= priv_d["privacy"] <= 100.0, "'privacy' should be between 0.0 and 100.0"
    assert 0.0 <= priv_d["level"] <= 100.0, "'level' should be between 0.0 and 100.0"

    # 5
    assert np.isclose(priv_d["privacy"], exp_priv, atol=10.0), "Privacy score does not match the expected value"
    assert np.isclose(
        priv_d["level"],
        exp_lvl,
        atol=10.0,
    ), "Magnitude of displacement does not match the expected value"


# @pytest.mark.parametrize(
#     "real_params, syn_params, column_names, target_col, train_cols, task, high_qual_syn",
#     [
#         # Classification good quality
#         (
#             {
#                 "n_samples": 500,
#                 "n_features": 4,
#                 "n_clusters_per_class": 1,
#                 "n_classes": 2,
#                 "class_sep": 2.0,
#                 "random_state": 42,
#             },  # real data
#             {
#                 "n_samples": 500,
#                 "n_features": 4,
#                 "n_clusters_per_class": 1,
#                 "n_classes": 2,
#                 "class_sep": 2.0,
#                 "random_state": 84,
#             },  # synthetic data
#             ["feat1", "feat2", "feat3", "feat4", "target"],  # column_names (None means it will be auto-generated)
#             "target",  # target_col (None means the last column will be used)
#             ["feat1", "feat2", "feat3", "feat4"],  # train_cols (None means all columns except target will be used)
#             "clf",
#             True,
#         ),
#         # Regression good quality
#         (
#             {
#                 "n_samples": 500,
#                 "n_features": 4,
#                 "n_informative": 4,
#                 "n_targets": 1,
#                 "bias": 0.1,
#                 "noise": 0.0,
#                 "random_state": 42,
#             },  # real data
#             {
#                 "n_samples": 500,
#                 "n_features": 4,
#                 "n_informative": 4,
#                 "n_targets": 1,
#                 "bias": 0.1,
#                 "noise": 0.0,
#                 "random_state": 42,
#             },  # synthetic data
#             None,  # column_names (None means it will be auto-generated)
#             None,  # target_col (None means the last column will be used)
#             None,  # train_cols (None means all columns except target will be used)
#             "reg",
#             True,
#         ),
#         # Classification bad quality
#         (
#             {
#                 "n_samples": 500,
#                 "n_features": 4,
#                 "n_clusters_per_class": 1,
#                 "n_classes": 2,
#                 "class_sep": 1.0,
#                 "random_state": 42,
#             },  # real data
#             {
#                 "n_samples": 500,
#                 "n_features": 4,
#                 "n_clusters_per_class": 1,
#                 "n_classes": 2,
#                 "class_sep": 1.0,
#                 "shift": 40,
#                 "scale": 4.0,
#                 "random_state": 42,
#             },  # synthetic data
#             ["feat1", "feat2", "feat3", "feat4", "target"],  # column_names (None means it will be auto-generated)
#             "target",  # target_col (None means the last column will be used)
#             ["feat1", "feat2", "feat3", "feat4"],  # train_cols (None means all columns except target will be used)
#             "clf",
#             False,
#         ),
#         # Regression bad quality
#         (
#             {
#                 "n_samples": 300,
#                 "n_features": 4,
#                 "n_informative": 4,
#                 "n_targets": 1,
#                 "noise": 0.0,
#                 "random_state": 42,
#             },  # real data
#             {
#                 "n_samples": 300,
#                 "n_features": 4,
#                 "n_informative": 2,
#                 "n_targets": 1,
#                 "noise": 10.0,
#                 "random_state": 84,
#             },  # synthetic data
#             None,  # column_names (None means it will be auto-generated)
#             None,  # target_col (None means the last column will be used)
#             None,  # train_cols (None means all columns except target will be used)
#             "reg",
#             False,
#         ),
#     ],
# )
# def test_utility(
#     real_params: np.ndarray,
#     syn_params: np.ndarray,
#     column_names: List[str],
#     target_col: str,
#     train_cols: List[str],
#     task: str,
#     high_qual_syn: bool,
# ):
#     # generate data sets
#     # classification
#     if task == "clf":
#         # get data
#         real_feat, real_tgt = make_classification(**real_params)
#         syn_feat, syn_tgt = make_classification(**syn_params)

#     # regression
#     elif task == "reg":
#         # get data
#         real_feat, real_tgt = make_regression(**real_params)
#         syn_feat, syn_tgt = make_regression(**syn_params)

#     # stack features + target
#     real_data = np.column_stack([real_feat, real_tgt.reshape(-1, 1)])
#     syn_data = np.column_stack([syn_feat, syn_tgt.reshape(-1, 1)])

#     # initialize the metric
#     utility_metric = util_syn.Utility(
#         column_names=column_names,
#         target_col=target_col,
#         train_cols=train_cols,
#     )

#     # compute the metric
#     metric_result = utility_metric.compute(real_data, syn_data)

#     # extract dataset-level values
#     score_final_d = metric_result.dataset_level.value

#     # separated scores
#     rr_score = np.array(score_final_d.get("RR", []))
#     sr_score = np.array(score_final_d.get("SR", []))
#     srr_score = np.array(score_final_d.get("SRR", []))

#     # checkpoints
#     # 1
#     assert score_final_d is not None, "Utility metrics dictionary is None"

#     # 2
#     assert isinstance(score_final_d, dict), "Utility metrics should be a dictionary"

#     # 3
#     assert "RR" in score_final_d, "Utility metrics should contain 'RR'"
#     assert "SR" in score_final_d, "Utility metrics should contain 'SR'"
#     assert "SRR" in score_final_d, "Utility metrics should contain 'SRR'"

#     # 4
#     assert isinstance(score_final_d["RR"], list), "'RR' should be a list"
#     assert all(isinstance(val, float) for val in score_final_d["RR"]), "'RR' values should be of 'float' type"

#     assert isinstance(score_final_d["SR"], list), "'SR' should be a list"
#     assert all(isinstance(val, float) for val in score_final_d["SR"]), "'SR' values should be of 'float' type"

#     assert isinstance(score_final_d["SRR"], list), "'SRR' should be a list"
#     assert all(isinstance(val, float) for val in score_final_d["SRR"]), "'SRR' values should be of 'float' type"

#     # 5
#     if task in ["clf"]:
#         assert np.all((rr_score >= 0) & (rr_score <= 1)), "'RR' scores should be within [0, 1]"
#         assert np.all((sr_score >= 0) & (sr_score <= 1)), "'SR' scores should be within [0, 1]"
#         assert np.all((srr_score >= 0) & (srr_score <= 1)), "'SRR' scores should be within [0, 1]"

#     # 6
#     # metric scores
#     rr_mean = rr_score.mean()  # avg. raw scores
#     diff = np.abs(sr_score - rr_score).mean()  # syn vs real differences

#     # comparison
#     if task in ["clf"]:
#         # comparison
#         if high_qual_syn:
#             assert diff < 0.10, "Scores should not differ much from the original"
#         else:
#             assert diff > 0.30, "Scores should differ much from the original"

#     if task in ["reg"]:
#         if high_qual_syn:
#             assert diff / (rr_mean + 1e-10) < 0.10, "Scores should not differ much from the original"
#         else:
#             assert diff / (rr_mean + 1e-10) > 0.28, "Scores should differ much from the original"


####################################################################################################
################################# AUXILIARY TESTS ##################################################
####################################################################################################
@pytest.mark.parametrize(
    "scaler_tag, embed_tag, scaler_params, embed_params",
    [
        ("identity", "identity", {}, {}),
        ("standard", "umap", {}, {}),
        ("robust", "pca", {}, {}),
        ("minmax", "lle", {}, {}),
        ("maxabs", "isomap", {}, {}),
        ("identity", "spacy", {}, {}),
        ("identity", "bert", {}, {}),
        ("identity", "identity", {"dummy": 0}, {"dummy": 0}),
        ("standard", "umap", {"dummy": 0}, {"dummy": 0}),
        ("robust", "pca", {"dummy": 0}, {"dummy": 0}),
        ("minmax", "lle", {"dummy": 0}, {"dummy": 0}),
        ("maxabs", "isomap", {"dummy": 0}, {"dummy": 0}),
        ("identity", "spacy", {"dummy": 0}, {}),
        ("identity", "bert", {"dummy": 0}, {"model_name": "none"}),
    ],
)
def test_utils_transformers(scaler_tag: str, embed_tag: str, scaler_params: dict, embed_params: dict, **kwargs):
    # column names
    col_names = ["Age", "Gender", "Country"]

    # artificial data
    data = np.column_stack(
        [
            np.random.default_rng(42).uniform(18, 90, size=(300, 1)),
            np.random.default_rng(42).choice([0, 1], p=[0.5, 0.5], size=(300, 1)),
            np.random.default_rng(42).choice([0, 1, 2, 3, 4, 5], p=[1 / 6] * 6, size=(300, 1)),
        ],
    )

    # get normalizer
    scaler_obj = scale.GlobalNorm(scaler_tag, **scaler_params)

    # get embedding encoder
    embed_obj = embed.GlobalEmbedder(embed_tag, **embed_params)

    # get normalised data
    data_s = scaler_obj.fit_transform(data)
    data_s = scaler_obj.transform(data_s)

    # get embeddings
    data_emb0 = embed_obj.fit_transform(data, column_names=col_names)
    data_emb0 = embed_obj.transform(data, column_names=col_names)
    _ = embed_obj.inverse_transform(data, columns_names=col_names)

    data_emb1 = embed_obj.fit_transform(data_s, column_names=col_names)
    data_emb1 = embed_obj.transform(data_s, column_names=col_names)
    _ = embed_obj.inverse_transform(data, columns_names=col_names)

    # load global dataset instance
    data_df = pd.DataFrame(data, columns=col_names)

    dataset = load.TabularDataset(
        file_path=None,
        data=data_df,
        scaler=scaler_tag,
        embed=embed_tag,
        scaler_kwargs=scaler_params,
        embed_kwargs=embed_params,
        meta=None,
    )

    # embeddings decode
    _ = dataset.embed_decode(dataset.data_emb)

    # inverse transform
    _ = dataset.inverse_transform(dataset.data_s)

    # reset metadata
    dataset.meta.reset_params()

    # checkpoints
    # 1
    assert data.shape == data_s.shape, "Normalized sets should have the same shape as the original"

    # 2
    assert data_emb0.shape == data_emb1.shape, "Distinct embeddings should have the same shape"

    # 3
    assert len(data) == len(data_s) == len(data_emb0) == len(data_emb1), "The number of samples should remain the same"


@pytest.mark.parametrize(
    "with_fpath, with_onehot, scaler_tag, embed_tag, scaler_params, embed_params, qi_cols",
    [
        (False, True, "identity", "identity", {}, {}, ["Age"]),
        (False, False, "standard", "umap", {}, {}, ["Gender"]),
        (False, True, "robust", "pca", {}, {}, ["Age", "Country"]),
        (False, False, "minmax", "lle", {}, {}, None),
        (False, True, "maxabs", "isomap", {}, {}, None),
        (True, False, "identity", "identity", {}, {}, None),
        (True, True, "standard", "umap", {}, {}, None),
        (True, False, "robust", "pca", {}, {}, None),
        (True, True, "minmax", "lle", {}, {}, None),
        (True, False, "maxabs", "isomap", {}, {}, None),
        (True, True, "identity", "identity", {}, {}, None),
        (True, False, "standard", "umap", {}, {}, None),
        (True, True, "robust", "pca", {}, {}, None),
        (True, False, "minmax", "lle", {}, {}, None),
        (True, True, "maxabs", "isomap", {}, {}, None),
        (False, False, "identity", "identity", {}, {}, ["Gender", "Country"]),
        (False, True, "standard", "umap", {}, {}, ["Country"]),
        (False, False, "robust", "pca", {}, {}, ["Gender", "Age"]),
        (False, True, "minmax", "lle", {}, {}, None),
        (False, False, "maxabs", "isomap", {}, {}, None),
    ],
)
def test_utils_load_dataset(
    with_fpath: bool,
    with_onehot: bool,
    scaler_tag: str,
    embed_tag: str,
    scaler_params: dict,
    embed_params: dict,
    qi_cols: List[str],
    **kwargs,
):
    # filepath
    if with_fpath:
        fpath = join(cfg.data_test_dir, "tabular", "input_val", "dataset", "dummy_ref_tab.json")
    else:
        fpath = None

    # column names
    col_names = ["Age", "Gender", "Country"]

    # artificial data
    data = np.column_stack(
        [
            np.random.default_rng(42).uniform(18, 90, size=(300, 1)),
            np.random.default_rng(42).choice([0, 1], p=[0.5, 0.5], size=(300, 1)),
            np.random.default_rng(42).choice([0, 1, 2, 3, 4, 5], p=[1 / 6] * 6, size=(300, 1)),
        ],
    )

    # get dataframe
    data_df = pd.DataFrame(data, columns=col_names) if not with_fpath else None

    # load dataset
    dataset = load.TabularDataset(
        file_path=fpath,
        data=data_df,
        qi_names=qi_cols,
        sens_names=qi_cols,
        scaler=scaler_tag,
        embed=embed_tag,
        scaler_kwargs=scaler_params,
        embed_kwargs=embed_params,
        meta=None,
        with_onehot=with_onehot,
    )

    # embeddings decode
    emb_inv = dataset.embed_decode(dataset.data_emb)

    # inverse transform
    data_inv = dataset.inverse_transform(dataset.data_s)

    # reset metadata
    dataset.meta.reset_params()

    # embedder has inverse tranform
    has_inv = dataset.embed.has_inv

    # checkpoints
    # 1
    if has_inv:
        assert (
            emb_inv.shape == dataset.data_s.shape
        ), "Original data and reversely transformed embeddings should have the same shape."
    else:
        assert (
            emb_inv.shape == dataset.data_emb.shape
        ), "Embedder does not have inverse transform. Original and Inverse embeddings should have the same shape."

    assert len(emb_inv) == len(dataset.data), "Both sets should have the same no. samples."

    # 2
    assert (
        data_inv.shape == dataset.data.shape
    ), "Original and reversely transformed datasets should have the same shape."
    assert len(data_inv) == len(dataset.data), "Both sets should have the same no. samples."

    # 3
    if with_onehot:
        assert (
            dataset.data_s.shape[-1] >= dataset.data.shape[-1]
        ), "Transformed number of columns should be equal/greater than the original due to OneHotEncoding."
    else:
        assert (
            dataset.data_s.shape == dataset.data.shape
        ), "Original and transformed datasets should have the same shape."
