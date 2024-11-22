from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ....data.utils import is_categorical
from ...utils_syn import _get_js_divergence, _get_kl_divergence, _get_ks_similarity, _get_nn_pdf, _get_tv_similarity


class StatisticalSimScore(Metric):
    """Computes a dataset-level statistical similarity score between real and
    synthetic data.

    This metric assesses how closely the statistical properties of the synthetic dataset
    resemble those of the real dataset, providing a fidelity measure for synthetic data generation.

    **Objective**: Fidelity

    Parameters
    ----------
    col_map : dict, optional, default=None
        A mapping of column names to their types and properties. This is used to determine
        how to compute similarity for each column.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Yang et al., Structured evaluation of synthetic tabular data (2024).
    https://arxiv.org/abs/2403.10424

    Returns
    -------
    MetricResult
        A MetricResult object containing the similarity scores and their statistics.

    Examples
    --------
    >>> # Example 1: Evaluating statistical similarity for a dataset with discrete and continuous variables
    >>> import numpy as np
    >>> real_data = np.array([
    ...     [1, 2.5],
    ...     [1, 3.0],
    ...     [2, 3.5],
    ...     [2, 4.0]
    ... ])
    >>> syn_data = np.array([
    ...     [1, 2.6],
    ...     [1, 3.1],
    ...     [2, 3.4],
    ...     [2, 4.2]
    ... ])
    >>> col_map = {
    ...     "column1": {"type": {"tag": "discrete"}},
    ...     "column2": {"type": {"tag": "continuous"}},
    ... }
    >>> sim_score = StatisticalSimScore(col_map=col_map)
    >>> result: MetricResult = sim_score.compute(real_data, syn_data)
    >>> dataset_level, _ = result.value # Output: similarity scores for each column

    >>> # Example 2: Evaluating similarity with mismatched column types
    >>> real_data = np.array([
    ...     [1, 2],
    ...     [2, 3],
    ...     [3, 4]
    ... ])
    >>> syn_data = np.array([
    ...     [1, 2],
    ...     [2, 3],
    ...     [3, 5]
    ... ])
    >>> col_map = {
    ...     "column1": {"type": {"tag": "discrete"}},
    ...     "column2": {"type": {"tag": "discrete"}},
    ... }
    >>> sim_score = StatisticalSimScore(col_map=col_map)
    >>> result: MetricResult = sim_score.compute(real_data, syn_data)
    >>> dataset_level, _ = result.value  # Output: similarity scores for each column
    >>> dataset_stats, _ = result.stats  # Output: mean and std of similarity scores
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        col_map: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.col_map = col_map

    def _stat_sim_1d(
        self,
        real_col: np.ndarray,
        syn_col: np.ndarray,
        kind: str = "discrete",
        **kwargs,
    ):
        """Computes an 1D statistical similarity.

        Parameters
        ----------
        real_col : np.ndarray
            The real data for the specific attribute.
        syn_col : np.ndarray
            The synthetic data for the specific attribute.
        kind : str
            The type of data ('discrete' or 'continuous').
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        float
            The computed similarity score for the attribute.
        """
        # variable type assignment
        kind_ = kind.lower() if kind.lower() in ["discrete", "continuous"] else "continuous"

        # mapper
        kind_mapper = {
            "discrete": _get_tv_similarity,
            "continuous": _get_ks_similarity,
        }

        # score
        sim_score = kind_mapper.get(kind_)(
            real_col,
            syn_col,
        )
        return sim_score

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the statistical similarity score between real and synthetic
        datasets.

        Parameters
        ----------
        real_data : np.ndarray
            The real dataset for comparison.
        syn_data : np.ndarray
            The synthetic dataset to evaluate.
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the similarity scores and their statistics.
        """

        # checkpoint
        assert real_data.shape[1] == syn_data.shape[1], "Mismatched columns. Please fix before computing metrics."

        # column map
        col_map_exists = isinstance(self.col_map, dict)
        cols = self.col_map.keys() if col_map_exists else [f"att_{idx}" for idx in range(real_data.shape[1])]

        # similarity map
        sim_score = {}

        # column similarity
        for idx, col in enumerate(cols):
            # continuous OR discrete
            if col_map_exists:
                # dtype
                vtype = self.col_map.get(col).get("type").get("tag")
            else:
                # dtype
                vtype = "discrete" if is_categorical(real_data[:, idx]) else "continuous"

            # compute similarity
            sim_ = self._stat_sim_1d(
                real_data[:, idx],
                syn_data[:, idx],
                kind=vtype,
            )

            # assign
            sim_score[col] = sim_

        # global scores
        global_d = {
            "mean": np.mean(list(sim_score.values())).round(2),
            "std": np.std(list(sim_score.values())).round(2),
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": sim_score,
                "stats": global_d,
            },
        )


class StatisiticalDivergenceScore(Metric):
    """Computes a statistical divergence score for each column, specifically
    the Jensen-Shannon (JS) and Kullback-Leibler (KL) divergence scores.

    **Objective**: Fidelity

    Parameters
    ----------
    column_names : list of str, optional, default=None
        List of the names of the columns (features) in the dataset.
    score : str, optional, default='kl'
        Specifies the divergence score to compute ('js' for Jensen-Shannon, 'kl' for Kullback-Leibler, 'all' for both).
        By default, it is set to 'kl'.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Fonseca and Bacao,  Tabular and latent space synthetic data generation: a literature review (2023).
    https://doi.org/10.1186/s40537-023-00792-7

    Returns
    -------
    MetricResult
        A MetricResult object containing the divergence scores and their statistics.

    Examples
    --------
    >>> # Example 1: Evaluating statistical divergence for a dataset
    >>> import numpy as np
    >>> real_data = np.array([
    ...     [1, 2, 3],
    ...     [2, 3, 4],
    ...     [3, 4, 5]
    ... ])
    >>> syn_data = np.array([
    ...     [1, 2, 2],
    ...     [2, 2, 3],
    ...     [3, 3, 4]
    ... ])
    >>> col_map = {
    ...     "column1": {"type": {"tag": "continuous"}},
    ...     "column2": {"type": {"tag": "continuous"}},
    ...     "column3": {"type": {"tag": "continuous"}},
    ... }
    >>> divergence_score = StatisticalDivergenceScore(col_map=col_map, score='kl')
    >>> result: MetricResult = divergence_score.compute(real_data, syn_data)
    >>> dataset_level, _ = result.value  # Output: divergence scores for each column

    >>> # Example 2: Using JS divergence instead of KL
    >>> divergence_score_js = StatisticalDivergenceScore(col_map=col_map, score='js')
    >>> result_js: MetricResult = divergence_score_js.compute(real_data, syn_data)
    >>> dataset_level_js, _ = result_js.value  # Output: JS divergence scores for each column
    >>> dataset_stats_js, _ = result_js.stats  # Output: mean and std of JS divergence scores
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = -np.inf
    max_value: float = np.inf

    def __init__(
        self,
        column_names: Optional[List[str]] = None,
        score: Literal["js", "kl", "all"] = "kl",
        **kwargs,
    ):
        """Initializes the StatisticalDivergenceScore metric to evaluate the
        divergence between real and synthetic datasets based on defined column
        characteristics.

        Parameters
        ----------
        column_names : list of str, optional, default=None
            List of the names of the columns (features) in the dataset.
        score : str, optional, default='kl'
            Specifies the divergence score to compute ('js' for Jensen-Shannon,
            'kl' for Kullback-Leibler). Default is 'kl'.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """

        super().__init__(**kwargs)

        # column names
        self.column_names = column_names

        # score type
        self.score = score

    def _diverg_score_1d(
        self,
        real_col: np.ndarray,
        syn_col: np.ndarray,
        score: Literal["js", "kl", "all"] = "all",
        **kwargs,
    ):
        """Computes a column-level statistical divergence.

        Parameters
        ----------
        real_col : np.ndarray
            The real data for the specific attribute.
        syn_col : np.ndarray
            The synthetic data for the specific attribute.
        score : str
            The type of divergence to compute ('js' or 'kl').
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        dict
            A dictionary containing the computed divergence scores.
        """

        # score map
        score_map = {"js": _get_js_divergence, "kl": _get_kl_divergence}

        # score tags
        score_tag = ["js", "kl"] if score.lower() == "all" else [score]

        # get probability distributions
        real_pdf, syn_pdf, bins = _get_nn_pdf(real_col, syn_col)

        # compute divergence scores
        div_score = {tag: score_map.get(tag)(real_pdf, syn_pdf) for tag in score_tag if tag in score_map.keys()}

        return div_score

    def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the statistical divergence score between real and synthetic
        datasets.

        Parameters
        ----------
        real_data : np.ndarray
            The real dataset for comparison.
        syn_data : np.ndarray
            The synthetic dataset to evaluate.
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the divergence scores and their statistics.
        """

        # checkpoint
        assert real_data.shape[1] == syn_data.shape[1], "Mismatched columns. Please fix before computing metrics."

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(real_data.shape[1])]
        )

        # divergence map
        div_score = {}

        # column-wise
        for idx, col in enumerate(cols):
            # compute scores
            sim_ = self._diverg_score_1d(
                real_data[:, idx],
                syn_data[:, idx],
                score=self.score,
            )

            # assign
            div_score[col] = list(sim_.values())

        # global scores
        # auxiliary score dataframe
        aux_df = pd.DataFrame.from_dict(div_score.values())

        # aggregates
        mean_g, std_g = aux_df.mean(0).to_dict(), aux_df.std(0).to_dict()

        # global
        glob_d = {
            f"{self.score}_mean": mean_g[0],
            f"{self.score}_std": std_g[0],
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_ARRAY,
                "subtype": "float",
                "value": div_score,
                "stats": glob_d,
            },
        )


class CoherenceScore(Metric):
    """Computes the coherence score between the correlation matrices of the
    target and synthetic datasets. A higher coherence score indicates better
    fidelity between the datasets in terms of their correlation structures.

    **Objective**: Fidelity

    Parameters
    ----------
    weights : np.ndarray, optional, default=None
        Weights for the correlations, allowing for weighted contributions
        to the coherence score. If None, uniform weights are applied.
    corr_type : str, optional, default='pearson'
        The type of correlation to compute ('pearson' by default).
        Other types like 'spearman' may be supported depending on the implementation.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Yang et al., Structured evaluation of synthetic tabular data (2024).
    https://arxiv.org/abs/2403.10424

    Returns
    -------
    MetricResult
        A MetricResult object containing the coherence score.

    Examples
    --------
    >>> # Example 1: Evaluating coherence score for a dataset
    >>> import numpy as np
    >>> real_data = np.array([
    ...     [1, 2, 3],
    ...     [2, 3, 4],
    ...     [3, 4, 5]
    ... ])
    >>> syn_data = np.array([
    ...     [1, 2, 3],
    ...     [1, 2, 3],
    ...     [3, 4, 5]
    ... ])
    >>> coherence_score = CoherenceScore(corr_type='pearson')
    >>> result: MetricResult = coherence_score.compute(real_data, syn_data)
    >>> dataset_level, _ = result.value  # Output: coherence score

    >>> # Example 2: Evaluating with custom weights
    >>> weights = np.array([0.5, 1.0, 1.5])  # Example weights
    >>> coherence_score_weighted = CoherenceScore(weights=weights, corr_type='spearman')
    >>> result_weighted: MetricResult = coherence_score_weighted.compute(real_data, syn_data)
    >>> dataset_level, _ = result_weighted.value  # Output: weighted coherence score
    """

    reference_type = ReferenceType.DATASET
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        corr_type: Optional[str] = "pearson",
        **kwargs,
    ):
        """Initializes the CoherenceScore metric to evaluate the coherence
        between the correlation matrices of real and synthetic datasets.

        Parameters
        ----------
        weights : np.ndarray, optional, default=None
            Weights for the correlations, allowing for weighted contributions
            to the coherence score. If None, uniform weights are applied.
        corr_type : str, optional, default=None
            The type of correlation to compute ('pearson' by default).
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """

        super().__init__(**kwargs)

        # correlation type
        self.corr = corr_type

        # weights array
        self.weights = weights

    def compute(
        self,
        real_data: np.ndarray,
        syn_data: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes the coherence score between the correlation matrices of
        real and synthetic datasets.

        Parameters
        ----------
        real_data : np.ndarray
            The real dataset for comparison.
        syn_data : np.ndarray
            The synthetic dataset to evaluate.
        **kwargs : dict
            Additional keyword arguments for computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the coherence score.
        """

        # compute correlation matrices
        real_corr = pd.DataFrame(real_data).corr(self.corr).replace(np.nan, 1).to_numpy()
        syn_corr = pd.DataFrame(syn_data).corr(self.corr).replace(np.nan, 1).to_numpy()

        # number columns
        n_cols = len(real_corr)

        # compute similarity between real and syn matrices
        delta_corr = np.abs(
            np.nan_to_num(real_corr) - np.nan_to_num(syn_corr),
        )  # differences

        # weight matrix
        id_mask = np.abs(np.identity(n_cols) - 1)

        if self.weights is not None:
            w_mask = np.array([self.weights] * n_cols) * id_mask
        else:
            w_mask = np.ones((n_cols, n_cols)) * id_mask

        # norm
        w_mask /= sum(w_mask)

        # correlation similarity (weighted avg.)
        corr_sim = np.sum(delta_corr * w_mask) / np.sum(w_mask)
        # ((n_cols * (n_cols - 1)))

        # average correlation
        avg_corr_coh = np.mean(np.round(1 - corr_sim / 2, 3))

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": avg_corr_coh,
            },
        )


__all__ = ["StatisticalSimScore", "StatisiticalDivergenceScore", "CoherenceScore"]
