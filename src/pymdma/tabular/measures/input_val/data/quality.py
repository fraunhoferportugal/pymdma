from typing import Dict, List, Optional

import numpy as np

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ....data.utils import is_categorical
from ...utils_inp import (  # proximity_score,
    compute_vif,
    corr_matrix,
    corr_strong,
    iqr_outliers,
    uniformity_score_per_column,
    z_score_outliers,
)


class CorrelationScore(Metric):
    """Computes linear correlations between attributes in a dataset and returns
    the average percentage of attributes that are moderately or strongly
    correlated with each attribute.

    **Objective**: Correlation

    Parameters
    ----------
    column_names : list of str, optional, default=None
        List of column names corresponding to the attributes in the dataset.
    correlation_thresh : float, optional, default=0.5
        The correlation threshold to consider an attribute as moderately or strongly correlated.
        Defaults to 0.5.
    **kwargs : dict
        Additional keyword arguments for compatibility or future use.

    References
    ----------
    Shrestha, Detecting multicollinearity in regression analysis (2020).
    http://pubs.sciepub.com/ajams/8/2/1

    Returns
    -------
    MetricResult
        A MetricResult object containing the percentage of columns correlated with each other, and global summary statistics.

    Examples
    --------
    >>> # Example 1: Initializing and computing correlation on random data
    >>> import numpy as np
    >>> column_names = [f'col_{i}' for i in range(10)]
    >>> data = np.random.rand(100, 10)
    >>> correlation_score = CorrelationScore(column_names=column_names)
    >>> result: MetricResult = correlation_score.compute(data)
    >>> dataset_level, _ = result.value # Percentage of correlated attributes
    >>> dataset_stats, _ = result.stats # Mean and std of correlation percentages

    >>> # Example 2: Specifying a different correlation threshold
    >>> correlation_score = CorrelationScore(column_names=column_names, correlation_thresh=0.7)
    >>> result: MetricResult = correlation_score.compute(data)
    >>> dataset_level, _ = result.value # Percentage of correlated attributes
    >>> dataset_stats, _ = result.stats # Mean and std of correlation percentages
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        column_names: Optional[List[str]] = None,
        correlation_thresh: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c_thresh = correlation_thresh
        self.column_names = column_names

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the correlation matrix for the input data and determines
        the percentage of attributes that are moderately or strongly correlated
        with each attribute.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which the correlation matrix will be computed.
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the percentage of correlated attributes for each
            attribute and statistics (mean and standard deviation) of the correlations.
        """

        # correlation matrix
        corr_m = corr_matrix(
            data=data,
            **kwargs,
        )

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(data.shape[-1])]
        )

        # Moderate/highly correlated attributes per attribute
        stats_d = corr_strong(
            corr_matrix=corr_m,
            cols=cols,
            c_thresh=self.c_thresh,
            **kwargs,
        )

        # global score
        perc_corr = {col: round(100 * len(att) / (len(self.column_names) - 1), 1) for col, att in stats_d.items()}

        # stats
        perc_stats = {
            "mean": np.mean(list(perc_corr.values())),
            "std": np.std(list(perc_corr.values())),
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": perc_corr,
                "stats": perc_stats,
            },
        )


class UniquenessScore(Metric):
    """Computes the percentage of duplicate records in a dataset, providing a
    measure of the dataset's uniqueness.

    The uniqueness score is calculated by determining the proportion of duplicate rows in the dataset. A higher
    percentage indicates more duplicates, while a lower percentage indicates higher uniqueness.

    **Objective**: Uniqueness

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments for compatibility or future use.

    References
    ----------
    Sukhobok, Tabular data anomaly patterns (2017).
    https://ieeexplore.ieee.org/document/8316296

    Returns
    -------
    MetricResult
        A MetricResult object containing the uniformity score for each column and summary statistics.

    Examples
    --------
    >>> # Example 1: Computing uniqueness score on a dataset with no duplicates
    >>> import numpy as np
    >>> data = np.random.rand(100, 5)  # Random dataset (no duplicates)
    >>> uniqueness_score = UniquenessScore()
    >>> result: MetricResult = uniqueness_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: 0.0 (no duplicates)

    >>> # Example 2: Computing uniqueness score on a dataset with duplicates
    >>> data_with_dupl = np.concatenate([data, data[:10]])  # Add 10 duplicate rows
    >>> result: MetricResult = uniqueness_score.compute(data_with_dupl)
    >>> dataset_level, _ = result.value  # Output: Percentage of duplicate rows
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the percentage of duplicate records in a dataset, providing
        a measure of the dataset's uniqueness.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which the uniqueness score will be computed.
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the percentage of duplicate rows in the dataset.
        """

        # total number of samples
        n_total = len(data)

        # number of non-duplicates
        _, cnt = np.unique(
            data,
            axis=0,
            return_counts=True,
        )
        n_dupl = cnt[cnt > 1].sum()

        # percentage of duplicates
        p_dupl = 100 * n_dupl / n_total

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": p_dupl,
            },
        )


class UniformityScore(Metric):
    """Computes a uniformity score for each attribute in the dataset,
    evaluating both discrete and continuous columns.

    For each column, the score assesses its uniformity, entropy, and imbalance level, which can be aggregated to
    provide insights into the overall distribution of values. Discrete columns are scored based on categories,
    while continuous columns are assessed based on the spread of values.

    **Objective**: Uniformity

    Parameters
    ----------
    column_names : list of str, optional, default=None
        List of column names in the dataset for which the uniformity score will be computed, by default None.
    col_map : dict of str, optional, default=None
        Dictionary mapping each column name to its data type information, including whether it's continuous or discrete,
        by default None.
    **kwargs : dict
        Additional keyword arguments for compatibility or future use.

    Returns
    -------
    MetricResult
        A MetricResult object containing the uniformity score for each column and summary statistics.

    Examples
    --------
    >>> # Example 1: Computing uniformity score on a dataset with random data
    >>> import numpy as np
    >>> column_names = ['A', 'B', 'C']
    >>> col_map = {'A': {'type': {'tag': 'discrete', 'opt': [1, 2, 3]}},
    ...            'B': {'type': {'tag': 'discrete', 'opt': [0, 1]}},
    ...            'C': {'type': {'tag': 'continuous'}}}
    >>> data = np.random.rand(100, 3)
    >>> uniformity_score = UniformityScore(column_names=column_names, col_map=col_map)
    >>> result: MetricResult = uniformity_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: Uniformity scores per column

    >>> # Example 2: Computing uniformity score on a dataset with predefined categories
    >>> data_with_categories = np.array([[1, 0, 3.2], [1, 1, 2.5], [2, 0, 4.1]])
    >>> result: MetricResult = uniformity_score.compute(data_with_categories)
    >>> dataset_level, _ = result.value  # Output: Uniformity scores per column
    >>> dataset_stats, _ = result.stats  # Output: Mean and standard deviation of imbalance levels
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        column_names: Optional[List[str]] = None,
        col_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.col_map = col_map

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the uniformity score for each column in the dataset.

        For discrete columns, the function calculates how uniformly the categories are distributed.
        For continuous columns, it assesses the spread of values. The results are returned for each column,
        including the overall mean and standard deviation of imbalance levels.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which the uniformity score will be computed.
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the uniformity scores for each column and summary statistics.
        """

        # score dictionary
        score_d = {}

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(data.shape[-1])]
        )

        # column map
        col_map_exists = isinstance(self.col_map, dict)

        # loop over columns
        for idx, col in enumerate(cols):
            # type and number of tag (discrete/continuous)
            if col_map_exists:
                # get column key
                vtype = self.col_map.get(col, {}).get("type", {})
                vtag = vtype.get("tag")
                vnum = len(vtype.get("opt"))
            else:
                vtag = "discrete" if is_categorical(data) else "continuous"
                vnum = None

            # compute scores
            if vtag == "discrete":
                stat, ent, imb = uniformity_score_per_column(
                    data_col=data[:, idx],
                    is_continuous=False,
                    n_categories=vnum,
                )
            elif vtag == "continuous":
                stat, ent, imb = uniformity_score_per_column(
                    data_col=data[:, idx],
                    is_continuous=True,
                )
            else:
                stat, ent, imb = [None] * 4

            # assign to dict
            score_d[col] = {
                "stat_score": stat,
                "entropy_score": ent,
                "level_score": imb,
            }

            # aggregated uniformity score per column
            imb_d = {col: val.get("level_score") for col, val in score_d.items()}

            # imbalance values
            imb_v = list(imb_d.values())

            # stats
            imb_stats = {
                "mean": np.mean(imb_v),
                "std": np.std(imb_v),
            }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": imb_d,
                "stats": imb_stats,
            },
        )


class OutlierScore(Metric):
    """Computes the percentage of outliers in each column of a dataset.

    For each column, the function detects outliers using both z-score and interquartile range (IQR) methods,
    calculates the percentage of outliers, and averages the results of both methods. It also computes summary
    statistics (mean and standard deviation) of the outlier percentages across all columns.

    **Objective**: Out of Distribution Detection

    Parameters
    ----------
    column_names : list of str, optional, default=None
        List of column names in the dataset, by default None.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Iglewicz, B. and Hoaglin, D. (1993) The ASQC Basic References in Quality Control: Statistical Techniques. 
    In: Mykytka, E.F., Eds., How to Detect and Handle Outliers, ASQC Quality Press, Milwaukee, Vol. 16

    Returns
    -------
    MetricResult
        A MetricResult object containing the outlier percentage for each column and summary statistics.

    Examples
    --------
    >>> # Example 1: Computing outlier score on a random dataset
    >>> import numpy as np
    >>> column_names = ['A', 'B', 'C']
    >>> data = np.random.rand(100, 3)  # Random dataset of 100 samples and 3 columns
    >>> outlier_score = OutlierScore(column_names=column_names)
    >>> result: MetricResult = outlier_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: Percentage of outliers per column

    >>> # Example 2: Computing outlier score on a dataset with some extreme values
    >>> data_with_outliers = np.array([[1, 2, 3], [4, 5, 1000], [6, 7, 8]])  # Column 'C' contains an outlier
    >>> result: MetricResult = outlier_score.compute(data_with_outliers)
    >>> dataset_level, _ = result.value  # Output: Percentage of outliers per column
    >>> dataset_stats, _ = result.stats  # Output: Mean and standard deviation of outlier percentages
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(self, column_names: Optional[List[str]] = None, **kwargs):
        """Initializes the OutlierScore metric with the column names.

        Parameters
        ----------
        column_names : list of str, optional
            List of column names for which outlier detection will be performed.
        **kwargs : dict
            Additional keyword arguments passed to the parent class.
        """

        super().__init__(**kwargs)

        # column list
        self.column_names = column_names

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the percentage of outliers for each column in the dataset.

        For each column, it calculates the outlier percentage using both z-score and interquartile range (IQR) methods,
        and then averages the results. The summary statistics (mean and standard deviation) of the outlier percentages
        are also computed.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which the outlier percentage will be computed. Rows with NaN values are excluded.
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the outlier percentages for each column and summary statistics.
        """

        # curate data
        data_ = data[~np.isnan(data).sum(axis=1, dtype=bool)]

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(data.shape[-1])]
        )

        perc_out_d = {}
        for idx, col in enumerate(cols):
            # number of samples
            n_samples = len(data_[:, idx])

            # statistical outliers
            z_score = z_score_outliers(data_[:, idx])
            iqr_score = iqr_outliers(data_[:, idx])

            # percentage of outliers
            perc_z = 100 * z_score / n_samples
            perc_iqr = 100 * iqr_score / n_samples

            # average both
            perc_outliers = np.mean([perc_z, perc_iqr])

            # assign
            perc_out_d[col] = perc_outliers

        # statistical aggregates
        out_vals = list(perc_out_d.values())

        perc_out_stats = {
            "mean": np.mean(out_vals),
            "std": np.std(out_vals),
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": perc_out_d,
                "stats": perc_out_stats,
            },
        )


class MissingScore(Metric):
    """Computes the percentage of missing values per column in the dataset and
    provides summary statistics for missing rates across samples and columns.

    **Objective**: Missing Values

    Parameters
    ----------
    column_names : list of str, optional, default=None
        List of column names in the dataset, by default None.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Taleb et al., Big data quality: A quality dimensions evaluation (2016).
    https://ieeexplore.ieee.org/document/7816918

    Returns
    -------
    MetricResult
        A MetricResult object containing the missing value percentage for each column and summary statistics.

    Examples
    --------
    >>> # Example 1: Computing missing values score on a random dataset
    >>> import numpy as np
    >>> column_names = ['A', 'B', 'C']
    >>> data = np.random.rand(100, 3)  # Random dataset of 100 samples and 3 columns
    >>> data[0, 0] = np.nan  # Introducing a missing value
    >>> missing_score = MissingScore(column_names=column_names)
    >>> result: MetricResults = missing_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: Percentage of missing values per column

    >>> # Example 2: Computing missing score on a dataset with significant missing values
    >>> data_with_missing = np.array([[1, np.nan, 3], [np.nan, 5, np.nan], [6, 7, 8]])  # Missing values in multiple columns
    >>> result: MetricResults = missing_score.compute(data_with_missing)
    >>> dataset_level, _ = result.value  # Output: Percentage of missing values per column
    >>> dataset_stats, _ = result.stats  # Output: Mean missing rate for samples and columns
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(self, column_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the percentage of missing values for each column or row in
        the dataset.

        It calculates the percentage of missing values across rows and columns.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which the missing value percentage will be computed.
            The data should be a 2D array where NaN values represent missing entries.
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the missing value percentages for each column and summary statistics.
        """

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(data.shape[-1])]
        )

        # missing values
        miss_samp = np.round(100 * np.isnan(data).sum(axis=1) / data.shape[1], 3)
        miss_col = np.round(100 * np.isnan(data).sum(axis=0) / data.shape[0], 3)

        # missing rates per column
        miss_col_d = {col: miss for col, miss in zip(cols, miss_col)}

        # aggregated missing rates
        miss_agg_d = {
            "sample": miss_samp.mean(),
            "column": miss_col.mean(),
        }

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_VAL,
                "subtype": "float",
                "value": miss_col_d,
                "stats": miss_agg_d,
            },
        )


class DimCurseScore(Metric):
    """Computes the ratio of the number of columns (features) to the number of
    samples (instances) in the dataset to evaluate the curse of dimensionality.
    A higher ratio indicates that the dataset may suffer from high
    dimensionality relative to the number of samples.

    **Objective**: Dimensionality

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Returns
    -------
    MetricResult
        A MetricResult object containing the ratio of columns to samples.

    Examples
    --------
    >>> # Example 1: Evaluating dimensionality on a dataset with more samples than features
    >>> import numpy as np
    >>> data = np.random.rand(100, 10)  # 100 samples, 10 columns
    >>> dim_curse_score = DimCurseScore()
    >>> result: MetricResult = dim_curse_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: 0.1 (indicating more samples than features)

    >>> # Example 2: Evaluating dimensionality on a dataset with more features than samples
    >>> data = np.random.rand(10, 100)  # 10 samples, 100 columns
    >>> result: MetricResult = dim_curse_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: 10.0 (indicating more features than samples)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the ratio of the number of columns (features) to the number
        of rows (samples) to assess the dataset's susceptibility to the curse
        of dimensionality.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which the dimensionality ratio will be computed.
            The data should be a 2D array where the shape is (samples, columns).
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the computed ratio of columns to samples.
        """

        # columns vs samples ratio
        ratio = np.divide(*data.shape)

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": ratio,
            },
        )


class VIFactorScore(Metric):
    """Calculates the Variance Inflation Factor (VIF) to assess the
    multicollinearity of each attribute (feature) in the dataset. VIF measures
    how much the variance of an estimated regression coefficient increases if
    your predictors are correlated.

    **Objective**: Multicollinearity

    Parameters
    ----------
    column_names : list of str, optional, default=None
        List of the names of the columns (features) in the dataset.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Marcoulides and Raykov, Evaluation of variance inflation factors in regression models using latent variable modeling methods (2019).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6713981/

    Returns
    -------
    MetricResult
        A MetricResult object containing the variance inflation factor (VIF) for each attribute.

    Examples
    --------
    >>> # Example 1: Evaluating VIF on a dataset with low multicollinearity
    >>> import numpy as np
    >>> data = np.random.rand(100, 5)  # 100 samples, 5 features
    >>> vif_score = VIFactorScore(column_names=["col1", "col2", "col3", "col4", "col5"])
    >>> result: MetricResult = vif_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: VIF scores per column

    >>> # Example 2: Evaluating VIF on a dataset with high multicollinearity
    >>> data = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])  # Multicollinear features
    >>> vif_score = VIFactorScore(column_names=["A", "B", "C"])
    >>> result: MetricResult = vif_score.compute(data)
    >>> dataset_level, _ = result.value  # Output: Very high VIF scores for multicollinear columns
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = np.inf

    def __init__(self, column_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the Variance Inflation Factor (VIF) for each feature in the
        dataset to assess multicollinearity.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which VIF will be computed. The dataset should be in array-like
            format with the shape (samples, features).
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the VIF scores for each feature.
        """

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(data.shape[-1])]
        )

        # compute VIF for each column
        vif_p, _ = compute_vif(
            data=data,
            column_names=cols,
            **kwargs,
        )

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "int",
                "value": vif_p,
            },
        )


__all__ = [
    "CorrelationScore",
    "UniquenessScore",
    "UniformityScore",
    "OutlierScore",
    "MissingScore",
    "DimCurseScore",
    "VIFactorScore",
]
