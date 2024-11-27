from typing import List, Optional

import numpy as np

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

from ...utils_inp import compute_k_anonymity


class KAnonymityScore(Metric):
    """Calculates the k for k-anonymity. A higher k value indicates that each
    record is less unique, meaning it is more difficult to re-identify
    individuals within the dataset.

    **Objective**: Privacy

    Parameters
    ----------
    column_names : list
        List of the names of the columns (features) in the dataset.
    qi_names : list, optional, default=None
        List of the quasi-identifier column names.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    References
    ----------
    Díaz and García, A python library to check the level of anonymity of a dataset. (2022).
    http://dx.doi.org/10.1038/s41597-022-01894-2

    Returns
    -------
    MetricResult
        A MetricResult object containing the k-anonymity score.

    Examples
    --------
    >>> # Example 1: Evaluating k-anonymity on a dataset with sufficient quasi-identifiers
    >>> import numpy as np
    >>> data = np.array([
    ...     ['Alice', 'Smith', 'NY'],
    ...     ['Alice', 'Smith', 'NY'],
    ...     ['Bob', 'Jones', 'CA'],
    ...     ['Bob', 'Jones', 'CA']
    ... ])
    >>> k_anonymity = KAnonymityScore(
    ...     column_names=['first_name', 'last_name', 'state'],
    ...     qi_names=['first_name', 'last_name']
    ... )
    >>> result = k_anonymity.compute(data)
    >>> dataset_level, _ = result.value # Output: k-anonymity score

    >>> # Example 2: Evaluating k-anonymity on a dataset with low uniqueness
    >>> data = np.array([
    ...     ['Alice', 'Smith', 'NY'],
    ...     ['Alice', 'Smith', 'CA'],
    ...     ['Bob', 'Jones', 'NY']
    ... ])
    >>> k_anonymity = KAnonymityScore(
    ...     column_names=['first_name', 'last_name', 'state'],
    ...     qi_names=['first_name', 'last_name']
    ... )
    >>> result: MetricResult = k_anonymity.compute(data)
    >>> dataset_level, _ = result.value # Output: k-anonymity score
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.DATASET
    metric_group = MetricGroup.PRIVACY

    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 100.0

    def __init__(self, column_names: Optional[List[str]] = None, qi_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.qi_names = qi_names

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """Computes the k-anonymity score for the dataset based on the
        specified quasi-identifiers.

        Parameters
        ----------
        data : np.ndarray
            The input dataset for which k-anonymity will be computed. The dataset
            should be in array-like format with the shape (samples, features).
        **kwargs : dict
            Additional keyword arguments for controlling the computation.

        Returns
        -------
        MetricResult
            A MetricResult object containing the k-anonymity score.
        """

        # columns
        cols = (
            self.column_names
            if isinstance(self.column_names, list)
            else [f"att_{idx}" for idx in range(data.shape[-1])]
        )

        # qi names
        qi_names = self.qi_names if isinstance(self.qi_names, list) else cols[-1:]

        # k anonimity
        k_anom = compute_k_anonymity(
            data=data,
            column_names=cols,
            qi_names=qi_names,
        )

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMERIC,
                "subtype": "float",
                "value": k_anom,
            },
        )


__all__ = ["KAnonymityScore"]
