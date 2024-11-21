# from typing import List, Optional

# import numpy as np

# from pymdma.common.definitions import Metric
# from pymdma.common.output import MetricResult
# from pymdma.constants import EvaluationLevel, OutputsTypes, ReferenceType, SyntheticFeatureMetrics

# from ...utils_syn import compute_utility_scores


# class Utility(Metric):
#     """Computes the utility evaluation scores for synthetic datasets. It
#     calculates the percentage change of performance scores presented as [1] RR
#     (Real vs Real), [2] SR (Synthetic vs Real), and [3] SRR (Synthetic and Real
#     vs Real).

#     Parameters
#     ----------
#     column_names : list, optional
#         The names of the columns to be considered for the utility calculation.
#     target_col : str, optional
#         The name of the target column to predict.
#     train_cols : list, optional
#         The list of columns to be used for training the model.

#     Returns
#     -------
#     MetricResult
#         A MetricResult object containing the utility scores.

#     Examples
#     --------
#     >>> # Example 1: Evaluating utility scores for a dataset
#     >>> import numpy as np
#     >>> real_data = np.concatenate((
#     ...     np.random.rand(100, 3),
#     ...     np.random.choice([0, 1], 100, p=[0.5, 0.5]).reshape(-1, 1)),
#     ...     axis=-1
#     ... )
#     >>> syn_data = np.concatenate((
#     ...     np.random.rand(100, 3),
#     ...     np.random.choice([0, 1], 100, p=[0.5, 0.5]).reshape(-1, 1)),
#     ...     axis=-1
#     ... )
#     >>> utility_score = Utility(column_names=['att1', 'att2', 'att3', 'target'], target_col='target')
#     >>> result: MetricResult = utility_score.compute(real_data, syn_data)
#     >>> dataset_level, _ = result.value # Output: utility scores for RR, SR, and SRR

#     >>> # Example 2: Using specific training columns
#     >>> utility_score_specific = Utility(column_names=['att1', 'att2', 'att3', 'target'],
#     ...                                    target_col='target', train_cols=['att1', 'att2'])
#     >>> result_specific: MetricResult = utility_score_specific.compute(real_data, syn_data)
#     >>> dataset_level, _ = result_specific.value # Output: utitility scores with specified training columns
#     """

#     reference_type = ReferenceType.DATASET
#     evaluation_level = EvaluationLevel.DATASET
#     metric_group = SyntheticFeatureMetrics.UTILITY

#     higher_is_better: bool = True
#     min_value: float = 0
#     max_value: float = 100.0

#     def __init__(
#         self,
#         column_names: Optional[List[str]] = None,
#         target_col: Optional[str] = None,
#         train_cols: Optional[List[str]] = None,
#         **kwargs,
#     ):
#         """Initializes the Utility metric to evaluate the utility scores for
#         synthetic datasets.

#         Parameters
#         ----------
#         column_names : list, optional
#             The names of the columns to be considered for the utility calculation.
#         target_col : str, optional
#             The name of the target column to predict.
#         train_cols : list, optional
#             The list of columns to be used for training the model.
#         **kwargs : dict
#             Additional keyword arguments passed to the parent class.
#         """

#         super().__init__(**kwargs)

#         # column list
#         self.column_names = column_names

#         # target column
#         self.target_col = target_col

#         # train columns
#         self.train_cols = train_cols

#     def compute(self, real_data: np.ndarray, syn_data: np.ndarray, **kwargs) -> MetricResult:
#         """Computes the utility evaluation scores comparing synthetic and real
#         datasets.

#         Parameters
#         ----------
#         real_data : np.ndarray
#             The target dataset for evaluation.
#         syn_data : np.ndarray
#             The synthetic dataset to evaluate.
#         **kwargs : dict
#             Additional keyword arguments for computation.

#         Returns
#         -------
#         MetricResult
#             A MetricResult object containing the utility scores.
#         """

#         kind = ["rr", "sr", "srr"]

#         # define columns
#         if isinstance(self.column_names, (list, np.ndarray)):
#             cols_ = self.column_names
#         else:
#             cols_ = [f"att{x}" for x in range(real_data.shape[-1])]

#         # target column
#         targ_ = cols_[-1] if self.target_col is None else self.target_col

#         # train columns
#         train_ = cols_[:-1] if self.train_cols is None else self.train_cols

#         # perform utility using ML
#         score_map = compute_utility_scores(
#             real_data=real_data,
#             syn_data=syn_data,
#             cols=cols_,
#             target_col=targ_,
#             train_cols=train_,
#             kind=kind,
#         )

#         # get utility metric names
#         # metric_names = list(score_map["rr"].columns)

#         # build final score dict
#         score_final_d = {
#             # 'metric': metric_names,
#             "RR": score_map["rr"].round(3).loc[0].to_list(),
#             "SR": score_map["sr"].round(3).loc[0].to_list(),
#             "SRR": score_map["srr"].round(3).loc[0].to_list(),
#         }

#         return MetricResult(
#             dataset_level={
#                 "dtype": OutputsTypes.KEY_ARRAY,
#                 "subtype": "float",
#                 "value": score_final_d,
#             },
#         )


# __all__ = ["Utility"]
