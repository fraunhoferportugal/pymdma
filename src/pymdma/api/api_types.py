from typing import Dict, List, Optional, Union

from fastapi import Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Annotated

from ..common.output import EvalLevelOutput
from ..constants import (
    DataModalities,
    EvaluationLevel,
    InputAnnotationMetrics,
    InputMetricGroups,
    InputPrivacyMetrics,
    InputQualityMetrics,
    ReferenceType,
    SyntheticFeatureMetrics,
    SyntheticMetricGroups,
    ValidationTypes,
    valid_subclass,
)

_METRIC_GOALS = Union[
    InputQualityMetrics,
    InputPrivacyMetrics,
    SyntheticFeatureMetrics,
    InputAnnotationMetrics,
    None,
]

_METRIC_GOAL_TYPES = Union[InputQualityMetrics, InputAnnotationMetrics, InputPrivacyMetrics, SyntheticFeatureMetrics]


class DatasetParams(BaseModel):
    """Parameters for the Data Evaluation endpoint."""

    validation_type: ValidationTypes = Field(
        Query(
            ...,
            title="Validation Types",
            description="Type of validation (e.g. input_val, synthesis_val)",
        ),
    )
    reference_type: ReferenceType = Field(
        Query(
            ...,
            title="Reference Type",
            description="Type of reference to use (e.g. instance, dataset, none)",
        ),
    )
    metric_group: Annotated[
        List[Union[SyntheticMetricGroups, InputMetricGroups]],
        Query(),
    ] = Field(
        Query(
            ...,
            title="Metric Group",
            description="Main evaluation group(s) of the metrics (e.g. quality, annotation, privacy, feature). Mandatory field.",
        ),
    )
    evaluation_level: Optional[EvaluationLevel] = Field(
        Query(
            None,
            title="Evaluation Level",
            description="Compute metrics on a dataset or instance level (e.g. dataset_level, instance_level)",
        ),
    )
    metric_goal: Annotated[List[_METRIC_GOALS], Query()] = Field(
        Query(
            [],
            title="Metric Goals",
            description="Metric specific goal(s) (e.g. contrast, brightness, completeness, etc.). Defaults to None - evaluate on all goals of the main goal.",
        ),
    )
    annotation_file: Optional[str] = Field(
        Query(
            None,
            title="Annotation File",
            description="Name of the annotation file (with extension) in the modality folder.",
        ),
    )

    @field_validator("metric_goal", "metric_group")
    def process_goals(cls, v):
        if isinstance(v, list) and len(v) == 0:
            return None
        return v

    @model_validator(mode="after")
    def check_model_dependencies(self):
        # check if metric specific goal is part of the specfied metric group
        if self.metric_goal:
            if not all(any(valid_subclass(group, goal) for group in self.metric_group) for goal in self.metric_goal):
                raise RequestValidationError(
                    errors=[
                        {
                            "loc": ("query", "metric_goal"),
                            "msg": "Some metric_goals are not a part of any metric_group.",
                            "type": "error",
                        },
                    ],
                )

        # check if metric groups are in line with the specified validation type
        if self.metric_group:
            expected = InputMetricGroups if self.validation_type == ValidationTypes.INPUT else SyntheticMetricGroups
            valid = False
            if self.validation_type == ValidationTypes.INPUT:
                valid = all(InputMetricGroups.has_value(group) for group in self.metric_group)
            elif self.validation_type == ValidationTypes.SYNTH:
                valid = all(SyntheticMetricGroups.has_value(group) for group in self.metric_group)
            if not valid:
                raise RequestValidationError(
                    errors=[
                        {
                            "loc": ("query", "metric_goal"),
                            "msg": f"Metric group(s) {self.metric_group} not valid for validation type {self.validation_type}",
                            "type": "string",
                            "input": self.metric_group,
                            "ctx": {"expected": ", ".join([group for group in expected])},
                        },
                    ],
                )


class MetricInfoParams(BaseModel):
    data_modalities: Annotated[List[Union[DataModalities, None]], Query()] = Field(
        Query(
            [],
            title="Data Modalities",
            description="Select metrics for specific data modalities (e.g. time_series, image, tabular, text). Defaults to None - fetch all data modalities.",
        ),
    )
    validation_types: Annotated[List[Union[ValidationTypes, None]], Query()] = Field(
        Query(
            [],
            title="Validation Types",
            description="Select metrics for input or synthesis validation. (e.g. input_val, synthesis_val). Defaults to None - fetch all validation types.",
        ),
    )
    metric_groups: Annotated[List[Union[InputMetricGroups, SyntheticMetricGroups, None]], Query()] = Field(
        Query(
            [],
            title="Metric Groups",
            description="Metric groups of the functions (e.g. quality, annotation, privacy, feature). Defaults to None - fetch all modality groups.",
        ),
    )

    @field_validator("data_modalities", "validation_types", "metric_groups")
    def process_goals(cls, v):
        if isinstance(v, list) and len(v) == 0:
            return None
        return v


class SpecificFunctionParams(BaseModel):
    """Parameters for the Data Evaluation endpoint."""

    validation_type: ValidationTypes = Field(
        Query(
            ...,
            title="Validation Types",
            description="Type of validation (e.g. input_val, synthesis_val)",
        ),
    )
    metric_names: List[str] = Field(
        Query(
            ...,
            title="Function Names",
            description="List of metric names to be computed.",
        ),
    )
    reference_type: ReferenceType = Field(
        Query(
            ...,
            title="Reference Type",
            description="Type of reference to use (e.g. instance, dataset, none)",
        ),
    )
    annotation_file: Optional[str] = Field(
        Query(
            None,
            title="Annotation File",
            description="Name of the annotation file (with extension) in the modality folder.",
        ),
    )


class DatasetLevelResponse(BaseModel):
    """Reponse for the dataset level evaluation request."""

    num_instances: int = Field(
        title="Number of Instances",
        description="Number of instances used to calculate the metrics.",
    )
    label_metrics: Optional[Dict[str, EvalLevelOutput]] = Field(
        None,
        title="Label Metrics",
        description="Results of the label metrics.",
    )
    raw_data_metrics: Dict[str, EvalLevelOutput] = Field(
        title="Raw Data Metrics",
        description="Values for each metric (refer to SimpleMetric schema).",
    )


class InstanceLevelResponse(BaseModel):
    """Response for instance level evaluation request."""

    instance_names: List[Union[str, int]] = Field(
        title="Instance Names",
        description="List with the names of every instance in the results.",
    )
    label_metrics: Optional[Dict[str, EvalLevelOutput]] = Field(
        None,
        title="Label Metrics",
        description="Results of the label metrics.",
    )
    raw_data_metrics: Dict[str, EvalLevelOutput] = Field(
        title="Raw Data Metrics",
        description="Values obtained for a given metric and for each image (refer to SimpleMetrics Schema). Metric values are respective to the files in instance names.",
    )


class InstanceEvalResponse(BaseModel):
    """Response for the instance evaluation request."""

    instance_level: InstanceLevelResponse = Field(
        title="Instance Level Response",
        description="Instance level response for the dataset evaluation.",
    )
    errors: Optional[Dict[str, List[str]]] = Field(
        None,
        title="Errors",
        description="Caught errors during the evaluation (mostly user errors or inherent problems with metric formulations).",
    )


class DatasetEvalResponse(BaseModel):
    """Response for the dataset evaluation request."""

    dataset_level: DatasetLevelResponse = Field(
        ...,
        title="Dataset Level Response",
        description="Dataset level response for the dataset evaluation.",
    )
    instance_level: Optional[InstanceLevelResponse] = Field(
        None,
        title="Instance Level Response",
        description="Instance level response for the dataset evaluation (Defaults to None).",
    )
    errors: Optional[Dict[str, List[str]]] = Field(
        None,
        title="Errors",
        description="Caught errors during the evaluation (mostly user errors or inherent problems with metric formulations).",
    )


class MetricInfo(BaseModel):
    """Schema for metric information."""

    name: str = Field(title="Name", description="Metric name.")
    data_modality: DataModalities = Field(title="Data Modality", description="Data modality of the metric.")
    validation_type: ValidationTypes = Field(title="Validation Type", description="Data validation type of the metric.")
    evaluation_level: Union[EvaluationLevel, List[EvaluationLevel]] = Field(
        title="Evaluation Level",
        description="Evaluation level of the metric (e.g. dataset_wise, instance_wise).",
    )
    reference_type: ReferenceType = Field(
        title="Reference Type",
        description="Reference type of the metric (e.g. dataset, instance, none).",
    )
    metric_group: Union[SyntheticMetricGroups, InputMetricGroups] = Field(
        title="Metric Group",
        description="Evaluation group for the gieven metric (e.g. quality, annotation, privacy, feature).",
    )
    metric_goal: Optional[Union[_METRIC_GOAL_TYPES, List[_METRIC_GOAL_TYPES]]] = Field(
        None,
        title="Metric Goal",
        description="Metric specific goal (e.g. contrast, brightness, colorfulness, sharpness, perceptual_quality, noise, similarity, uniformity, uniqueness, consistency, completeness, correctness, uniqueness, etc.).",
    )

    description: Optional[str] = Field(None, title="Description", description="Docstring description of the metric.")


class Message(BaseModel):
    """Simple model for error message reporting."""

    msg: str = Field(title="Message", description="Error message.")
    msg_type: str = Field(
        "error",
        alias="type",
        title="Message Type",
        description="Type of message (e.g. error, warning, info).",
    )
    inputs: Optional[Dict[str, str]] = Field(
        None,
        title="Inputs",
        description="Inputs that caused the error.",
    )
    info: Optional[Dict[str, str]] = Field(
        None,
        title="Info",
        description="Additional information.",
    )
