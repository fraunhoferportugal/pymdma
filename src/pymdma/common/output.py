from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from pymdma.constants import OutputsTypes

from .viz import plot_from_results, plt

number = Union[int, float]
SUPPORTED_OUTPUTS = Union[number, List[number], Dict[Union[str, number], Union[number, List[number]]]]


class PlotParams(BaseModel):
    kind: Literal["bar", "line", "hist"] = Field("bar", title="Plot Type", description="Type of plot to be generated")
    x_label: Optional[str] = Field(None, title="X axis label", description="Label for the x axis")
    y_label: Optional[str] = Field(None, title="Y axis label", description="Label for the y axis")
    x_key: Optional[str] = Field(
        None,
        title="X Value Key",
        description="Key to se when fecthing X plot values from the value field",
    )
    y_key: Optional[str] = Field(
        None,
        title="Y Value Key",
        description="Key to se when fecthing Y plot values from the value field",
    )


class EvalLevelOutput(BaseModel):
    dtype: OutputsTypes = Field(..., title="Data Type", description="Internal type representation of the output value")
    subtype: Literal["int", "float", "str"] = Field(..., title="Subtype", description="Subtype of the output value")
    value: SUPPORTED_OUTPUTS = Field(..., title="Value", description="Metric value")
    stats: Optional[Dict[str, SUPPORTED_OUTPUTS]] = Field(
        None,
        title="Stats",
        description="Additional statistics of the metric value",
    )
    plot_params: Optional[PlotParams] = Field(
        None,
        title="Plot Params",
        description="Plotting parameters for the metric",
    )

    @model_validator(mode="after")
    def _validate_plot_level(self):
        if self.plot_params is None:
            return self

        plot_params = self.plot_params
        if plot_params.x_key:
            assert OutputsTypes(self.dtype) in {
                OutputsTypes.KEY_ARRAY,
                OutputsTypes.KEY_VAL,
            }, "X key can only be provided when the output type is KEY_ARRAY or KEY_VAL"
            assert plot_params.x_key in self.value, "The provided X key is not present in the output value"
        if plot_params.y_key:
            assert OutputsTypes(self.dtype) in {
                OutputsTypes.KEY_VAL,
                OutputsTypes.KEY_ARRAY,
            }, "Y key can only be provided when the output type is KEY_VAL or ARRAY"
            assert plot_params.y_key in self.value, "The provided Y key is not present in the output value"
        return self


class MetricResult(BaseModel):
    dataset_level: Optional[EvalLevelOutput] = Field(None, title="Dataset Level", description="Dataset level output")
    instance_level: Optional[EvalLevelOutput] = Field(None, title="Instance Level", description="Instance level output")
    errors: Optional[List[str]] = Field(
        None,
        title="Errors",
        description="Errors during the metric computation",
    )

    @model_validator(mode="after")
    def check_levels(self):
        if self.dataset_level is None and self.instance_level is None:
            raise ValueError("At least one of the output levels should be provided")
        return self

    @property
    def value(self) -> Tuple[Optional[EvalLevelOutput], Optional[EvalLevelOutput]]:
        """Return the dataset-level and instance-level tuple of the metric
        result.

        Returns
        -------
        Tuple[Optional[EvalLevelOutput], Optional[EvalLevelOutput]]
            The dataset-level and instance-level output of the metric result.
        """
        return self.dataset_level.value if self.dataset_level else None, (
            self.instance_level.value if self.instance_level else None
        )

    @property
    def stats(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Return the dataset-level and instance-level statistics of the metric
        result.

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]
            The dataset-level and instance-level statistics of the metric result.
        """
        return self.dataset_level.stats if self.dataset_level else None, (
            self.instance_level.stats if self.instance_level else None
        )

    @property
    def verbose_value(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the dataset-level and instance-level verbose value
        (dictionary) of the metric result.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            The dataset-level and instance-level verbose value of the metric result.
        """
        return self.dataset_level.model_dump() if self.dataset_level else None, (
            self.instance_level.model_dump() if self.instance_level else None
        )

    # TODO validate plot_params

    def plot(self, title: str = "", ax: Optional[plt.Axes] = None, **plot_kwargs):
        """Plot the metric results.

        Parameters
        ----------
        title : str, optional
            The title of the plot, by default "".
        ax : Optional[plt.Axes], optional
            The axes to plot the results on, by default None.
        **plot_kwargs
            Additional keyword arguments passed to the plotting function.
        """
        plot_from_results(
            title=title,
            instance_level=self.instance_level,
            dataset_level=self.dataset_level,
            ax=ax,
            **plot_kwargs,
        )


class DistributionResult(MetricResult):
    """Simple wrapper for populating the plot_params for distribution
    metrics."""

    @model_validator(mode="after")
    def hist_params(self):
        assert self.instance_level is not None, "Instance level should be provided"
        assert self.instance_level.dtype == OutputsTypes.ARRAY, "Instance level should be an array"
        self.instance_level.plot_params = PlotParams(
            **{
                "kind": "hist",
                "x_label": "Values",
                "y_label": "Frequency",
            },
        )
        return self


def create_output_structure(
    group_results: Dict[str, MetricResult],
    schema: Literal["v2", "v1"] = "v1",
    instance_ids: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Create the output structure for the evaluation results.

    Parameters
    ----------
    group_results : Dict[str, MetricResult]
        The results of the evaluation metrics.
    schema : Literal["v2", "v1"], optional
        The schema version to use, by default "v1".
    instance_ids : Optional[List[str]], optional
        The instance ids, by default None.
    n_samples : Optional[int], optional
        The number of samples, by default None.

    Returns
    -------
    Dict[str, Any]
        Structured JSON file with the evaluation results.
    """
    assert instance_ids is not None or n_samples is not None, "Either instance_ids or n_samples should be provided"
    output = {}
    errors = {}
    if schema == "v1":
        for metric_category, metric_results in group_results.items():
            for metric_name, metric_result in metric_results.items():
                dataset_level, instance_level = metric_result.verbose_value

                # create dataset level structure
                if dataset_level is not None:
                    output.setdefault(
                        "dataset_level",
                        {
                            "num_instances": len(instance_ids) if instance_ids is not None else n_samples,
                            "label_metrics": {} if "annotation" in group_results else None,
                            "raw_data_metrics": {},
                        },
                    )

                    level_key = "raw_data_metrics" if not metric_category == "annotation" else "label_metrics"
                    output["dataset_level"][level_key].update(
                        {
                            metric_name: dataset_level,
                        },
                    )

                # create instance level strcuture
                if instance_level is not None:
                    output.setdefault(
                        "instance_level",
                        {
                            "instance_names": (
                                instance_ids if instance_ids is not None else list(map(str, range(n_samples)))
                            ),
                            "label_metrics": {} if "annotation" in group_results else None,
                            "raw_data_metrics": {},
                        },
                    )

                    level_key = "raw_data_metrics" if not metric_category == "annotation" else "label_metrics"
                    output["instance_level"][level_key].update(
                        {
                            metric_name: instance_level,
                        },
                    )

                if metric_result.errors is not None:
                    for error in metric_result.errors:
                        errors.setdefault(metric_name, []).append(error)
        output["errors"] = errors if len(errors) > 0 else None
        return output
    elif schema == "v2":

        for metric_category, metric_results in group_results.items():
            for metric_name, metric_result in metric_results.items():
                dataset_level, instance_level = metric_result.verbose_value

                if dataset_level is not None:
                    output.setdefault(
                        "dataset_level",
                        {
                            "num_instances": len(instance_ids),
                            "label_metrics": None,
                        },
                    )

                    output["dataset_level"].setdeafult({metric_category: {}})

                    output["dataset_level"][metric_category].update(
                        {
                            metric_name: dataset_level,
                        },
                    )

                if instance_level is not None:
                    output.setdefault(
                        "instance_level",
                        {
                            "instance_names": (
                                instance_ids if instance_ids is not None else list(map(str, range(n_samples)))
                            ),
                            "label_metrics": None,
                        },
                    )

                    output["instance_level"][metric_category].update(
                        {
                            metric_name: instance_level,
                        },
                    )

                if metric_result.errors is not None:
                    for error in metric_result.errors:
                        errors.setdefault("metric_name", []).append(error)
        output["errors"] = errors if len(errors) > 0 else None
        return output
