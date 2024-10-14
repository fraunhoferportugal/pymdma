from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt


def plot_from_results(
    title: str,
    instance_level=None,
    dataset_level=None,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
):
    if instance_level is not None and instance_level.plot_params is not None:
        plot_params = instance_level.plot_params
        x_value = instance_level.value[plot_params.x_key] if plot_params.x_key else None
        y_value = instance_level.value[plot_params.y_key] if plot_params.y_key else instance_level.value
        return plot_single_multi_value_metric(
            title=title,
            x_value=x_value,
            y_value=y_value,
            x_label=plot_params.x_label,
            y_label=plot_params.y_label,
            kind=plot_params.kind,
            ax=ax,
            **plot_kwargs,
        )
    if dataset_level is not None and dataset_level.plot_params is not None:
        plot_params = dataset_level.plot_params
        x_value = dataset_level.value[plot_params.x_key] if plot_params.x_key else None
        y_value = dataset_level.value[plot_params.y_key] if plot_params.y_key else dataset_level.value
        return plot_single_multi_value_metric(
            title=title,
            x_value=x_value,
            y_value=y_value,
            x_label=plot_params.x_label,
            y_label=plot_params.y_label,
            kind=plot_params.kind,
            ax=ax,
            **plot_kwargs,
        )

    if instance_level is not None:
        return plot_single_multi_value_metric(
            title=title,
            y_value=instance_level.value,
            kind="hist",
            ax=ax,
            **plot_kwargs,
        )

    if dataset_level is not None:
        return plot_single_multi_value_metric(
            title=title,
            y_value=dataset_level.value,
            kind="bar",
            ax=ax,
            **plot_kwargs,
        )


def plot_single_multi_value_metric(
    title: str,
    y_value: Union[Union[int, float], List[Union[int, float]]],
    x_value: Optional[List[Union[int, float]]] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
    kind: Literal["bar", "line", "hist"] = "bar",
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Plot a single or multi value metric.

    Args:
        metric_name: Name of the metric.
        y_value: Value of the metric.
        x_value: Optional x value.
        kind: Type of plot to be generated.
        output_format: Format of the output.
        **kwargs: Additional arguments to be passed to the plot function.

    Returns:
        If output_format is "plot", the plot will be displayed.
        If output_format is "json", the plot will be saved as a json file.
    """

    args = (x_value, y_value) if x_value is not None else (y_value,)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if kind == "bar":
        if x_value is not None:
            ax.bar(x_value, height=y_value, **kwargs)
        else:
            # single value plot
            ax.bar([0], height=y_value, **kwargs)
    elif kind == "line":
        ax.plot(*args, **kwargs)
    elif kind == "hist":
        ax.hist(y_value, **kwargs)
    else:
        raise ValueError(f"Plot type {kind} is not supported")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig
