# Implementing Metrics

Before adding any metrics, please consult the [contributing](contributing.md) guidelines and the [developer](developer.md) documentation.

> **Important:** We only accept metrics that are published in peer-reviewed journals. Every metric metric in the library must have a valid reference to a published paper or be widely known in the field.

## Base Metric Class

::: pymdma.common.definitions.Metric

Every metric inherits from the `Metric` abstract class in `pymdma.common.definitions`. The new metric must inherit from this class and implement the following methods:

- `__init__()`: Initialize the metric with the provided keyword arguments. The keyword arguments must be documented in the class docstring.
- `compute()`: Compute the metric value. This method consumes raw data and must return a `MetricResult` object. Make sure that the inputs and the outputs are well documented and are consistent with the data types used in similar metrics.

### Class Attributes

Every metric is categorized with specific attributes that should be overriden in the metric class:

- `reference_type`: Indicates wether the compute method expects a reference and a target dataset or not.
- `evaluation_level`: Can either be a list of `EvaluationLevel` or a single `EvaluationLevel`. Indicates wether the metric is dataset-wise or instance-wise.
- `metric_group`: Indicates the metric category (consult the hierarchy diagram in the homepage).
- `higher_is_better`: If a higher metric result is better, set this to `True`.
- `min_value`: Minimum possible value for the metric.
- `max_value`: Highest possible value for the metric.

### Metric Documentation

In addition to a clear description, the class documentation include the following sections:

- **Objective**: Describe the objective of the metric, e.g., "Similarity", "Authenticity", etc.
- **Parameters**: List the parameters of the metric `__init__` method, e.g., `fs: int, optional, default=2048`
- **References**: List relevant references for the metric, such as journal articles, conference papers, or books. If the metric is published in a peer-reviewed journal, please provide the DOI link.
- **Example**: Provide a simple example of how to use the metric in code, e.g., `metric = YourMetric(...)`.

Following is an example of a new metric class:

```python
from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult

from pymdma.constants import EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType


class NewMetric(Metric):
    """Metric description

    **Objective**: An objective

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility.

    References
    ----------
    Author et al. Paper title (year). <link to paper>.

    Examples
    --------
    >>> new_metric = NewMetric()
    >>> data = np.random.rand(100, 100)
    >>> result: MetricResult = new_metric.compute(data)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_group = MetricGroup.QUALITY

    higher_is_better: bool = False
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def compute(self, data, **kwargs) -> MetricResult:
        """Computes colorfulness level of list of images.

        Parameters
        ----------
        data : type
            description

        Returns
        -------
        result: MetricResult
            small description
        """
        # Delete one of the level results if the metric only has a single evaluation level
        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.NUMBER,
                "subtype": "float",
                "value": 0.0,
            },
            instance_level={
                "dtype": OutputsTypes.ARRAY,
                "subtype": "float",
                "value": scores,
            },
        )
```

### Metric Result Class

The `MetricResult` class is used to store and validate the metric outputs. It is mostly based on [pydantic](https://docs.pydantic.dev/latest/) validation models. The class is defined in `pymdma.common.output` and should be instanciated and returned in the `compute` method of the metric class.

::: pymdma.common.output.MetricResult

It has the following attributes:

- `dataset_level`: Result of the metric for the entire dataset (if available).
- `instance_level`: Result of the metric for each instance (if available).
- `errors`: A list of errors that occurred during the metric computation (can ignore).

::: pymdma.common.output.EvalLevelOutput

Each level attribute is a [pydantic](https://docs.pydantic.dev/latest/) model with the following fields:

- `dtype`: The data type of the metric result.
- `subtype`: The data subtype of the metric result.
- `value`: The metric result value.
- `stats`: Additional statistics of the metric result.
- `plot_params`: Plotting parameters for the metric.

## Contributing your metric

Remember to read the [contributing](contributing.md) and the [developer](developer.md) documentation. For any new metric, you must adhere to the metric hierarchy diagram in the [homepage](index.md). If the metric does not fit the hierarchy, please raise an issue in the [GitHub repository](https://github.com/fraunhoferportugal/pymdma/issues).

Once the hierarchy is clear, you can create a new metric class. Start by creating a new python script under `src/pymdma/<data_modality>/measures/<validation_domain>/<metric_category>/<your_metric>.py`. And then follow the instructions explained bellow.

1. Define a new metric class that inherits from the `Metric` abstract class in `pymdma.common.definitions`. The new metric must inherit from this class and implement the above mentioned methods.
   - Avoid introducing any third party dependencies.
   - Functions used for intermediate computation should be included in this script.
   - At the end of the file define the `__all__` variable to export the metric class.
1. Add a metric import to the `__init__.py` file in the `src/pymdma/<data_modality>/measures/<validation_domain>` module and add the name to the `__all__` variable of the same file.
1. Develop at least one test case for the metric using the `pytest` framework. Create a test script with the name `test_<your_metric>.py` under the `tests/` folder and write at least one method starting with `test_` that uses the metric. Try to cover edge cases when possible.

Once you're done, feel free to open a pull request in the [GitHub repository](https://github.com/fraunhoferportugal/pymdma/issues).\
If you have any questions or run into any issues along the way, just leave a comment in the PR — we're happy to help!
