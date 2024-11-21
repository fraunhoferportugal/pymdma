import inspect
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from pymdma.common.definitions import InputLayer, Metric
from pymdma.config import package_base_dir
from pymdma.constants import (
    METRICS_PACKAGE_NAME,
    DataModalities,
    EvaluationLevel,
    InputCategories,
    MetricGroup,
    ReferenceType,
    SyntheticCategories,
    ValidationDomain,
)


def _get_constant_values(class_name):
    # Get all attributes of the class
    attributes = [
        attr for attr in dir(class_name) if not callable(getattr(class_name, attr)) and not attr.startswith("__")
    ]
    # Filter out only the constants and get their values
    constant_values = [getattr(class_name, attr) for attr in attributes if not callable(getattr(class_name, attr))]
    return constant_values


def _check_class_attribute(source, target):
    if isinstance(target, list):
        return any(source == t for t in target)
    return source == target


def select_modality_input_layer(
    data_modality: DataModalities,
    validation_domain: ValidationDomain,
    reference_type: ReferenceType,
    target_data: Path,
    reference_data: Optional[Path] = None,
    batch_size: int = 20,
    output_dir: Optional[Path] = None,
    annotation_file: Optional[Path] = None,
    device: str = "cpu",
) -> InputLayer:
    """Selects the input layer based on the data modality.

    Parameters
    ----------
    data_modality : str
        The data modality to be used (time_series, image, tabular, text)
    validation_domain : str
        The validation type to be used (input_val or synthesis_val)
    reference_type : str
        The reference type (dataset-wise or instance-wise)
    target_data : Path
        The path to the target data
    reference_data : Path, optional
        The path to the reference data
    batch_size : int, optional
        The batch size to be used (default=20)
    output_dir : Path, optional
        The path to the output directory (default=None)
    annotation_file : Path, optional
        The path to the annotation file (default=None)
    device : str, optional
        The device to be used (default=cpu)

    Returns
    -------
    input_layer : InputLayer
        The input layer
    """
    if data_modality == "image":
        from pymdma.image.input_layer import ImageInputLayer

        # allow for repeat reference in  full reference input validation
        repeat_reference = validation_domain == ValidationDomain.INPUT and reference_type == ReferenceType.INSTANCE
        return ImageInputLayer(
            validation_domain,
            reference_type,
            target_data,
            reference_data,
            batch_size=batch_size,
            repeat_reference=repeat_reference,
            device=device,
            features_cache=output_dir,
            annot_file=annotation_file,
        )
    elif data_modality == "time_series":
        from pymdma.time_series.input_layer import TimeSeriesInputLayer

        return TimeSeriesInputLayer(
            validation_domain,
            reference_type,
            target_data,
            reference_data,
            batch_size=batch_size,
        )
    elif data_modality == "tabular":
        from pymdma.tabular.input_layer import TabularInputLayer

        return TabularInputLayer(
            validation_domain,
            reference_type,
            target_data,
            reference_data,
        )
    elif data_modality == "text":
        from pymdma.text.input_layer import TextInputLayer

        return TextInputLayer(
            validation_domain,
            reference_type,
            target_data,
            reference_data,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError("Unknown data modality")


def import_module_or_folder(validation_root: Path, group_name: str) -> List[str]:
    """Import a module or a folder of modules."""
    if not validation_root.exists():
        raise FileNotFoundError(f"Validation root not found: {validation_root}")

    group_root = validation_root / group_name
    # module is a folder
    if group_root.is_dir():
        return [
            f"{group_name}.{module.stem}"
            for module in group_root.iterdir()
            if module.is_file() and module.suffix == ".py" and module.stem != "__init__"
        ]

    # module is a file
    if group_root.with_suffix(".py").is_file():
        return [group_name]
    # module is not found
    return []


def select_metric_classes(
    data_modality: DataModalities,
    validation_domain: ValidationDomain,
    metric_categorys: Optional[List[Union[SyntheticCategories, InputCategories]]] = None,
    ignore_missing: bool = False,
) -> Dict[str, List[Metric]]:
    """Helper function for importing classes from the metrics package."""
    # script names
    module_root = ".".join((data_modality, METRICS_PACKAGE_NAME, validation_domain))
    # fetch all modules when None # TODO review for current structure

    # fetch all metric groups in a module
    if metric_categorys is None:
        metric_categorys = [path.stem for path in Path(f"{package_base_dir}/{module_root.replace('.', '/')}").iterdir()]
        metric_categorys = [group for group in metric_categorys if "__" not in group]

    group_classes: Dict[str, List[Metric]] = {}
    for metric_category in metric_categorys:
        group_root = module_root.replace(".", "/")
        group_root = Path(f"{package_base_dir}/{group_root}")

        group_modules = import_module_or_folder(group_root, metric_category)

        if len(group_modules) < 1 and not ignore_missing:
            logger.error(
                f"No modules found for {group_root} and the provided metric groups: {metric_categorys}. Check if the provided parameters are correct.",
            )
            return group_classes

        for module in group_modules:
            module_path = f"{package_base_dir.split('/')[-1]}.{module_root}.{module}"
            try:
                module = import_module(module_path)
            except ModuleNotFoundError as e:
                logger.error(f"Error loading module: {module_path}. Detail: {e}")
                return {}

            # Get classes names
            classes_in_module = [
                obj
                for _name, obj in inspect.getmembers(
                    module,
                    lambda member: inspect.isclass(member)
                    and issubclass(member, Metric)
                    and not member.__name__ in {"Metric", "FeatureMetric"},
                )
            ]
            group_classes.setdefault(metric_category, []).extend(classes_in_module)
    return group_classes


def select_metric_functions(
    data_modality: DataModalities,
    validation_domain: ValidationDomain,
    reference_type: ReferenceType,
    evaluation_level: Optional[EvaluationLevel] = None,
    metric_category: Optional[Union[SyntheticCategories, InputCategories]] = None,
    metric_groups: Optional[List[MetricGroup]] = None,
) -> Dict[str, List[Metric]]:
    """Helper function for selecting specific subset of measures.

    Parameters
    ----------
    data_modality : str
        The data modality to be used (time_series, image, tabular, text)
    validation_domain : str
        The validation type to be used (input_val or synthesis_val)
    reference_type : str
        The reference type (dataset-wise or instance-wise)
    evaluation_level : str, optional (default=None)
        The evaluation level (dataset-wise or instance-wise)
    metric_category : str, optional (default=None)
        The metric group to be used
    metric_groups : list, optional (default=None)
        The metric goals to be used

    Returns
    -------
    selected_functions : Dict[str, List[object]]
        The selected functions
    """

    if isinstance(metric_groups, str):
        metric_groups = [metric_groups]

    # get evaluation group class objects
    evaluation_group_classes = select_metric_classes(data_modality, validation_domain, metric_category)

    # class name
    if len(evaluation_group_classes) == 0:
        logger.error(
            f"No classes found for {data_modality}.{validation_domain} and the provided metric groups: {metric_category}. Check if the provided parameters are correct.",
        )
        return {}

    # filter functions based on reference, evaluation and metric_group
    selected_functions = {}
    for evaluation_group, metric_classes in evaluation_group_classes.items():
        # Filter out only the methods defined in the class and exclude those starting with "_"
        selected_functions.setdefault(evaluation_group, [])
        selected_names = set()
        for class_obj in metric_classes:
            # class_obj: Metric
            # discard classes with missing attributes
            if metric_groups is None or class_obj.metric_group in metric_groups:
                if (
                    evaluation_level is None or _check_class_attribute(evaluation_level, class_obj.evaluation_level)
                ) and _check_class_attribute(
                    reference_type,
                    class_obj.reference_type,
                ):
                    class_name = class_obj.__name__
                    selected_functions[evaluation_group] += [class_obj] if class_name not in selected_names else []
                    selected_names.add(class_name)
    return selected_functions


def select_specific_metric_functions(
    metric_names: List[str],
    data_modality: DataModalities,
    validation_domain: ValidationDomain,
    reference_type: Optional[ReferenceType] = None,
):
    """Helper function for selecting metrics by name."""
    # get evaluation group class objects
    evaluation_group_classes = select_metric_classes(data_modality, validation_domain)

    assert (
        len(evaluation_group_classes) > 0
    ), f"No groups found for {data_modality}.{validation_domain}. Check if the provided parameters are correct."

    # assert all(len(metric_fns) > 0 for metric_fns in evaluation_group_classes.values()), "One or more missing metrics. Check if the provided names are correct."

    metric_names = set(metric_names)

    selected_functions = {}
    for eval_group, group_classes in evaluation_group_classes.items():
        for class_obj in group_classes:
            class_name = class_obj.__name__
            if class_name in metric_names and (reference_type is None or reference_type == class_obj.reference_type):
                selected_functions.setdefault(eval_group, [])
                selected_functions[eval_group].append(class_obj)
    return selected_functions


def get_metrics_metadata(
    data_modalities: Optional[DataModalities] = None,
    validation_domains: Optional[ValidationDomain] = None,
    metric_categorys: Optional[Union[SyntheticCategories, InputCategories]] = None,
):
    """Get metadata for all metrics within the specified modalities, validation
    types and metric groups."""
    if data_modalities is None:
        data_modalities = _get_constant_values(DataModalities)
    elif isinstance(data_modalities, str):
        data_modalities = [data_modalities]

    if validation_domains is None:
        validation_domains = _get_constant_values(ValidationDomain)
    elif isinstance(validation_domains, str):
        validation_domains = [validation_domains]

    metrics = {}
    for data_modality in data_modalities:
        for validation_domain in validation_domains:
            evaluation_group_classes = select_metric_classes(
                data_modality,
                validation_domain,
                metric_categorys,
                ignore_missing=True,
            )
            # class name

            if len(evaluation_group_classes) == 0:
                continue

            # Get methods from class
            for evaluation_group, metric_classes in evaluation_group_classes.items():
                # Filter out only the methods defined in the class and exclude those starting with "_"
                selected_metrics = {
                    f"{data_modality}.{validation_domain}.{evaluation_group}.{method.__name__}": {
                        "name": method.__name__,
                        "data_modality": data_modality,
                        "validation_domain": validation_domain,
                        "metric_category": evaluation_group,
                        "description": method.__doc__,
                        **vars(method),
                    }
                    for method in metric_classes
                }

                metrics.update(selected_metrics)
    return metrics
