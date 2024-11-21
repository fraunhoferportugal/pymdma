import argparse
import json
from importlib import import_module
from pathlib import Path

from loguru import logger

from pymdma.common.compute import ComputationManager
from pymdma.common.output import create_output_structure
from pymdma.common.selection import (
    select_metric_functions,
    select_modality_input_layer,
    select_specific_metric_functions,
)
from pymdma.constants import (
    DataModalities,
    EvaluationLevel,
    InputCategories,
    ReferenceType,
    SyntheticCategories,
    ValidationDomain,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the data evaluation pipeline.")
    parser.add_argument(
        "--modality",
        "-m",
        type=str,
        required=True,
        choices=["image", "time_series", "tabular", "text"],
        help="Data modality to be evaluated. Options: image, time_series, tabular.",
    )
    parser.add_argument(
        "--validation_domain",
        type=str,
        default="synth",
        choices=["input", "synth"],
        help="Validation type to be evaluated. Options: synth, input, output.",
    )
    parser.add_argument(
        "--reference_type",
        type=str,
        default="dataset",
        choices=["dataset", "instance", "none"],
        help="Reference type to be evaluated. Options: dataset, instance.",
    )
    parser.add_argument(
        "--evaluation_level",
        type=str,
        default=["dataset", "instance"],
        help="Evaluation level. Options: dataset, instance.",
    )
    parser.add_argument(
        "--metric_category",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to be evaluated. E.g. privacy, quality etc.",
    )
    parser.add_argument(
        "--metric_groups",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to be evaluated. E.g: diversity, fidelity.",
    )
    parser.add_argument(
        "--specific_metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metric names to be evaluated.",
    )
    parser.add_argument(
        "--reference_data",
        type=Path,
        default=None,
        help="""Path to the reference data used in full-reference metrics or in synthetic evaluation.
        Can be either a directory of files or a single file.""",
    )
    parser.add_argument(
        "--target_data",
        type=Path,
        required=True,
        help=""""Path to the target data that is to be evaluated.
        Can be either a directory of files or a single file.""",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for the data input layer. Defaults to 20.",
    )
    parser.add_argument("--output_dir", type=Path, default="metric_results", help="Path to the output file.")
    parser.add_argument(
        "--file_path_column",
        type=str,
        default="file_path",
        help="Column name for the file path in the json/csv file. Only needed if one of the data sources is in json or csv format. Defaults to 'file_path'.",
    )
    parser.add_argument(
        "--allow_feature_cache",
        action="store_true",
        help="Allow caching of features for faster evaluation. Defaults to False. Only caching reference features in synthetic evaluation.",
    )
    parser.add_argument(
        "--extractor_model_name",
        type=str,
        default=None,
        help="Name of the model to be used in the feature extraction module. Defaults to None.",
    )
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=None,
        help="Path to the annotation file of the reference dataset. Defaults to None.",
    )
    parser.add_argument(
        "--compute_n_workers",
        type=int,
        default=1,
        help="Number of workers to be used in the computation. Defaults to 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to be used for computation. Defaults to 'cpu'.",
    )
    return parser.parse_args()


def infer_data_source(data_modality: str, data_path: Path):
    """Infer the data source based on the provided data modality and path."""
    if data_path is None:
        return None

    if not data_path.exists():
        raise FileNotFoundError(f"Data source not found: {data_path}")

    if data_path.is_dir():
        return data_path

    # modality custom data parsers
    module = import_module(f"pymdma.{data_modality}.data.parsers")
    if data_path.suffix == ".jsonl":
        return module.jsonl_files(data_path)

    # TODO add more
    raise NotImplementedError("Data source not supported")


def main() -> None:
    args = parse_args()

    # INPUTS
    # System-generated inputs
    data_modality = DataModalities[args.modality.upper()]
    validation_domain = ValidationDomain[args.validation_domain.upper()]
    reference_type = ReferenceType[args.reference_type.upper()]
    evaluation_level = EvaluationLevel[args.evaluation_level.upper()]

    # METRIC GOALS
    # None means all metrics
    if args.metric_category is None:
        metric_category = None
    elif validation_domain == ValidationDomain.SYNTH:
        metric_category = [SyntheticCategories[value.upper()] for value in args.metric_category]
    else:
        metric_category = [InputCategories[value.upper()] for value in args.metric_category]

    # ENDPOINT 3 - Specific metric obtained by metric name
    if args.specific_metrics is not None:
        s_func = select_specific_metric_functions(
            args.specific_metrics,
            data_modality,
            validation_domain,
            reference_type,
        )
    else:
        s_func = select_metric_functions(
            data_modality,
            validation_domain,
            reference_type,
            evaluation_level,
            metric_category,
            metric_groups=None,
        )

    if args.annotation_file is None:
        s_func.pop("annotation", None)

    for eval_group in list(s_func.keys()):
        funcs = s_func[eval_group]
        if len(funcs) == 0:
            logger.error(f"Found no metrics for {eval_group} group and the provided parameters. Skipping.")
            s_func.pop(eval_group)

    if len(s_func) == 0:
        logger.error("No metrics were found for the provided parameters. Exiting.")
        exit(1)

    # Prepare data input layer
    reference_data = infer_data_source(args.modality, args.reference_data)
    target_data = infer_data_source(args.modality, args.target_data)
    assert (
        reference_type != ReferenceType.NONE and reference_data is not None or reference_type == ReferenceType.NONE
    ), "Reference data is required for this evaluation."
    data_input_layer = select_modality_input_layer(
        data_modality,
        validation_domain,
        reference_type,
        target_data,
        reference_data,
        args.batch_size,
        args.output_dir if args.allow_feature_cache else None,
        annotation_file=args.annotation_file,
        device=args.device,
    )

    logger.info(
        f"""
    ========= RUNNING EVALUATION =========
    Data Modality = {data_modality}
    Validation Type = {validation_domain}
    Reference Type = {reference_type}
    Evaluation Level = {evaluation_level}
    Metric Goal = {metric_category}
    Reference Data = {args.reference_data}
    Target Data = {args.target_data}
    ======================================
    """,
    )

    for group, funcs in s_func.items():
        logger.info(f"{group.split('.')[-1].title()} Metrics: {', '.join([func.__name__ for func in funcs])}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    manager = ComputationManager(
        group_classes=s_func,
        data_input_layer=data_input_layer,
        data_modality=data_modality,
        validation_domain=validation_domain,
        output_dir=args.output_dir,
        pretrained_extractor_name=args.extractor_model_name,
        n_workers=args.compute_n_workers,
    )

    results = manager.compute_metrics()

    output = create_output_structure(
        results,
        schema="v1",
        instance_ids=data_input_layer.instance_ids,
        n_samples=len(data_input_layer),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "output.json", "w") as f:
        f.write(json.dumps(output, indent=2))

    with open(args.output_dir / "config.json", "w") as f:
        args_vals = {key: str(val) for key, val in dict(vars(args)).items()}
        json.dump(args_vals, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
