from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from loguru import logger

from pymdma.constants import DataModalities, OutputsTypes, ValidationDomain

from .definitions import InputLayer


class ComputationManager:
    """Manager class to compute metrics, reduce results and keep track of
    states.

    Parameters
    ----------
    group_classes : Dict[str, List[callable]]
        Dictionary with a list of every metric class for a given metric group.
    data_input_layer : InputLayer
        Input layer instance with data properties.
    data_modality : DataModalities
        Data modality type.
    validation_domain : ValidationDomain
        Validation domain.
    output_dir : Optional[Path], optional
        Output directory to save results, by default None.
    pretrained_extractor_name : Optional[str], optional
        Name of the pretrained extractor model, by default None.
    n_workers : int, optional
        Number of workers to use for parallel computation, by default 1.
    """

    def __init__(
        self,
        group_classes: Dict[str, List[callable]],
        data_input_layer: InputLayer,
        data_modality: DataModalities,
        validation_domain: ValidationDomain,
        output_dir: Optional[Path] = None,
        pretrained_extractor_name: Optional[str] = None,
        n_workers: int = 1,
    ) -> None:
        self.data_input_layer = data_input_layer
        self.data_modality = data_modality
        self.validation_domain = validation_domain
        self.output_dir = output_dir
        self.n_workers = n_workers

        self.metrics = self._instanciate_metric_classes(group_classes)

        self.global_context: Dict[str, any] = {}

        self.extractors = set() if pretrained_extractor_name is None else {pretrained_extractor_name}
        self._overrode_extractor = pretrained_extractor_name is not None
        self._collect_requirements(self.metrics)

    def _collect_requirements(self, group_metrics: Dict[str, List[callable]]) -> None:
        """Collects requirements for the computation of metrics."""
        if (
            not self._overrode_extractor
            and "feature" in group_metrics
            and self.validation_domain == ValidationDomain.SYNTH
        ):
            self.extractors = {
                metric.extractor_model_name
                for metric in group_metrics["feature"]
                if hasattr(metric, "extractor_model_name")
            }
            logger.info(f"Extractors required: {self.extractors}")

    def _instanciate_metric_classes(self, group_functions: Dict[str, List[callable]]) -> Dict[str, List[object]]:
        """Instanciate metric classes for each group."""
        group_instances = {group: [] for group in group_functions.keys()}
        for group, class_def in group_functions.items():
            if class_def is not None:
                # TODO custom args here
                group_instances[group] = [
                    metric_class(**self.data_input_layer.data_properties) for metric_class in class_def
                ]
            else:
                logger.warning(f"Class for {group} not found.")
        return group_instances

    def _compute_and_reduce(
        self,
        metric_category: str,
        metrics: List[object],
        metric_args: Tuple[Any],
        single_value_strategy: Literal["mean", "sum"] = "mean",
    ) -> None:
        """Compute metrics and reduce results.

        Parameters
        ----------
        metric_category : str
            Metric group name.
        metrics : List[object]
            List of metric instances.
        metric_args : Tuple[Any]
            Arguments to pass to the metric.
        single_value_strategy : Literal["mean", "sum"], optional
            Strategy to reduce single value metrics when merging results, by default "mean".

        Notes
        -----
        This method is parallelized using ThreadPoolExecutor, the number of workers is defined by the n_workers attribute.
        This allows for the parallel computation of metrics for each batch.
        """

        def _compute_task(metric, metric_args):
            metric_name = metric.__class__.__name__
            logger.info(f"Metric: {metric_name}")
            new_result = metric.compute(*metric_args, context=self.global_context)

            # merge metric with already compute one (batch calculation)
            if metric_name in self.metric_results[metric_category]:
                previous_result = self.metric_results[metric_category][metric_name]
                prev_dataset_lvl, prev_instance_lvl = previous_result.value
                new_dataset_lvl, new_instance_lvl = new_result.value

                # merge dataset level result
                if prev_dataset_lvl is not None:
                    logger.warning(
                        f"Metric {metric_name} already computed. Combining with {single_value_strategy} strategy.",
                    )
                    new_result.dataset_level.value = new_dataset_lvl + prev_dataset_lvl
                    new_result.dataset_level.value /= 2 if single_value_strategy == "mean" else 1

                # merge instance level result
                if prev_instance_lvl is not None:
                    instance_dtype = new_result.instance_level.dtype
                    if instance_dtype == OutputsTypes.ARRAY:
                        new_result.instance_level.value = prev_instance_lvl + new_instance_lvl
                    else:
                        raise NotImplementedError(f"{instance_dtype} strategy not implemented for instance level.")
            return metric_name, new_result

        with ThreadPoolExecutor(self.n_workers) as executor:
            tasks = [executor.submit(_compute_task, metric, metric_args) for metric in metrics]
            for task in as_completed(tasks):
                metric_name, result = task.result()
                self.metric_results[metric_category][metric_name] = result

    def compute_metrics(self, model_instances: Optional[Dict[str, callable]] = None):
        """Compute metrics for each group and reduce results.

        Parameters
        ----------
        model_instances : Optional[Dict[str, callable]], optional
            Dictionary with preloaded model instances, by default None.

        Returns
        -------
        metric_results : Dict[str, Dict[str, MetricResult]]
            Dictionary with the computed metric results.
        """
        self.metric_results = {group: {} for group in self.metrics.keys()}
        # compute feature based metrics
        if self.validation_domain == ValidationDomain.SYNTH and "feature" in self.metrics:
            for extractor_name in self.extractors:
                logger.info(f"Computing metrics for extractor: {extractor_name}")

                # extract features for the specific extractor and compute metrics for it
                reference_feats, synthetic_feats = self.data_input_layer.get_embeddings(extractor_name, model_instances)
                specific_metrics = [
                    metric
                    for metric in self.metrics["feature"]
                    if (metric.extractor_model_name == extractor_name or self._overrode_extractor)
                ]

                # compute all metrics for the specific extractor
                self._compute_and_reduce("feature", specific_metrics, (reference_feats, synthetic_feats))

                # remove metrics for the specific extractor to avoid recomputing them
                del reference_feats, synthetic_feats
                self.metrics["feature"] = [
                    metric
                    for metric in self.metrics["feature"]
                    if (metric.extractor_model_name != extractor_name or self._overrode_extractor)
                ]
            self.metrics.pop("feature")

        # compute annotation metrics
        if "annotation" in self.metrics:
            self._compute_and_reduce("annotation", self.metrics["annotation"], (self.data_input_layer.annotations,))
            self.metrics.pop("annotation")

        # compute data based metrics
        n_batches = len(self.data_input_layer.target_loader)
        for eval_group, class_instances in self.metrics.items():
            for idx, batch in enumerate(self.data_input_layer.batched_samples):
                logger.info(f"Processing batch [{idx + 1}/{n_batches}]")
                self._compute_and_reduce(eval_group, class_instances, batch)
        return self.metric_results
