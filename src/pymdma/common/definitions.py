from abc import ABC, abstractmethod
from typing import Optional

from pymdma.common.output import MetricResult
from pymdma.constants import EvaluationLevel, MetricGroup, ReferenceType


class MetricClass:
    def _post_process_hooks(self, results):
        pass


class InputLayer(ABC):
    def __init__(self):
        self.reference_loader = None
        self.instance_ids = None

    @abstractmethod
    def __len__(self):
        pass

    @property
    def annotations(self):
        raise NotImplementedError

    @property
    def data_properties(self):
        return {}

    @property
    @abstractmethod
    def batched_samples(self):
        pass

    @abstractmethod
    def get_embeddings(self, model_name: str, **kwargs):
        pass


class Metric(ABC):
    # evaluation params
    evaluation_level: EvaluationLevel = EvaluationLevel.DATASET
    metric_group: MetricGroup
    reference_type: ReferenceType = ReferenceType.NONE

    # metric specific
    higher_is_better: Optional[bool] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        pass


class FeatureMetric(Metric):
    extractor_model_name: str = "default"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class EmbedderInterface(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def extract_features_from_files(self, *args, **kwargs):
        pass

    @abstractmethod
    def _extract_features_dataloader(self, dataloader, **kwargs):
        pass
