from enum import Enum

METRICS_PACKAGE_NAME = "measures"
SEED = 42


class StrEnum(Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class DataModalities(str, StrEnum):
    """Types of data supported by the library."""

    TIME_SERIES = "time_series"
    IMAGE = "image"
    TABULAR = "tabular"
    TEXT = "text"


class ValidationDomain(str, StrEnum):
    """Domains of validation.

    Input data metrics or Synthetic data metrics.
    """

    INPUT = "input_val"
    SYNTH = "synthesis_val"


class SyntheticCategories(str, StrEnum):
    """Synthetic metric categories based on compute requirements."""

    FEATURE = "feature"
    DATA = "data"


class InputCategories(str, StrEnum):
    """Input Metric categories based on compute requirements."""

    QUALITY = "data"
    ANNOTATION = "annotation"


class MetricGroup(str, StrEnum):
    """Metric group indetifiers."""

    VALIDITY = "validity"
    QUALITY = "quality"
    PRIVACY = "privacy"
    UTILITY = "utility"


class EvaluationLevel(str, StrEnum):
    """Types of the output returned by a metric."""

    DATASET = "dataset"
    INSTANCE = "instance"


class ReferenceType(str, StrEnum):
    """Types of reference of a metric."""

    DATASET = "dataset"
    INSTANCE = "instance"
    NONE = "none"


class OutputsTypes(str, StrEnum):
    """Supported output types by the MetricResult class."""

    PLOT = "plot"
    NUMERIC = "numeric"
    BOOL = "boolean"
    ARRAY = "array"
    STRING = "string"
    KEY_VAL = "key_value"  # key-value pair (dict) str -> float | int | str
    KEY_ARRAY = "key_array"  # key-values pair (dict) str -> list of float | int | str


class AnnotationType(str, StrEnum):
    """Types of annotations supported by the library."""

    LABEL = "label"
    MASK = "segmentation_mask"
    BBOX = "bounding_box"
    KEYPOINTS = "keypoints"
