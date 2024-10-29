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
    TIME_SERIES = "time_series"
    IMAGE = "image"
    TABULAR = "tabular"
    TEXT = "text"


class ValidationTypes(str, StrEnum):
    INPUT = "input_val"
    SYNTH = "synthesis_val"


class MetricGoal(str, StrEnum):
    VALIDITY = "validity"
    QUALITY = "quality"
    PRIVACY = "privacy"
    UTILITY = "utility"


################################################################
####################### SYNTHETIC GROUP ########################
################################################################
class SyntheticMetricGroups(str, StrEnum):
    FEATURE = "feature"
    DATA = "data"


################################################################
####################### INPUT GROUP ############################
################################################################
class InputMetricGroups(str, StrEnum):
    QUALITY = "data"
    ANNOTATION = "annotation"


# ============ Annotation Types ===============
class AnnotationType(str, StrEnum):
    LABEL = "label"
    MASK = "segmentation_mask"
    BBOX = "bounding_box"
    KEYPOINTS = "keypoints"


# ===================== OTHER CONFIGS =====================
class EvaluationLevel(str, StrEnum):
    DATASET = "dataset"
    INSTANCE = "instance"


class ReferenceType(str, StrEnum):
    DATASET = "dataset"
    INSTANCE = "instance"
    NONE = "none"


class OutputsTypes(str, StrEnum):
    PLOT = "plot"
    NUMERIC = "numeric"
    BOOL = "boolean"
    ARRAY = "array"
    STRING = "string"
    KEY_VAL = "key_value"  # key-value pair (dict) str -> float | int | str
    KEY_ARRAY = "key_array"  # key-values pair (dict) str -> list of float | int | str
