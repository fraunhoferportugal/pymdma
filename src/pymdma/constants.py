from enum import Enum
from typing import Union

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


################################################################
####################### SYNTHETIC GROUP ########################
################################################################
class SyntheticMetricGroups(str, StrEnum):
    FEATURE = "feature"
    DATA = "data"


# ============ Synthetic Metric Specific Goals ============
class SyntheticFeatureMetrics(str, StrEnum):
    FIDELITY = "fidelity"
    DIVERSITY = "diversity"
    AUTHENTICITY = "authenticity"  # indicates privacy
    PRIVACY = "privacy"
    UTILITY = "utility"  # TODO move utility to another group (not based on features)
    QUALITY = "quality"  # general quality


################################################################
####################### INPUT GROUP ############################
################################################################
class InputMetricGroups(str, StrEnum):
    QUALITY = "quality"
    ANNOTATION = "annotation"
    PRIVACY = "privacy"


# ============ Input Metric Specific Goals ===============
class InputQualityMetrics(str, StrEnum):
    CONTRAST = "contrast"
    BRIGHTNESS = "brightness"
    COLORFULNESS = "colorfulness"
    SHARPNESS = "sharpness"
    PERCEPTUAL_QUALITY = "perceptual_quality"
    NOISE = "noise"
    SIMILARITY = "similarity"
    UNIFORMITY = "uniformity"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    OTHER = "other"


class InputPrivacyMetrics(str, StrEnum):
    PRIVACY = "privacy"
    ANONYMITY = "anonymity"
    CONFIDENTIALITY = "confidentiality"
    NON_REPUDIATION = "non_repudiation"
    UNIQUENESS = "uniqueness"


class InputAnnotationMetrics(str, StrEnum):
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    UNIQUENESS = "uniqueness"


# ============ Annotation Types ===============
class AnnotationType(str, StrEnum):
    LABEL = "label"
    MASK = "segmentation_mask"
    BBOX = "bounding_box"
    KEYPOINTS = "keypoints"


def valid_subclass(goal, subgoal):
    dependency_map = {
        InputMetricGroups.QUALITY: InputQualityMetrics,
        InputMetricGroups.ANNOTATION: InputAnnotationMetrics,
        InputMetricGroups.PRIVACY: InputPrivacyMetrics,
        SyntheticMetricGroups.FEATURE: SyntheticFeatureMetrics,
    }
    return goal in dependency_map and dependency_map[goal].has_value(subgoal)


# ===================== OTHER CONFIGS =====================
class EvaluationLevel(str, StrEnum):
    DATASET = "dataset"
    INSTANCE = "instance"
    # FEATURE = "feature_wise"


class ReferenceType(str, StrEnum):
    DATASET = "dataset"
    INSTANCE = "instance"
    # FEATURE = "feature"
    NONE = "none"


class OutputsTypes(str, StrEnum):
    PLOT = "plot"
    NUMERIC = "numeric"
    BOOL = "boolean"
    ARRAY = "array"
    STRING = "string"
    KEY_VAL = "key_value"  # key-value pair (dict) str -> float | int | str
    KEY_ARRAY = "key_array"  # key-values pair (dict) str -> list of float | int | str


METRIC_GOALS = Union[InputQualityMetrics, InputPrivacyMetrics, InputAnnotationMetrics, SyntheticFeatureMetrics]
