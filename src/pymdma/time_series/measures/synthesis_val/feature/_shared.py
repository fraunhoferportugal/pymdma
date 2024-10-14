from pymdma.general.measures.external.piq import FrechetDistance, GeometryScore, MultiScaleIntrinsicDistance
from pymdma.general.measures.prd import PrecisionRecallDistribution
from pymdma.general.measures.prdc import Authenticity, Coverage, Density, ImprovedPrecision, ImprovedRecall

# Set the default model name for the feature extractors as inception is not currently supported for TS
FrechetDistance.extractor_model_name = "default"
MultiScaleIntrinsicDistance.extractor_model_name = "default"

__all__ = [
    "ImprovedPrecision",
    "ImprovedRecall",
    "Authenticity",
    "Density",
    "Coverage",
    "FrechetDistance",
    "GeometryScore",
    "MultiScaleIntrinsicDistance",
    "PrecisionRecallDistribution",
]
