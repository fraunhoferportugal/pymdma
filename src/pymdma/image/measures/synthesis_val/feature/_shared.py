from pymdma.general.measures.external.piq import FrechetDistance, GeometryScore, MultiScaleIntrinsicDistance
from pymdma.general.measures.prd import PrecisionRecallDistribution
from pymdma.general.measures.prdc import Authenticity, Coverage, Density, ImprovedPrecision, ImprovedRecall

FrechetDistance.extractor_model_name = "inception_fid"

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
