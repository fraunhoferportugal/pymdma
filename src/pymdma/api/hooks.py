from loguru import logger

from ..image.models.features import ExtractorFactory as ImageFeatureExtractor
from ..time_series.utils.extract_features import FeatureExtractor as TimeSeriesFeatureExtractor


def load_models_hook(ml_models, device="cpu"):
    logger.info("Loading ml models")
    # feature extractors
    ml_models["dino_vits8"] = ImageFeatureExtractor.model_from_name("dino_vits8").to(device)
    ml_models["tsfel"] = TimeSeriesFeatureExtractor("tsfel", device)
    logger.info("Models loaded successfully")
