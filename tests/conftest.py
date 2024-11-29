"""conftest.py contains configuration for pytest.

Configuration file for tests in test/ and scripts/ folders.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pymdma.api.run_api import app
from pymdma.config import data_dir
from pymdma.image.models.features import ExtractorFactory as ImageFeatureExtractor
from pymdma.time_series.data.simple_dataset import _read_sig_file
from pymdma.time_series.input_layer import _get_data_files_path
from pymdma.time_series.utils.extract_features import FeatureExtractor as TimeSeriesFeatureExtractor

MODALITIES = ["image", "tabular", "time_series"]
VALIDATION_TYPES = ["input_val", "synthesis_val"]


@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as test_client:
        yield test_client


# ###################################################################################################
# ##################################### Image Fixtures ##############################################
# ###################################################################################################
@pytest.fixture(scope="function")
def coco_bbox_dataset():
    return json.load(
        (Path(data_dir) / "test/image/input_val/annotations/COCO_annotation_example_bb_exp.json").open("r"),
    )


@pytest.fixture(scope="function")
def coco_mask_dataset():
    return json.load(
        (Path(data_dir) / "test/image/input_val/annotations/COCO_annotation_example_mask_exp.json").open("r"),
    )


@pytest.fixture(scope="module")
def image_dataset():
    images = list((Path(data_dir) / "test/image/input_val/dataset").rglob("*.jpg"))
    return [Image.open(image).convert("RGB") for image in images]


@pytest.fixture(scope="module")
def synth_image_filenames():
    return list((Path(data_dir) / "test/image/synthesis_val/dataset").rglob("*.png"))


@pytest.fixture(scope="module")
def image_feature_extractor():
    def get_extractor(name):
        return ImageFeatureExtractor.model_from_name(name)

    return get_extractor


# ###################################################################################################
# ################################## Time-Series Fixtures ###########################################
# ###################################################################################################


@pytest.fixture(scope="module")
def ts_dataset():
    file_paths = _get_data_files_path(Path(data_dir) / "test/time_series/input_val/dataset")
    signals = [_read_sig_file(signal_path) for signal_path in file_paths]
    return signals


@pytest.fixture(scope="module")
def synth_ts_filenames():
    return _get_data_files_path(Path(data_dir) / "test/time_series/input_val/dataset")


@pytest.fixture()
def ts_feature_extractor():
    def get_extractor(name):
        return TimeSeriesFeatureExtractor(name)

    return get_extractor


@pytest.fixture
def sample_distribution():
    def _sample_distribution(shape: Tuple[int], sigma: float = 1.0, mu: float = 0.0):
        np.random.seed(0)
        return np.random.randn(*shape) * sigma + mu

    return _sample_distribution
