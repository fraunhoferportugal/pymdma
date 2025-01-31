import matplotlib.pyplot as plt
import numpy as np
import pytest
from torch.utils.data import DataLoader

from pymdma.image.data.simple_dataset import SimpleDataset
from pymdma.image.measures import input_val
from pymdma.image.measures.input_val.annotation import coco as ann
from pymdma.image.measures.synthesis_val import ImprovedPrecision


@pytest.mark.parametrize(
    "area_range",
    [
        (1000, 200000),
        (10, 20),
    ],
)
def test_ann_bbox__area(coco_bbox_dataset, area_range):
    ann_val = ann.AnnotationCorrectness(
        mask_area_range=area_range,
    )
    metric_result = ann_val.compute(coco_bbox_dataset)
    assert metric_result.dataset_level is not None, "Dataset level is None"
    dataset_level, _ = metric_result.value
    assert isinstance(dataset_level, dict), "Dataset level is not of evaluation level type"
    assert "annots_mask_oob" in dataset_level and len(dataset_level["annots_mask_oob"]) > 0, dataset_level


@pytest.mark.parametrize(
    "valid_names, invalid",
    [
        (["Right Lung", "Left Lung"], False),
        (["Right Lung"], True),
    ],
)
def test_ann_bbox__valid_names(coco_bbox_dataset, valid_names, invalid):
    ann_val = ann.AnnotationCorrectness(
        valid_label_names=valid_names,
    )
    metric_result = ann_val.compute(coco_bbox_dataset)
    assert metric_result.dataset_level is not None, "Dataset level is None"
    dataset_level, _ = metric_result.value
    assert isinstance(dataset_level, dict), "Dataset level is not of evaluation level type"
    assert "invalid_labels" in dataset_level, dataset_level.keys()
    assert "annots_invalid_label" in dataset_level, dataset_level.keys()
    if invalid:
        assert len(dataset_level["invalid_labels"]) > 0, dataset_level
        assert len(dataset_level["annots_invalid_label"]) > 0, dataset_level
    else:
        assert len(dataset_level["invalid_labels"]) == 0, dataset_level
        assert len(dataset_level["annots_invalid_label"]) == 0, dataset_level


@pytest.mark.parametrize(
    "metric_cls",
    [
        input_val.CLIPIQA,
        input_val.BRISQUE,
    ],
)
def test_no_ref_batch_metrics(image_dataset, metric_cls):
    metric = metric_cls(same_size=True)
    images = [np.asarray(image.resize((256, 256))) for image in image_dataset]

    result = metric.compute(images)
    _, instance_level = result.value

    assert isinstance(instance_level, list), "Instance level is not a list"
    assert len(instance_level) == len(images), "Instance level length does not match input length"


@pytest.mark.parametrize(
    "metric_cls",
    [
        input_val.SSIM,
        input_val.MSSSIM,
    ],
)
def test_ref_batch_metrics(image_dataset, metric_cls):
    metric = metric_cls(same_size=True)
    images = [np.asarray(image.resize((256, 256))) for image in image_dataset]

    result = metric.compute(images, images)
    _, instance_level = result.value
    assert isinstance(instance_level, list), "Instance level is not a list"
    assert len(instance_level) == len(images), "Instance level length does not match input length"


####################################################################################################
################################# SYNTHETIC METRICS TESTS #########################################
####################################################################################################
@pytest.mark.parametrize(
    "extractor_name",
    [
        "inception_fid",
        "vgg16",
        "dino_vits8",
        "vit_b_16",
    ],
)
def test_extractor_models(image_feature_extractor, synth_image_filenames, extractor_name):
    extractor = image_feature_extractor(extractor_name)
    features = extractor.extract_features_from_files(synth_image_filenames)

    assert features.shape[0] == len(synth_image_filenames), "Feature length does not match input length"
    assert features.shape[1] > 0, "Empty second dimension"

    prec = ImprovedPrecision()
    result = prec.compute(features, features)
    assert result.dataset_level is not None and result.instance_level is not None, "Eval level is None"
    dataset_level, instance_level = result.value
    assert dataset_level > 0.90, "Dataset level is below threshold"
    assert all(inst == 1 for inst in instance_level), "Same image instance should be precise"


@pytest.mark.parametrize(
    "extractor_name, input_size",
    [
        ("inception_fid", (299, 299)),
        ("vgg16", (224, 224)),
        ("dino_vits8", (224, 224)),
        ("vit_b_16", (224, 224)),
    ],
)
def test_extractor_methods(
    image_feature_extractor,
    synth_image_filenames,
    image_transforms,
    extractor_name,
    input_size,
):
    extractor = image_feature_extractor(extractor_name)

    dataset = SimpleDataset(synth_image_filenames, image_transforms(input_size, extractor.interpolation))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    features_files = extractor.extract_features_from_files(synth_image_filenames)
    features_dataloader = extractor.extract_features_from_dataloader(dataloader, normalize=True)

    assert features_files.shape[0] == len(synth_image_filenames), "Feature length does not match input length"
    assert features_dataloader.shape[0] == len(synth_image_filenames), "Feature length does not match input length"

    assert (
        features_dataloader.mean() == features_files.mean()
    ), "Feature extraction from files and dataloader should be the same"
    assert (
        features_dataloader.std() == features_files.std()
    ), "Feature extraction from files and dataloader should be the same"

    prec = ImprovedPrecision()
    result = prec.compute(features_files, features_dataloader)
    dataset_level, instance_level = result.value
    assert dataset_level > 0.98, "Dataset level is below threshold"
    assert all(inst == 1 for inst in instance_level), "Same image instance should be precise"
