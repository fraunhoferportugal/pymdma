import pytest

from tests._utils import prune_params


@pytest.mark.parametrize(
    "modality, validation_domain, reference_type, metric_category",
    [
        ("image", "input_val", "none", ["data"]),
        ("time_series", "input_val", "none", ["data"]),
    ],
)
def test_empty_eval_level(test_client, modality, validation_domain, reference_type, metric_category):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_category": metric_category,
        },
    )
    response = test_client.get(f"/metrics/{modality}", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"


@pytest.mark.parametrize(
    "modality, validation_domain, reference_type, metric_category, code",
    [
        ("image", "input_val", "dataset", ["data"], 501),
        ("image", "input_val", "dataset", ["data"], 501),
        ("time_series", "input_val", "", ["data"], 422),
        ("tabular", "input_val", "dat", ["data"], 422),
        # ("text", "input_val", "dataset", [""], 422),
        # ("text", "synth", "dataset", ["feature"], 422),
        # ("text", "input_val", "dataset", ["quality"], 501),
        ("image", "input_val", "none", ["annotation"], 400),
    ],
)
def test_dataset_eval_fail(test_client, modality, validation_domain, reference_type, metric_category, code):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_category": metric_category,
        },
    )
    response = test_client.get(f"/metrics/{modality}", params=params)
    assert response.status_code == code, response.json()
    assert "msg" in response.json() or "detail" in response.json(), "Missing error type"
    assert len(response.json()) > 0, "Empty response"


@pytest.mark.parametrize(
    "validation_domain, evaluation_level, reference_type, metric_category",
    [
        ("input_val", "instance", "none", ["data"]),
        ("input_val", "instance", "instance", ["data"]),
        ("synthesis_val", "dataset", "dataset", ["feature"]),
    ],
)
def test_dataset_image_pass(test_client, validation_domain, evaluation_level, reference_type, metric_category):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "evaluation_level": evaluation_level,
            "metric_category": metric_category,
        },
    )
    response = test_client.get("/metrics/image", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"


@pytest.mark.parametrize(
    "validation_domain, evaluation_level, ann_file",
    [
        ("input_val", "instance", "COCO_annotation_example_bb_exp.json"),
        ("input_val", "instance", "COCO_annotation_example_mask_exp.json"),
    ],
)
def test_dataset_image_ann_pass(test_client, validation_domain, evaluation_level, ann_file):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": "none",
            "evaluation_level": evaluation_level,
            "metric_category": ["annotation", "data"],
            "annotation_file": ann_file,
        },
    )
    response = test_client.get("/metrics/image", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"
    label_field = False
    if "instance_level" in response.json() and response.json()["instance_level"] is not None:
        label_field |= "label_metrics" in response.json()["instance_level"]
    if "dataset_level" in response.json() and response.json()["dataset_level"] is not None:
        label_field |= "label_metrics" in response.json()["dataset_level"]
    assert label_field, "Missing label metrics"


@pytest.mark.parametrize(
    "validation_domain, evaluation_level, reference_type, metric_category",
    [
        ("input_val", "instance", "none", ["data"]),
        ("synthesis_val", "dataset", "dataset", ["feature"]),
    ],
)
def test_dataset_time_series_pass(test_client, validation_domain, evaluation_level, reference_type, metric_category):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "evaluation_level": evaluation_level,
            "metric_category": metric_category,
        },
    )
    response = test_client.get("/metrics/time_series", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"


@pytest.mark.parametrize(
    "validation_domain, evaluation_level, reference_type, metric_category",
    [
        ("input_val", "dataset", "none", ["data"]),
        ("synthesis_val", "dataset", "dataset", ["data"]),
    ],
)
def test_dataset_tabular_pass(test_client, validation_domain, evaluation_level, reference_type, metric_category):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "evaluation_level": evaluation_level,
            "metric_category": metric_category,
        },
    )
    response = test_client.get("/metrics/tabular", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"


@pytest.mark.parametrize(
    "modality, validation_domain, reference_type, metric_category, metric_group",
    [
        ("image", "input_val", "none", ["data"], ["quality"]),
        ("image", "synthesis_val", "dataset", ["feature"], ["quality", "privacy"]),
        ("tabular", "input_val", "none", ["data"], ["quality", "privacy"]),
    ],
)
def test_dataset_metric_group_pass(
    test_client, modality, validation_domain, reference_type, metric_category, metric_group
):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_category": metric_category,
            "metric_group": metric_group,
        },
    )
    response = test_client.get(f"/metrics/{modality}", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"


@pytest.mark.parametrize(
    "modality, validation_domain, reference_type, metric_category, metric_group, code",
    [
        ("image", "input_val", "none", ["data"], ["privacy"], 501),
        ("time_series", "synthesis_val", "dataset", ["data"], ["privacy"], 501),
        ("tabular", "input_val", "none", ["feature"], ["privacy", "quality"], 422),
    ],
)
def test_dataset_metric_group_fail(
    test_client,
    modality,
    validation_domain,
    reference_type,
    metric_category,
    metric_group,
    code,
):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_category": metric_category,
            "metric_group": metric_group,
        },
    )
    response = test_client.get(f"/metrics/{modality}", params=params)
    assert response.status_code == code, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "msg" in response.json() or "detail" in response.json(), "Missing error type"


###############################################################################################################
################################### TEST SPECIFIC METRICS #####################################################
###############################################################################################################
@pytest.mark.parametrize(
    "modality, validation_domain, metric_names, reference_type",
    [
        ("image", "input_val", ["Tenengrad", "DOM"], "none"),
        ("image", "input_val", ["PSNR", "SSIM"], "instance"),
        ("image", "synthesis_val", ["ImprovedPrecision", "Authenticity"], "dataset"),
        ("time_series", "input_val", ["SNR", "Uniqueness"], "none"),
        ("time_series", "synthesis_val", ["ImprovedPrecision", "WassersteinDistance"], "dataset"),
        ("tabular", "input_val", ["KAnonymityScore"], "none"),
        ("tabular", "synthesis_val", ["ImprovedPrecision", "ImprovedRecall"], "dataset"),
        # ("text", "input_val", ["Identifiability"], "none"),
    ],
)
def test_specific_metric_pass(test_client, modality, validation_domain, metric_names, reference_type):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_names": metric_names,
        },
    )
    response = test_client.get(f"/metrics/{modality}/specific-metrics", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"


@pytest.mark.parametrize(
    "modality, validation_domain, metric_names, reference_type, code",
    [
        ("image", "input_val", ["Tene", "DOM"], "none", 404),
        ("image", "inpu", ["PSNR", "SSIM"], "instance", 422),
        ("time_series", "input_val", ["SNR", "Uniqueness"], "no", 422),
        ("time_series", "synthesis_val", ["WassersteinDist"], "dataset", 404),
    ],
)
def test_specific_metric_fail(test_client, modality, validation_domain, metric_names, reference_type, code):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_names": metric_names,
        },
    )
    response = test_client.get(f"/metrics/{modality}/specific-metrics", params=params)
    assert response.status_code == code, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "msg" in response.json() or "detail" in response.json(), "Missing error type"


@pytest.mark.parametrize(
    "validation_domain, metric_names, reference_type, ann_file, code",
    [
        ("input_val", ["AnnotationCorrectness", "AnnotationUniqueness"], "none", None, 400),
        (
            "input_val",
            ["AnnotationCorrectness", "AnnotationUnique"],
            "none",
            "COCO_annotation_example_mask_exp.json",
            404,
        ),
    ],
)
def test_specific_metric_ann_fail(test_client, validation_domain, metric_names, reference_type, ann_file, code):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_names": metric_names,
            "annotation_file": ann_file,
        },
    )
    response = test_client.get(f"/metrics/image/specific-metrics", params=params)
    assert response.status_code == code, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "msg" in response.json() or "detail" in response.json(), "Missing error type"


@pytest.mark.parametrize(
    "validation_domain, metric_names, reference_type, ann_file",
    [
        (
            "input_val",
            ["AnnotationCorrectness", "AnnotationUniqueness"],
            "none",
            "COCO_annotation_example_mask_exp.json",
        ),
        (
            "input_val",
            ["AnnotationCorrectness", "AnnotationUniqueness", "DatasetCompletness"],
            "none",
            "COCO_annotation_example_bb_exp.json",
        ),
    ],
)
def test_specific_metric_ann_pass(test_client, validation_domain, metric_names, reference_type, ann_file):
    params = prune_params(
        {
            "validation_domain": validation_domain,
            "reference_type": reference_type,
            "metric_names": metric_names,
            "annotation_file": ann_file,
        },
    )
    response = test_client.get(f"/metrics/image/specific-metrics", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    assert "instance_level" in response.json() or "dataset_level" in response.json(), "Missing evaluation level"
    label_field = False
    if "instance_level" in response.json() and response.json()["instance_level"] is not None:
        label_field |= "label_metrics" in response.json()["instance_level"]
    if "dataset_level" in response.json() and response.json()["dataset_level"] is not None:
        label_field |= "label_metrics" in response.json()["dataset_level"]
    assert label_field, "Missing label metrics"
