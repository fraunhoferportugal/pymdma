import pytest


def test_root(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome from the API."}


def test_healthcheck(test_client):
    response = test_client.get("/api/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"message": "API is alive!"}


@pytest.mark.parametrize(
    "data_modalities, validation_domains, metric_categorys",
    [
        (["image"], ["input_val", "synthesis_val"], []),
        (["image"], ["input_val"], ["data", "annotation"]),
        (["image"], ["synthesis_val"], ["feature"]),
        (["tabular"], ["input_val", "synthesis_val"], []),
        (["tabular"], ["input_val", "synthesis_val"], ["data", "feature"]),
        (["time_series"], ["input_val", "synthesis_val"], []),
        (["time_series"], ["input_val", "synthesis_val"], ["data"]),
        (["time_series"], ["input_val", "synthesis_val"], ["feature"]),
        (["image", "tabular", "time_series"], ["input_val", "synthesis_val"], []),
    ],
)
def test_metric_info_pass(test_client, data_modalities, validation_domains, metric_categorys):
    params = {
        "data_modalities": data_modalities,
        "validation_domains": validation_domains,
        "metric_categorys": metric_categorys,
    }
    response = test_client.get("/metrics/info", params=params)
    assert response.status_code == 200, response.json()
    assert len(response.json()) > 0, "Empty response"
    # check parameters are in reponse names
    assert len(metric_categorys) == 0 or all(
        any(metric_category in entry for metric_category in metric_categorys) for entry in response.json()
    ), "Missing metric group"
    assert all(
        any(modality in entry for modality in data_modalities) for entry in response.json()
    ), "Missing data modality"
    assert all(
        any(validation_domain in entry for validation_domain in validation_domains) for entry in response.json()
    ), "Missing validation type"


@pytest.mark.parametrize(
    "data_modalities, validation_domains, metric_categorys, code",
    [
        ([""], [], [], 422),
        (["imag"], ["input_val", "synthesis_val"], [], 422),
        (["image"], ["input", "synth"], [], 422),
        (["image"], ["synthesis_val"], ["data"], 404),
    ],
)
def test_metric_info_fail(test_client, data_modalities, validation_domains, metric_categorys, code):
    params = {
        "data_modalities": data_modalities,
        "validation_domains": validation_domains,
        "metric_categorys": metric_categorys,
    }
    response = test_client.get("/metrics/info", params=params)
    assert response.status_code == code, response.json()
    assert code == 422 or "msg" in response.json(), f"Missing message in response {response.json()}"
    assert code == 422 or "type" in response.json(), f"Missing type in response {response.json()}"
