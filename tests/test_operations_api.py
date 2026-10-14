from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from triage_system import api
from triage_system.model_assets import ModelAssets
from triage_system.predictor import Prediction

TOP = (("Orthopedics", 0.9), ("Neurology", 0.05), ("General Medicine", 0.02))


@pytest.fixture
def served(monkeypatch, tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        '{"git": {"commit": "abc123"}, "created_utc": "20261013T164433Z", '
        '"base_model": {"name": "base"}}', encoding="utf-8")
    loaded = SimpleNamespace(
        assets=ModelAssets(tmp_path, ("A",), temperature=0.95, uncertain_threshold=0.65),
        weights_sha256="f" * 64)
    monkeypatch.setattr(api, "_loaded_model", lambda: loaded)
    monkeypatch.setattr(api, "predict_description",
                        lambda model, text: Prediction("Orthopedics", 0.9, False, TOP))
    return loaded


def test_version_identifies_weights_commit_and_calibration(served):
    with TestClient(api.app) as client:
        body = client.get("/version").json()
    assert body["model"] == {
        "weights_sha256": "f" * 64, "training_commit": "abc123",
        "trained_utc": "20261013T164433Z", "base_model": "base",
        "temperature": 0.95, "uncertain_threshold": 0.65,
    }
    assert body["service_version"]


def test_metrics_count_predictions_and_statuses_without_text(served):
    with TestClient(api.app) as client:
        before = client.get("/metrics").json()
        client.post("/predict", json={"description": "private knee text"})
        client.post("/predict", json={"description": "   "})
        after = client.get("/metrics").json()
    assert after["predictions"] - before["predictions"] == 1
    assert after["label_counts"]["Orthopedics"] - before["label_counts"]["Orthopedics"] == 1
    assert after["predict_status_counts"]["2xx"] - before["predict_status_counts"].get("2xx", 0) == 1
    assert after["predict_status_counts"]["4xx"] - before["predict_status_counts"].get("4xx", 0) == 1
    assert after["model_weights_sha256"] == "f" * 64
    assert "private knee text" not in str(after)


def test_description_limit_matches_training_distribution():
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"description": "x" * 1001})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_too_long"


def test_oversized_body_is_rejected_before_parsing():
    with TestClient(api.app) as client:
        response = client.post("/predict", content=b"x" * 20_000,
                               headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_body_without_length_is_rejected():
    def chunks():
        yield b'{"description": "knee pain"}'

    with TestClient(api.app) as client:
        response = client.post("/predict", content=chunks(),
                               headers={"content-type": "application/json"})
    assert response.status_code == 411
