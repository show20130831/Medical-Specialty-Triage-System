from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from triage_system import api
from triage_system.model_assets import ModelAssets
from triage_system.monitoring import LENGTH_EDGES, SCORE_EDGES, bucket_labels
from triage_system.predictor import Prediction

TOP = (("Orthopedics", 0.9), ("Neurology", 0.05), ("General Medicine", 0.02))
REFERENCE = {
    "predictions": 2, "uncertain": 0, "label_counts": {"Orthopedics": 2},
    "score_histogram": dict(zip(bucket_labels(SCORE_EDGES), [0, 0, 0, 0, 2])),
    "length_histogram": dict(zip(bucket_labels(LENGTH_EDGES), [2, 0, 0, 0, 0])),
}


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


def _sample(name, labels=None):
    return api.METRICS.registry.get_sample_value(name, labels or {}) or 0


def test_metrics_expose_operational_counters_without_text_or_specialties(served):
    before = _sample("triage_requests_total", {"route": "/predict", "status_class": "2xx"})
    rejected = _sample("triage_requests_total", {"route": "/predict", "status_class": "4xx"})
    with TestClient(api.app) as client:
        client.post("/predict", json={"description": "private knee text"})
        client.post("/predict", json={"description": "   "})
        response = client.get("/metrics")
    body = response.text
    assert response.headers["content-type"].startswith("text/plain")
    assert _sample("triage_requests_total", {"route": "/predict", "status_class": "2xx"})         == before + 1
    assert _sample("triage_requests_total", {"route": "/predict", "status_class": "4xx"})         == rejected + 1
    assert "triage_request_duration_seconds_bucket" in body
    assert "private knee text" not in body
    assert not any(name in body for name in ("Orthopedics", "Neurology", "General Medicine"))


def test_unknown_paths_share_one_route_label():
    with TestClient(api.app) as client:
        client.get("/no/such/page/12345")
        body = client.get("/metrics").text
    assert _sample("triage_requests_total", {"route": "other", "status_class": "4xx"}) >= 1
    assert "12345" not in body


def test_drift_failure_never_changes_the_prediction_response(served, monkeypatch):
    api.DRIFT.set_reference(REFERENCE)
    failures = _sample("triage_drift_evaluation_failures_total")
    monkeypatch.setattr("triage_system.monitoring.bucket_index",
                        lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    try:
        with TestClient(api.app) as client:
            response = client.post("/predict", json={"description": "knee pain"})
    finally:
        api.DRIFT.set_reference(None)
    assert response.status_code == 200 and response.json()["label"] == "Orthopedics"
    assert _sample("triage_drift_evaluation_failures_total") == failures + 1


def test_failed_preload_keeps_health_up_and_model_unloaded(monkeypatch):
    from triage_system.model_loader import ModelLoadError

    api.METRICS.model_loaded.set(0)
    monkeypatch.setattr(api, "_loaded_model",
                        lambda: (_ for _ in ()).throw(ModelLoadError("missing")))
    api._preload()
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
    assert _sample("triage_model_loaded") == 0


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


def test_missing_bundle_directory_is_unavailable_not_an_error(monkeypatch, tmp_path):
    """Found by drill D1: an absent bundle returned 500 because ModelAssetError escaped."""
    monkeypatch.setenv("TRIAGE_MODEL_DIR", str(tmp_path / "missing_bundle"))
    api.reset_model_cache()
    api.METRICS.model_loaded.set(0)
    try:
        api._preload()
        with TestClient(api.app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 503
            response = client.post("/predict", json={"description": "knee pain"})
    finally:
        api.reset_model_cache()
    assert response.status_code == 503
    assert _sample("triage_model_loaded") == 0


def test_concurrent_callers_load_the_model_once(monkeypatch):
    """Preload and an early request must not load the model twice."""
    import threading
    import time

    calls = []

    def slow_load():
        calls.append(1)
        time.sleep(0.2)
        return object()

    api.reset_model_cache()
    monkeypatch.setattr(api, "_load_model", slow_load)
    try:
        threads = [threading.Thread(target=api._loaded_model) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        api.reset_model_cache()
    assert len(calls) == 1
