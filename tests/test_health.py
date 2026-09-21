from fastapi.testclient import TestClient

from triage_system.api import app


def test_health_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_ready_when_model_loads(monkeypatch):
    from triage_system import api

    monkeypatch.setattr(api, "_loaded_model", lambda: object())
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_service_unavailable_when_model_is_not_configured(monkeypatch):
    from triage_system import api
    from triage_system.model_loader import ModelLoadError

    monkeypatch.setattr(api, "_loaded_model", lambda: (_ for _ in ()).throw(ModelLoadError("model unavailable")))
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "model unavailable"}


def test_web_page_is_served_from_same_origin():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Medical Specialty Triage System" in response.text
    assert "fetch('/predict'" in response.text


def test_predict_returns_label_and_score(monkeypatch):
    from triage_system import api
    from triage_system.predictor import Prediction

    monkeypatch.setattr(api, "_loaded_model", lambda: object())
    monkeypatch.setattr(api, "predict_description", lambda model, text: Prediction("Neurology", 0.91))
    with TestClient(app) as client:
        response = client.post("/predict", json={"description": "persistent headache"})

    assert response.status_code == 200
    assert response.json() == {"label": "Neurology", "score": 0.91}


def test_predict_rejects_blank_description():
    with TestClient(app) as client:
        response = client.post("/predict", json={"description": "  "})

    assert response.status_code == 400
    assert "non-blank" in response.json()["detail"]
