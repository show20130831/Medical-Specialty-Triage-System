from fastapi.testclient import TestClient

from triage_system.api import app


def test_health_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
