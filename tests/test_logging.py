import logging

from fastapi.testclient import TestClient

from triage_system.api import app


def test_request_log_contains_metadata_but_not_description(caplog):
    description = "private medical text that must not be logged"
    with caplog.at_level(logging.INFO, logger="triage_system.http"):
        with TestClient(app) as client:
            response = client.post("/predict", json={"description": description})

    assert response.status_code == 400
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "POST" in messages
    assert "/predict" in messages
    assert description not in messages
