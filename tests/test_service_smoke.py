"""End-to-end service check with a tiny randomly initialised bundle.

The real weights are private, so CI builds a bundle with the same file layout and
label contract but a two-layer, 16-dimension BERT. It exercises asset validation, offline
loading, tokenisation, inference, calibration, and every route. Predictions are
meaningless by design; only the contract is checked.
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

from triage_system import api  # noqa: E402
from triage_system.model_assets import SPECIALTIES  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "knee", "pain", "eye", "surgery", "."]


@pytest.fixture
def tiny_bundle(tmp_path, monkeypatch):
    labels = sorted(SPECIALTIES)
    (tmp_path / "vocab.txt").write_text("\n".join(VOCAB), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    tokenizer.save_pretrained(tmp_path)
    config = transformers.BertConfig(
        vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=512, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)},
        problem_type="single_label_classification")
    torch.manual_seed(0)
    transformers.BertForSequenceClassification(config).save_pretrained(
        tmp_path, safe_serialization=True)
    (tmp_path / "training_config.json").write_text(
        json.dumps({"input_type": "description", "max_length": 512}), encoding="utf-8")
    (tmp_path / "calibration.json").write_text(
        json.dumps({"temperature": 1.0, "uncertain_threshold": 0.65}), encoding="utf-8")
    monkeypatch.setenv("TRIAGE_MODEL_DIR", str(tmp_path))
    api._loaded_model.cache_clear()
    yield tmp_path
    api._loaded_model.cache_clear()


def test_service_serves_a_real_bundle_end_to_end(tiny_bundle):
    with TestClient(api.app) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        version = client.get("/version").json()["model"]
        assert len(version["weights_sha256"]) == 64 and version["uncertain_threshold"] == 0.65

        response = client.post("/predict", json={"description": "Knee pain after surgery."})
        body = response.json()
        assert response.status_code == 200
        assert body["label"] in SPECIALTIES and len(body["top"]) == 3
        assert body["uncertain"] == (body["score"] < 0.65)
        assert sum(item["score"] for item in body["top"]) <= 1.0 + 1e-6

        metrics = client.get("/metrics").json()
        assert metrics["predictions"] >= 1
        assert metrics["model_weights_sha256"] == version["weights_sha256"]
