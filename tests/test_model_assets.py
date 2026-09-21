import json
from pathlib import Path

import pytest

from triage_system.model_assets import (
    REQUIRED_FILES,
    SPECIALTIES,
    ModelAssetError,
    validate_model_assets,
)


@pytest.fixture
def export(tmp_path):
    labels = sorted(SPECIALTIES)
    config = {
        "model_type": "bert", "architectures": ["BertForSequenceClassification"],
        "problem_type": "single_label_classification", "max_position_embeddings": 512,
        "id2label": {str(i): label for i, label in enumerate(labels)},
        "label2id": {label: i for i, label in enumerate(labels)},
    }
    for name in REQUIRED_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    # Placeholder only: validation must not interpret this as real model weights.
    (tmp_path / "model.safetensors").write_bytes(b"synthetic-placeholder")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "training_config.json").write_text(
        json.dumps({"input_type": "description", "max_length": 512}), encoding="utf-8"
    )
    return tmp_path


def test_valid_export_does_not_read_weights(export, monkeypatch):
    original = Path.open
    def guarded(self, *args, **kwargs):
        if self.name == "model.safetensors":
            raise AssertionError("Weight contents must not be read")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    assets = validate_model_assets(export)
    assert assets.directory == export.resolve()
    assert set(assets.labels) == SPECIALTIES
    assert assets.max_length == 512


@pytest.mark.parametrize("name", REQUIRED_FILES)
@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_asset(export, name, empty):
    if empty:
        (export / name).write_bytes(b"")
    else:
        (export / name).unlink()
    with pytest.raises(ModelAssetError, match=name):
        validate_model_assets(export)


@pytest.mark.parametrize("path", ["", "   ", "nonexistent-export"])
def test_invalid_directory(path):
    with pytest.raises(ModelAssetError):
        validate_model_assets(path)


@pytest.mark.parametrize("filename,key,value", [
    ("training_config.json", "input_type", "transcription"),
    ("training_config.json", "max_length", 256),
    ("config.json", "max_position_embeddings", 128),
    ("config.json", "model_type", "roberta"),
    ("config.json", "id2label", {"0": "Urology"}),
    ("config.json", "label2id", {}),
])
def test_incompatible_metadata(export, filename, key, value):
    path = export / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelAssetError):
        validate_model_assets(export)


@pytest.mark.parametrize("content", ["invalid synthetic JSON", "[]"])
def test_invalid_json_has_actionable_error(export, content):
    (export / "config.json").write_text(content, encoding="utf-8")
    with pytest.raises(ModelAssetError, match="config.json") as exc:
        validate_model_assets(export)
    assert content not in str(exc.value)


def test_swapped_label_ids_are_rejected(export):
    path = export / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    a, b = list(config["label2id"])[:2]
    config["label2id"][a], config["label2id"][b] = 1, 0
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelAssetError, match="inverse"):
        validate_model_assets(export)
