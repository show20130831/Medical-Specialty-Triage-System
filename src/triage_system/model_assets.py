"""Validate the local description-model export without importing ML libraries."""

import json
from dataclasses import dataclass
from pathlib import Path

MAX_LENGTH = 512
SPECIALTIES = frozenset({
    "Cardiovascular & Pulmonary", "ENT & Ophthalmology", "Gastroenterology",
    "General Medicine", "Neurology", "OB-GYN", "Orthopedics", "Urology",
})
REQUIRED_FILES = (
    "model.safetensors", "config.json", "tokenizer.json",
    "tokenizer_config.json", "training_config.json",
)


class ModelAssetError(ValueError):
    """The local export does not satisfy this application's model contract."""


@dataclass(frozen=True)
class ModelAssets:
    directory: Path
    labels: tuple[str, ...]
    max_length: int = MAX_LENGTH
    temperature: float = 1.0
    uncertain_threshold: float | None = None


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ModelAssetError(f"Cannot read valid UTF-8 JSON from {path.name}.") from None
    if not isinstance(value, dict):
        raise ModelAssetError(f"{path.name} must contain a JSON object.")
    return value


def validate_model_assets(model_dir: str | Path) -> ModelAssets:
    """Check metadata and file presence; never read or validate weight bytes."""
    if isinstance(model_dir, str) and not model_dir.strip():
        raise ModelAssetError("Provide a non-empty local model directory.")
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ModelAssetError("Model directory does not exist; provide a local export directory.")
    for name in REQUIRED_FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ModelAssetError(f"Required asset is missing or empty: {name}.")

    config = _read_object(directory / "config.json")
    training = _read_object(directory / "training_config.json")
    _read_object(directory / "tokenizer_config.json")
    if (config.get("model_type") != "bert"
            or config.get("architectures") != ["BertForSequenceClassification"]
            or config.get("problem_type") != "single_label_classification"):
        raise ModelAssetError("config.json must describe a single-label BERT classifier.")
    if training.get("input_type") != "description":
        raise ModelAssetError("training_config.json must declare description input.")
    if type(training.get("max_length")) is not int or training["max_length"] != MAX_LENGTH:
        raise ModelAssetError("training_config.json must declare max_length=512.")
    capacity = config.get("max_position_embeddings")
    if type(capacity) is not int or capacity < MAX_LENGTH:
        raise ModelAssetError("Model position capacity must support 512 tokens.")

    id2label = config.get("id2label")
    label2id = config.get("label2id")
    if not isinstance(id2label, dict) or set(id2label) != {str(i) for i in range(8)}:
        raise ModelAssetError("id2label must contain exactly IDs 0 through 7.")
    labels = tuple(id2label[str(i)] for i in range(8))
    if any(not isinstance(label, str) for label in labels) or set(labels) != SPECIALTIES:
        raise ModelAssetError("id2label must contain the eight supported specialties once each.")
    if (not isinstance(label2id, dict) or set(label2id) != SPECIALTIES
            or any(type(label2id[label]) is not int or label2id[label] != i
                   for i, label in enumerate(labels))):
        raise ModelAssetError("label2id must be the exact inverse of id2label.")
    temperature, threshold = _read_calibration(directory / "calibration.json")
    return ModelAssets(directory=directory, labels=labels,
                       temperature=temperature, uncertain_threshold=threshold)


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


def _read_calibration(path: Path) -> tuple[float, float | None]:
    """Optional temperature and abstain threshold produced by the evaluation report."""
    if not path.is_file():
        return 1.0, None
    calibration = _read_object(path)
    temperature = calibration.get("temperature")
    threshold = calibration.get("uncertain_threshold")
    if not _is_number(temperature) or not 0 < temperature <= 100:
        raise ModelAssetError("calibration.json temperature must be a positive number.")
    if threshold is not None and (not _is_number(threshold) or not 0 <= threshold < 1):
        raise ModelAssetError("calibration.json uncertain_threshold must be in [0, 1).")
    return float(temperature), None if threshold is None else float(threshold)
