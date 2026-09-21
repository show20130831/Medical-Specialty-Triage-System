"""Load a validated local model on CPU, independently of the HTTP service."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage_system.model_assets import ModelAssets, validate_model_assets


class ModelLoadError(RuntimeError):
    """Inference dependencies or model contents could not be loaded."""


@dataclass(frozen=True)
class LoadedModel:
    assets: ModelAssets
    tokenizer: Any
    model: Any


def _get_auto_classes():
    # Lazy imports keep the health endpoint and lightweight CI independent of ML packages.
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ModelLoadError(
            "Install CPU PyTorch and the inference extra as described in README.md."
        ) from None
    return AutoTokenizer, AutoModelForSequenceClassification


def load_local_model(model_dir: str | Path) -> LoadedModel:
    """Load existing assets only; do not download, train, or run predictions."""
    assets = validate_model_assets(model_dir)
    tokenizer_class, model_class = _get_auto_classes()
    try:
        tokenizer = tokenizer_class.from_pretrained(
            str(assets.directory), local_files_only=True, trust_remote_code=False,
            model_max_length=assets.max_length,
        )
        model = model_class.from_pretrained(
            str(assets.directory), local_files_only=True, trust_remote_code=False,
            use_safetensors=True,
        )
        model = model.to("cpu")
        model.eval()
    except (OSError, ValueError, RuntimeError, ImportError):
        raise ModelLoadError(
            "Failed to load the local export on CPU. Check asset integrity, "
            "available memory, and the documented inference dependency versions."
        ) from None

    expected_ids = dict(enumerate(assets.labels))
    expected_labels = {label: i for i, label in enumerate(assets.labels)}
    if model.config.id2label != expected_ids or model.config.label2id != expected_labels:
        raise ModelLoadError("Loaded model label mappings differ from the validated export.")
    if model.training or any(p.device.type != "cpu" for p in model.parameters()):
        raise ModelLoadError("Loaded model must be on CPU and in evaluation mode.")
    return LoadedModel(assets=assets, tokenizer=tokenizer, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify offline CPU model loading.")
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        loaded = load_local_model(args.model_dir)
    except (ValueError, ModelLoadError) as exc:
        parser.exit(1, f"Model check failed: {exc}\n")
    print(f"Loaded {len(loaded.assets.labels)} labels on CPU in evaluation mode.")
    print(f"Maximum input length: {loaded.assets.max_length} tokens. No prediction was run.")


if __name__ == "__main__":
    main()
