"""Run one description through the validated local classifier."""

from dataclasses import dataclass

from triage_system.model_loader import LoadedModel, load_local_model


class PredictionError(ValueError):
    """The request or loaded model cannot produce a prediction."""


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


def predict_description(loaded: LoadedModel, description: str) -> Prediction:
    """Predict one non-empty description without logging or persisting its text."""
    if not isinstance(description, str) or not description.strip():
        raise PredictionError("description must contain non-blank text.")
    try:
        import torch
    except ImportError:
        raise PredictionError("Install CPU PyTorch and the inference extra first.") from None
    encoded = loaded.tokenizer(
        description,
        return_tensors="pt",
        truncation=True,
        max_length=loaded.assets.max_length,
    )
    encoded = {key: value.to("cpu") for key, value in encoded.items()}
    with torch.inference_mode():
        logits = loaded.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    index = int(torch.argmax(probabilities).item())
    if index >= len(loaded.assets.labels):
        raise PredictionError("Model returned a label outside the validated mapping.")
    return Prediction(label=loaded.assets.labels[index], score=float(probabilities[index]))
