"""Run one description through the validated local classifier."""

import math
from dataclasses import dataclass

from triage_system.model_loader import LoadedModel

TOP_K = 3


class PredictionError(ValueError):
    """The request or loaded model cannot produce a prediction."""


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    uncertain: bool = False
    top: tuple[tuple[str, float], ...] = ()


def rank_specialties(
    logits: list[float], labels: tuple[str, ...], temperature: float = 1.0,
    uncertain_threshold: float | None = None, top_k: int = TOP_K,
) -> Prediction:
    """Temperature-scaled softmax, top-k ranking, and abstention below the threshold."""
    if len(logits) != len(labels):
        raise PredictionError("Model returned a label outside the validated mapping.")
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    ranked = sorted(zip(labels, (w / total for w in weights)), key=lambda item: -item[1])
    label, score = ranked[0]
    uncertain = uncertain_threshold is not None and score < uncertain_threshold
    return Prediction(label=label, score=score, uncertain=uncertain, top=tuple(ranked[:top_k]))


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
        logits = loaded.model(**encoded).logits[0].float().tolist()
    return rank_specialties(logits, loaded.assets.labels, loaded.assets.temperature,
                            loaded.assets.uncertain_threshold)
