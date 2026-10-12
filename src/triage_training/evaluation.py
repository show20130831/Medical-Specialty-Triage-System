"""Classification metrics shared by the baseline and the fine-tuned model."""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, labels: tuple[str, ...]
) -> dict:
    """Summarise argmax predictions; probabilities are kept separately for calibration."""
    y_pred = probabilities.argmax(axis=1)
    ids = list(range(len(labels)))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=ids, target_names=list(labels),
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def save_predictions(
    path: Path, source_row: np.ndarray, y_true: np.ndarray, scores: np.ndarray
) -> None:
    """Store per-row ids, labels, and scores (logits or probabilities) without any text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, source_row=source_row, y_true=y_true, scores=scores)
