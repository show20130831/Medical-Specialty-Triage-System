"""Statistical evaluation: bootstrap intervals, paired tests, and calibration."""

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from scipy.stats import binomtest

Metric = Callable[[np.ndarray, np.ndarray], float]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 8) -> float:
    """Macro F1 from a confusion matrix; fast enough for thousands of resamples."""
    matrix = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2)
    matrix = matrix.reshape(n_classes, n_classes)
    tp = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1.mean())


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, metric: Metric,
    n_resamples: int = 2000, seed: int = 0, level: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap interval over test examples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        samples[i] = metric(y_true[idx], y_pred[idx])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return {"value": metric(y_true, y_pred), "ci_low": float(low), "ci_high": float(high)}


def paired_bootstrap_difference(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, metric: Metric,
    n_resamples: int = 2000, seed: int = 0, level: float = 0.95,
) -> dict[str, float]:
    """Interval for metric(a) - metric(b) resampling the same examples for both models."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return {
        "difference": metric(y_true, pred_a) - metric(y_true, pred_b),
        "ci_low": float(low), "ci_high": float(high),
        # Share of resamples where a does not beat b: a one-sided bootstrap p-value.
        "p_value_one_sided": float((diffs <= 0).mean()),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired correctness: do the two models err on different rows?"""
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    p_value = binomtest(only_a, discordant, 0.5).pvalue if discordant else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p_value)}


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 15) -> float:
    """Weighted gap between confidence and accuracy over equal-width confidence bins."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def negative_log_likelihood(logits: np.ndarray, y_true: np.ndarray,
                            temperature: float = 1.0) -> float:
    log_probs = log_softmax(logits / temperature, axis=1)
    return float(-log_probs[np.arange(len(y_true)), y_true].mean())


def fit_temperature(val_logits: np.ndarray, val_true: np.ndarray) -> float:
    """Single-parameter temperature scaling fitted on validation NLL (Guo et al., 2017)."""
    result = minimize_scalar(
        lambda t: negative_log_likelihood(val_logits, val_true, t),
        bounds=(0.05, 20.0), method="bounded",
    )
    return float(result.x)


def calibrate(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def selective_prediction(probabilities: np.ndarray, y_true: np.ndarray,
                         thresholds: list[float]) -> list[dict[str, float]]:
    """Coverage and accuracy when predictions below a confidence threshold are withheld."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    rows = []
    for threshold in thresholds:
        answered = confidence >= threshold
        rows.append({
            "threshold": threshold,
            "coverage": float(answered.mean()),
            "accuracy": float(correct[answered].mean()) if answered.any() else float("nan"),
        })
    return rows


def choose_threshold(probabilities: np.ndarray, y_true: np.ndarray,
                     target_accuracy: float) -> float:
    """Lowest confidence threshold whose answered predictions reach the target accuracy."""
    for threshold in np.round(np.arange(0.0, 1.0, 0.01), 2):
        row = selective_prediction(probabilities, y_true, [float(threshold)])[0]
        if row["coverage"] > 0 and row["accuracy"] >= target_accuracy:
            return float(threshold)
    return 1.0
