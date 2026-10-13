import numpy as np
import pytest
from scipy.special import softmax
from sklearn.metrics import f1_score

from triage_training.stats import (
    accuracy, bootstrap_ci, calibrate, choose_threshold, expected_calibration_error,
    fit_temperature, macro_f1, mcnemar_exact, negative_log_likelihood,
    paired_bootstrap_difference, selective_prediction,
)

RNG = np.random.default_rng(0)


def test_macro_f1_matches_sklearn_including_absent_predictions():
    y_true = RNG.integers(0, 8, 300)
    y_pred = np.where(RNG.random(300) < 0.6, y_true, RNG.integers(0, 7, 300))
    expected = f1_score(y_true, y_pred, labels=range(8), average="macro", zero_division=0)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_bootstrap_interval_brackets_point_estimate():
    y_true = RNG.integers(0, 8, 200)
    y_pred = np.where(RNG.random(200) < 0.8, y_true, (y_true + 1) % 8)
    result = bootstrap_ci(y_true, y_pred, accuracy, n_resamples=500)
    assert result["ci_low"] <= result["value"] <= result["ci_high"]
    assert result["ci_high"] - result["ci_low"] < 0.2


def test_paired_difference_detects_a_clearly_better_model():
    y_true = RNG.integers(0, 8, 300)
    better = np.where(RNG.random(300) < 0.9, y_true, (y_true + 1) % 8)
    worse = np.where(RNG.random(300) < 0.5, y_true, (y_true + 1) % 8)
    result = paired_bootstrap_difference(y_true, better, worse, accuracy, n_resamples=500)
    assert result["ci_low"] > 0
    assert result["p_value_one_sided"] < 0.01


def test_mcnemar_uses_only_discordant_pairs():
    correct_a = np.array([True] * 10 + [True] * 5 + [False] * 5)
    correct_b = np.array([False] * 10 + [True] * 5 + [False] * 5)
    result = mcnemar_exact(correct_a, correct_b)
    assert (result["only_a_correct"], result["only_b_correct"]) == (10, 0)
    assert result["p_value"] == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(correct_a, correct_a)["p_value"] == 1.0


def test_temperature_scaling_recovers_overconfidence_and_lowers_ece():
    true_logits = RNG.normal(size=(4000, 8)) * 2
    y_true = np.array([RNG.choice(8, p=p) for p in softmax(true_logits, axis=1)])
    overconfident = true_logits * 3
    temperature = fit_temperature(overconfident, y_true)
    assert temperature == pytest.approx(3, rel=0.15)
    assert negative_log_likelihood(overconfident, y_true, temperature) < \
        negative_log_likelihood(overconfident, y_true)
    assert expected_calibration_error(calibrate(overconfident, temperature), y_true) < \
        expected_calibration_error(calibrate(overconfident, 1.0), y_true)


def test_perfect_confident_predictions_have_zero_ece():
    y_true = np.arange(8)
    assert expected_calibration_error(np.eye(8), y_true) == pytest.approx(0)


def test_selective_prediction_trades_coverage_for_accuracy():
    probabilities = np.array([[0.95, 0.05], [0.9, 0.1], [0.55, 0.45], [0.6, 0.4]])
    y_true = np.array([0, 0, 1, 1])
    rows = selective_prediction(probabilities, y_true, [0.0, 0.7])
    assert rows[0] == {"threshold": 0.0, "coverage": 1.0, "accuracy": 0.5}
    assert rows[1] == {"threshold": 0.7, "coverage": 0.5, "accuracy": 1.0}
    assert choose_threshold(probabilities, y_true, target_accuracy=0.9) == 0.61
