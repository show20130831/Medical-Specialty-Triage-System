import pytest

from triage_system.monitoring import (
    LENGTH_EDGES, SCORE_EDGES, PredictionMetrics, bucket_index, bucket_labels,
)
from triage_training.drift import check_drift, population_stability_index

THRESHOLDS = {"min_predictions": 100, "psi": {"warn": 0.1, "alert": 0.25},
              "uncertain_rate": {"warn": 0.15, "alert": 0.25},
              "server_error_rate": {"warn": 0.01, "alert": 0.05}}


def test_bucket_edges_are_left_closed():
    assert bucket_index(0.64, SCORE_EDGES) == 1
    assert bucket_index(0.65, SCORE_EDGES) == 2
    assert bucket_labels(LENGTH_EDGES)[0] == "[-inf, 50)"


def test_metrics_aggregate_without_per_request_records():
    metrics = PredictionMetrics(("A", "B"))
    metrics.record_prediction("A", 0.95, False, 120, 80.0)
    metrics.record_prediction("B", 0.40, True, 30, 600.0)
    metrics.record_status(200)
    metrics.record_status(422)
    snap = metrics.snapshot()
    assert snap["predictions"] == 2 and snap["uncertain"] == 1
    assert snap["label_counts"] == {"A": 1, "B": 1}
    assert snap["score_histogram"]["[0.9, inf)"] == 1
    assert snap["length_histogram"]["[-inf, 50)"] == 1
    assert snap["latency_ms_histogram"]["[500, 1000)"] == 1
    assert snap["predict_status_counts"] == {"2xx": 1, "4xx": 1}
    assert set(snap) == {"window_started_utc", "predictions", "uncertain", "label_counts",
                         "score_histogram", "length_histogram", "latency_ms_histogram",
                         "predict_status_counts"}


def _profile(labels, uncertain=10, errors=0):
    n = sum(labels.values())
    return {"predictions": n, "uncertain": uncertain, "label_counts": labels,
            "score_histogram": {"low": 10, "high": n - 10},
            "length_histogram": {"short": n // 2, "long": n - n // 2},
            "predict_status_counts": {"2xx": n, "5xx": errors}}


def test_identical_distribution_is_ok():
    reference = _profile({"A": 100, "B": 100})
    result = check_drift(_profile({"A": 100, "B": 100}), reference, THRESHOLDS)
    assert result["status"] == "OK"
    assert population_stability_index({"x": 5, "y": 5}, {"x": 1, "y": 1}) == pytest.approx(0)


def test_small_window_is_not_judged():
    result = check_drift(_profile({"A": 20, "B": 20}), _profile({"A": 100, "B": 100}), THRESHOLDS)
    assert result["status"] == "INSUFFICIENT_DATA"


def test_label_shift_and_uncertainty_raise_alerts():
    reference = _profile({"A": 100, "B": 100})
    shifted = _profile({"A": 190, "B": 10}, uncertain=60)
    result = check_drift(shifted, reference, THRESHOLDS)
    statuses = {c["check"]: c["status"] for c in result["checks"]}
    assert statuses["Predicted label distribution"] == "ALERT"
    assert statuses["Uncertain rate"] == "ALERT"
    assert result["status"] == "ALERT"


def test_server_errors_raise_warning():
    reference = _profile({"A": 100, "B": 100})
    result = check_drift(_profile({"A": 100, "B": 100}, errors=4), reference, THRESHOLDS)
    assert {c["check"]: c["status"] for c in result["checks"]}["Server error rate"] == "WARN"
