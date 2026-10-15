import threading

import pytest
from prometheus_client import generate_latest

from triage_system.monitoring import (
    LENGTH_EDGES, SCORE_EDGES, DriftMonitor, DriftThresholds, MonitoringMetrics, Window,
    bucket_index, bucket_labels, evaluate_window, population_stability_index,
)
from triage_training.config import REPO_ROOT

LABELS = ("A", "B")
REFERENCE = {
    "predictions": 200, "uncertain": 10,
    "label_counts": {"A": 100, "B": 100},
    "score_histogram": dict(zip(bucket_labels(SCORE_EDGES), [0, 10, 10, 30, 150])),
    "length_histogram": dict(zip(bucket_labels(LENGTH_EDGES), [40, 60, 60, 30, 10])),
}


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def monitor(min_predictions=4, **kwargs):
    metrics = MonitoringMetrics()
    clock = Clock()
    drift = DriftMonitor(metrics, DriftThresholds(min_predictions=min_predictions, **kwargs),
                         clock=clock)
    drift.set_reference(REFERENCE)
    return drift, metrics, clock


def value(metrics, name, labels=None):
    return metrics.registry.get_sample_value(name, labels or {})


def windows_closed(metrics):
    return sum(value(metrics, "triage_drift_windows_total", {"result": r}) or 0
               for r in ("insufficient_data", "ok", "warn", "alert"))


def test_bucket_edges_are_left_closed():
    assert bucket_index(0.64, SCORE_EDGES) == 1
    assert bucket_index(0.65, SCORE_EDGES) == 2
    assert bucket_labels(LENGTH_EDGES)[0] == "[-inf, 50)"


def test_identical_distribution_has_zero_psi():
    assert population_stability_index({"x": 5, "y": 5}, {"x": 1, "y": 1}) == pytest.approx(0)


def test_shifted_window_raises_status():
    window = Window(opened=0, labels={"A": 95, "B": 5}, score=[60, 20, 10, 5, 5],
                    length=[20, 40, 20, 15, 5], uncertain=60, predictions=100)
    result = evaluate_window(window, REFERENCE, DriftThresholds())
    assert result["status"] == "alert"
    assert result["psi"]["label"] > 0.25 and result["uncertain_rate"] == 0.6


def test_status_is_not_evaluated_until_a_window_completes():
    drift, metrics, _ = monitor()
    drift.record("A", 0.95, False, 120)
    drift.run_once()
    assert value(metrics, "triage_drift", {"triage_drift": "not_evaluated"}) == 1


def test_full_window_is_evaluated_and_published_without_specialty_values():
    drift, metrics, _ = monitor()
    for label in ("A", "B", "A", "B"):
        drift.record(label, 0.95, False, 120)
    assert drift.run_once()
    assert value(metrics, "triage_drift_window_predictions") == 4
    assert windows_closed(metrics) == 1
    assert value(metrics, "triage_drift", {"triage_drift": "not_evaluated"}) == 0
    assert value(metrics, "triage_drift_heartbeat_timestamp_seconds") == 1_000.0
    exported = generate_latest(metrics.registry).decode()
    assert 'specialty' not in exported
    assert '"A"' not in exported and '"B"' not in exported


def test_concurrent_predictions_land_in_exactly_one_window():
    drift, metrics, _ = monitor(min_predictions=50, max_pending=1000)
    threads = [threading.Thread(target=lambda: [drift.record("A", 0.95, False, 120)
                                                for _ in range(250)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    drift.run_once()
    assert windows_closed(metrics) == 2000 / 50
    assert value(metrics, "triage_drift_windows_dropped_total") == 0


def test_evaluation_failure_is_counted_and_keeps_heartbeat_stale(monkeypatch):
    drift, metrics, clock = monitor()
    drift.run_once()
    for _ in range(4):
        drift.record("A", 0.95, False, 120)
    monkeypatch.setattr("triage_system.monitoring.evaluate_window",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    clock.now = 2_000.0
    assert not drift.run_once()
    assert value(metrics, "triage_drift_evaluation_failures_total") == 1
    assert value(metrics, "triage_drift_windows_dropped_total") == 1
    assert value(metrics, "triage_drift_heartbeat_timestamp_seconds") == 1_000.0


def test_full_queue_drops_oldest_window():
    drift, metrics, _ = monitor(min_predictions=1, max_pending=2)
    for _ in range(3):
        drift.record("A", 0.95, False, 120)
    drift.run_once()
    assert value(metrics, "triage_drift_windows_dropped_total") == 1
    assert windows_closed(metrics) == 2


def test_unfilled_window_times_out_as_insufficient_data():
    drift, metrics, clock = monitor(min_predictions=100, insufficient_after_seconds=3600)
    drift.record("A", 0.95, False, 120)
    clock.now += 3600
    assert drift.run_once()
    assert value(metrics, "triage_drift", {"triage_drift": "insufficient_data"}) == 1
    assert value(metrics, "triage_drift_windows_total", {"result": "insufficient_data"}) == 1


def test_record_never_raises_into_the_prediction_path(monkeypatch):
    drift, metrics, _ = monitor()
    monkeypatch.setattr("triage_system.monitoring.bucket_index",
                        lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    drift.record("A", 0.95, False, 120)
    assert value(metrics, "triage_drift_evaluation_failures_total") == 1


def test_without_reference_nothing_is_recorded():
    metrics = MonitoringMetrics()
    drift = DriftMonitor(metrics, DriftThresholds(min_predictions=1))
    drift.record("A", 0.95, False, 120)
    assert drift.run_once()
    assert windows_closed(metrics) == 0


def test_thresholds_load_from_repository_config():
    thresholds = DriftThresholds.from_yaml(REPO_ROOT / "configs/monitoring.yaml")
    assert thresholds.min_predictions >= 100
    assert thresholds.max_pending == 10 and thresholds.psi_alert == 0.25
