"""Operational metrics and in-process drift windows for the serving API.

Two categories of telemetry (docs/operations-plan.md §5):

- Operational metrics (requests, latency, model state) are exported per scrape.
- Model behaviour is exported only as the drift result of the last completed window:
  PSI values, the uncertain rate, and a status. Per-specialty counts stay in memory,
  are never exported, and are discarded once a window has been evaluated.

The prediction path only adds to the open window under a lock. A dedicated evaluator
thread computes drift, so monitoring can neither slow nor break a prediction.
"""

import bisect
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from math import log
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Enum, Gauge, Histogram

SCORE_EDGES = (0.5, 0.65, 0.8, 0.9)          # buckets: <0.5, 0.5-0.65, ..., >=0.9
LENGTH_EDGES = (50, 100, 200, 400)            # characters
LATENCY_BUCKETS_SECONDS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
DRIFT_STATES = ["not_evaluated", "insufficient_data", "ok", "warn", "alert"]
DRIFT_CHECKS = ("label", "confidence", "length")
EPSILON = 1e-4

logger = logging.getLogger("triage_system.monitoring")


def bucket_labels(edges: tuple[float, ...]) -> list[str]:
    bounds = ["-inf", *map(str, edges), "inf"]
    return [f"[{low}, {high})" for low, high in zip(bounds, bounds[1:])]


def bucket_index(value: float, edges: tuple[float, ...]) -> int:
    return bisect.bisect_right(edges, value)


class MonitoringMetrics:
    """All exported series, on one registry so tests can use a fresh one."""

    def __init__(self) -> None:
        self.registry = registry = CollectorRegistry()
        self.requests = Counter("triage_requests", "HTTP requests by route and status class",
                                ["route", "status_class"], registry=registry)
        self.duration = Histogram("triage_request_duration_seconds", "Request duration",
                                  ["route"], buckets=LATENCY_BUCKETS_SECONDS, registry=registry)
        self.in_flight = Gauge("triage_requests_in_flight", "Requests being handled",
                               registry=registry)
        self.model_loaded = Gauge("triage_model_loaded", "1 when the model is loaded",
                                  registry=registry)
        self.model_load_seconds = Gauge("triage_model_load_duration_seconds",
                                        "Seconds from process start to model loaded",
                                        registry=registry)
        self.model_info = Gauge("triage_model_info", "Identity of the served model",
                                ["weights_sha256", "training_commit"], registry=registry)
        self.drift_state = Enum("triage_drift", "Drift status of the current state",
                                states=DRIFT_STATES, registry=registry)
        self.drift_psi = Gauge("triage_drift_psi", "PSI of the last completed window",
                               ["check"], registry=registry)
        self.drift_uncertain_rate = Gauge("triage_drift_uncertain_rate",
                                          "Uncertain share in the last completed window",
                                          registry=registry)
        self.drift_window_size = Gauge("triage_drift_window_predictions",
                                       "Predictions in the last completed window",
                                       registry=registry)
        self.drift_last_result = Gauge("triage_drift_last_result_timestamp_seconds",
                                       "When the last window result was published",
                                       registry=registry)
        self.drift_heartbeat = Gauge("triage_drift_heartbeat_timestamp_seconds",
                                     "When the evaluator last completed a run",
                                     registry=registry)
        self.drift_interval = Gauge("triage_drift_evaluation_interval_seconds",
                                    "Evaluator interval, so alert rules can judge staleness",
                                    registry=registry)
        self.drift_windows = Counter("triage_drift_windows", "Closed windows by result",
                                     ["result"], registry=registry)
        self.drift_failures = Counter("triage_drift_evaluation_failures",
                                      "Drift evaluations or window updates that raised",
                                      registry=registry)
        self.drift_dropped = Counter("triage_drift_windows_dropped",
                                     "Windows dropped by a failure or a full queue",
                                     registry=registry)
        self.drift_state.state("not_evaluated")


@dataclass(frozen=True)
class DriftThresholds:
    min_predictions: int = 100
    insufficient_after_seconds: float = 24 * 3600
    max_pending: int = 10
    interval_seconds: float = 900
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    uncertain_warn: float = 0.15
    uncertain_alert: float = 0.25

    @classmethod
    def from_yaml(cls, path: Path) -> "DriftThresholds":
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        window = raw["window"]
        return cls(
            min_predictions=int(window["min_predictions"]),
            insufficient_after_seconds=float(window["insufficient_after_hours"]) * 3600,
            max_pending=int(window["max_pending"]),
            interval_seconds=float(raw["evaluation_interval_seconds"]),
            psi_warn=float(raw["psi"]["warn"]), psi_alert=float(raw["psi"]["alert"]),
            uncertain_warn=float(raw["uncertain_rate"]["warn"]),
            uncertain_alert=float(raw["uncertain_rate"]["alert"]),
        )


@dataclass
class Window:
    """Aggregate counts only: no text and no per-prediction record."""
    opened: float
    labels: dict[str, int] = field(default_factory=dict)
    score: list[int] = field(default_factory=lambda: [0] * (len(SCORE_EDGES) + 1))
    length: list[int] = field(default_factory=lambda: [0] * (len(LENGTH_EDGES) + 1))
    uncertain: int = 0
    predictions: int = 0


def population_stability_index(observed: dict[str, int], expected: dict[str, int]) -> float:
    """PSI over the reference buckets; shares are floored so empty buckets stay finite."""
    obs_total = max(sum(observed.get(k, 0) for k in expected), 1)
    exp_total = max(sum(expected.values()), 1)
    value = 0.0
    for key, count in expected.items():
        p = max(observed.get(key, 0) / obs_total, EPSILON)
        q = max(count / exp_total, EPSILON)
        value += (p - q) * log(p / q)
    return value


def _level(value: float, warn: float, alert: float) -> str:
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def evaluate_window(window: Window, reference: dict, thresholds: DriftThresholds) -> dict:
    observed = {
        "label": window.labels,
        "confidence": dict(zip(bucket_labels(SCORE_EDGES), window.score)),
        "length": dict(zip(bucket_labels(LENGTH_EDGES), window.length)),
    }
    expected = {"label": reference["label_counts"], "confidence": reference["score_histogram"],
                "length": reference["length_histogram"]}
    psi = {check: population_stability_index(observed[check], expected[check])
           for check in DRIFT_CHECKS}
    uncertain_rate = window.uncertain / window.predictions
    levels = [_level(v, thresholds.psi_warn, thresholds.psi_alert) for v in psi.values()]
    levels.append(_level(uncertain_rate, thresholds.uncertain_warn, thresholds.uncertain_alert))
    status = "alert" if "alert" in levels else "warn" if "warn" in levels else "ok"
    return {"status": status, "psi": psi, "uncertain_rate": uncertain_rate,
            "predictions": window.predictions}


class DriftMonitor:
    def __init__(self, metrics: MonitoringMetrics, thresholds: DriftThresholds,
                 clock=time.time) -> None:
        self.metrics = metrics
        self.thresholds = thresholds
        metrics.drift_interval.set(thresholds.interval_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._reference: dict | None = None
        self._window = Window(opened=clock())
        self._pending: deque[Window] = deque()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_reference(self, reference: dict | None) -> None:
        with self._lock:
            self._reference = reference
            self._window = Window(opened=self._clock())
            self._pending.clear()

    # Prediction path -----------------------------------------------------------------

    def record(self, label: str, score: float, uncertain: bool, length_chars: int) -> None:
        """Add one prediction to the open window. Never raises."""
        try:
            with self._lock:
                if self._reference is None:
                    return
                w = self._window
                w.labels[label] = w.labels.get(label, 0) + 1
                w.score[bucket_index(score, SCORE_EDGES)] += 1
                w.length[bucket_index(length_chars, LENGTH_EDGES)] += 1
                w.uncertain += int(uncertain)
                w.predictions += 1
                if w.predictions < self.thresholds.min_predictions:
                    return
                if len(self._pending) >= self.thresholds.max_pending:
                    self._pending.popleft()
                    self.metrics.drift_dropped.inc()
                self._pending.append(w)
                self._window = Window(opened=self._clock())
            self._wake.set()
        except Exception:
            self.metrics.drift_failures.inc()
            logger.exception("Drift window update failed; prediction unaffected")

    # Evaluator ------------------------------------------------------------------------

    def run_once(self) -> bool:
        """Evaluate handed-over windows and close a timed-out one; True if all succeeded."""
        now = self._clock()
        with self._lock:
            windows, reference = list(self._pending), self._reference
            self._pending.clear()
            # Low traffic, including none at all, closes the window as insufficient data.
            timed_out = (reference is not None and now - self._window.opened
                         >= self.thresholds.insufficient_after_seconds)
            if timed_out:
                self._window = Window(opened=now)
        succeeded = True
        for window in windows:
            try:
                self._publish(evaluate_window(window, reference, self.thresholds), now)
            except Exception:
                succeeded = False
                self.metrics.drift_failures.inc()
                self.metrics.drift_dropped.inc()
                logger.exception("Drift evaluation failed; window dropped")
        if timed_out:
            self.metrics.drift_state.state("insufficient_data")
            self.metrics.drift_windows.labels(result="insufficient_data").inc()
            self.metrics.drift_last_result.set(now)
        if succeeded:
            self.metrics.drift_heartbeat.set(now)
        return succeeded

    def _publish(self, result: dict, now: float) -> None:
        m = self.metrics
        for check, value in result["psi"].items():
            m.drift_psi.labels(check=check).set(value)
        m.drift_uncertain_rate.set(result["uncertain_rate"])
        m.drift_window_size.set(result["predictions"])
        m.drift_state.state(result["status"])
        m.drift_windows.labels(result=result["status"]).inc()
        m.drift_last_result.set(now)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="drift-evaluator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                self.metrics.drift_failures.inc()
                logger.exception("Drift evaluator run failed")
            self._wake.wait(timeout=self.thresholds.interval_seconds)
            self._wake.clear()
