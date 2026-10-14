"""Aggregate, in-memory prediction metrics for monitoring.

Only counters and fixed-bucket histograms are kept: no request text and no per-request
records, so a snapshot cannot be traced back to an individual request. Counters live in
process memory and reset on restart; the window start is reported with every snapshot.
The same bucket edges are used to build the offline reference profile, so production
snapshots and the reference can be compared directly.
"""

import bisect
import threading
from datetime import datetime, timezone

SCORE_EDGES = (0.5, 0.65, 0.8, 0.9)          # buckets: <0.5, 0.5-0.65, ..., >=0.9
LENGTH_EDGES = (50, 100, 200, 400)            # characters
LATENCY_EDGES_MS = (50, 100, 200, 500, 1000)


def bucket_labels(edges: tuple[float, ...]) -> list[str]:
    bounds = ["-inf", *map(str, edges), "inf"]
    return [f"[{low}, {high})" for low, high in zip(bounds, bounds[1:])]


def bucket_index(value: float, edges: tuple[float, ...]) -> int:
    return bisect.bisect_right(edges, value)


class PredictionMetrics:
    """Thread-safe counters for one service process."""

    def __init__(self, labels: tuple[str, ...]):
        self._lock = threading.Lock()
        self._labels = labels
        self._started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._predictions = 0
        self._uncertain = 0
        self._label_counts = dict.fromkeys(labels, 0)
        self._score = [0] * (len(SCORE_EDGES) + 1)
        self._length = [0] * (len(LENGTH_EDGES) + 1)
        self._latency = [0] * (len(LATENCY_EDGES_MS) + 1)
        self._status: dict[str, int] = {}

    def record_prediction(self, label: str, score: float, uncertain: bool,
                          length_chars: int, latency_ms: float) -> None:
        with self._lock:
            self._predictions += 1
            self._uncertain += int(uncertain)
            self._label_counts[label] += 1
            self._score[bucket_index(score, SCORE_EDGES)] += 1
            self._length[bucket_index(length_chars, LENGTH_EDGES)] += 1
            self._latency[bucket_index(latency_ms, LATENCY_EDGES_MS)] += 1

    def record_status(self, status_code: int) -> None:
        with self._lock:
            key = f"{status_code // 100}xx"
            self._status[key] = self._status.get(key, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "window_started_utc": self._started,
                "predictions": self._predictions,
                "uncertain": self._uncertain,
                "label_counts": dict(self._label_counts),
                "score_histogram": dict(zip(bucket_labels(SCORE_EDGES), self._score)),
                "length_histogram": dict(zip(bucket_labels(LENGTH_EDGES), self._length)),
                "latency_ms_histogram": dict(zip(bucket_labels(LATENCY_EDGES_MS), self._latency)),
                "predict_status_counts": dict(self._status),
            }
