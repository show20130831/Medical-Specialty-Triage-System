"""Compare an aggregate /metrics snapshot with the model's reference profile."""

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chisquare

from triage_training.artifacts import utc_timestamp, write_json

STATUS_ORDER = ("OK", "WARN", "ALERT")
EPSILON = 1e-4


def population_stability_index(observed: dict, expected: dict) -> float:
    """PSI over shared buckets; proportions are floored to keep empty buckets finite."""
    keys = list(expected)
    obs = np.array([observed.get(k, 0) for k in keys], dtype=float)
    exp = np.array([expected[k] for k in keys], dtype=float)
    p = np.maximum(obs / max(obs.sum(), 1), EPSILON)
    q = np.maximum(exp / max(exp.sum(), 1), EPSILON)
    return float(((p - q) * np.log(p / q)).sum())


def _level(value: float, warn: float, alert: float) -> str:
    return "ALERT" if value >= alert else "WARN" if value >= warn else "OK"


def check_drift(snapshot: dict, reference: dict, thresholds: dict) -> dict:
    n = snapshot["predictions"]
    statuses = snapshot.get("predict_status_counts", {})
    responses = sum(statuses.values())
    result = {
        "created_utc": utc_timestamp(),
        "window_started_utc": snapshot.get("window_started_utc"),
        "model_weights_sha256": snapshot.get("model_weights_sha256"),
        "predictions": n,
        "checks": [],
    }
    if n < thresholds["min_predictions"]:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    psi = thresholds["psi"]
    for name, key in (("Predicted label distribution", "label_counts"),
                      ("Confidence distribution", "score_histogram"),
                      ("Input length distribution", "length_histogram")):
        value = population_stability_index(snapshot[key], reference[key])
        result["checks"].append({"check": name, "metric": "PSI", "value": value,
                                 "reference": 0.0, "status": _level(value, psi["warn"], psi["alert"])})

    labels = list(reference["label_counts"])
    observed = np.array([snapshot["label_counts"].get(k, 0) for k in labels], dtype=float)
    shares = np.maximum(np.array([reference["label_counts"][k] for k in labels], dtype=float)
                        / reference["predictions"], EPSILON)
    expected = shares / shares.sum() * observed.sum()
    result["label_chi_square_p_value"] = float(chisquare(observed, expected).pvalue)

    rate = snapshot["uncertain"] / n
    unc = thresholds["uncertain_rate"]
    result["checks"].append({"check": "Uncertain rate", "metric": "rate", "value": rate,
                             "reference": reference["uncertain"] / reference["predictions"],
                             "status": _level(rate, unc["warn"], unc["alert"])})
    if responses:
        errors = statuses.get("5xx", 0) / responses
        err = thresholds["server_error_rate"]
        result["checks"].append({"check": "Server error rate", "metric": "rate", "value": errors,
                                 "reference": 0.0,
                                 "status": _level(errors, err["warn"], err["alert"])})
    result["status"] = max((c["status"] for c in result["checks"]), key=STATUS_ORDER.index)
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# Drift Check",
        "",
        f"Overall status: **{result['status']}**  ",
        f"Window start: {result['window_started_utc']}; predictions: {result['predictions']}  ",
        f"Model weights: `{str(result['model_weights_sha256'])[:12]}`",
        "",
        "Signals compare production outputs and inputs with the test-set reference. Without "
        "ground-truth labels they show that something changed, not that accuracy dropped.",
    ]
    if result["status"] == "INSUFFICIENT_DATA":
        return "\n".join(lines + ["", "Too few predictions in this window to judge drift."]) + "\n"
    lines += ["", "| Check | Metric | Value | Reference | Status |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {c['check']} | {c['metric']} | {c['value']:.3f} | {c['reference']:.3f} | "
              f"{c['status']} |" for c in result["checks"]]
    lines += ["", f"Label distribution chi-square p-value: {result['label_chi_square_p_value']:.2g} "
                  "(informational; with many predictions tiny shifts become significant, so "
                  "status uses PSI)."]
    return "\n".join(lines) + "\n"


def run_drift_check(snapshot_path: Path, reference_path: Path, thresholds_path: Path,
                    out_dir: Path) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    result = check_drift(snapshot, reference, thresholds)
    write_json(out_dir / "drift_check.json", result)
    (out_dir / "drift_check.md").write_text(render_markdown(result), encoding="utf-8")
    return result
