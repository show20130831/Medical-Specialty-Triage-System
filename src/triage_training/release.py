"""Release gate: decide whether a model bundle may be deployed."""

import json
import time
from pathlib import Path

import numpy as np
import yaml

from triage_training.artifacts import git_state, utc_timestamp, write_json
from triage_training.config import REPO_ROOT


def _check(name: str, value, requirement: str, passed: bool, kind: str = "gate") -> dict:
    return {"check": name, "value": value, "requirement": requirement,
            "passed": bool(passed), "kind": kind}


def offline_checks(evaluation: dict, manifest: dict, limits: dict) -> list[dict]:
    """Gate on the evaluation report produced for exactly this bundle."""
    run = evaluation.get("model_run", {})
    matches = (run.get("created_utc") == manifest.get("created_utc")
               and run.get("git_commit") == manifest.get("git", {}).get("commit"))
    primary = evaluation["models"][0]
    gain = evaluation["comparison"]["macro_f1_difference"]
    per_class = {label: row["f1-score"] for label, row in evaluation["errors"]["per_class"].items()}
    weakest = min(per_class, key=per_class.get)
    calibration = evaluation["calibration"]
    answered = calibration["uncertain_threshold"]["test"]
    return [
        _check("Evaluation report belongs to this bundle", run.get("run"),
               "same training run and commit", matches),
        _check("Test macro F1", primary["macro_f1"]["value"],
               f">= {limits['test_macro_f1_min']}",
               primary["macro_f1"]["value"] >= limits["test_macro_f1_min"]),
        _check("Macro F1 gain over baseline, 95% CI lower bound", gain["ci_low"],
               f"> {limits['macro_f1_gain_over_baseline_ci_low_min']}",
               gain["ci_low"] > limits["macro_f1_gain_over_baseline_ci_low_min"]),
        _check(f"Lowest per-class F1 ({weakest})", per_class[weakest],
               f">= {limits['per_class_f1_min']}", per_class[weakest] >= limits["per_class_f1_min"]),
        _check("Calibrated ECE", calibration["test_ece"]["calibrated"],
               f"<= {limits['calibrated_ece_max']}",
               calibration["test_ece"]["calibrated"] <= limits["calibrated_ece_max"]),
        _check("Accuracy on answered inputs", answered["accuracy"],
               f">= {limits['answered_accuracy_min']}",
               answered["accuracy"] >= limits["answered_accuracy_min"]),
        _check("Share of inputs answered", answered["coverage"],
               f">= {limits['answered_coverage_min']}",
               answered["coverage"] >= limits["answered_coverage_min"]),
    ]


def behavioural_checks(predict, cases: dict, limits: dict) -> tuple[list[dict], list[dict]]:
    """Run synthetic cases through the bundle; return gate checks and failure details."""
    failures = []

    def rate(results: list[bool]) -> float:
        return sum(results) / len(results)

    expected = []
    for case in cases["expected_label"]:
        p = predict(case["text"])
        ok = p.label == case["label"] and not p.uncertain
        expected.append(ok)
        if not ok:
            failures.append({"type": "expected_label", "text": case["text"], "want": case["label"],
                             "got": p.label, "score": p.score, "uncertain": p.uncertain})
    invariance = []
    for group in cases["invariance"]:
        base = predict(group["base"]).label
        for variant in group["variants"]:
            got = predict(variant).label
            invariance.append(got == base)
            if got != base:
                failures.append({"type": "invariance", "text": variant, "want": base, "got": got})
    abstention = []
    for text in cases["abstention"]:
        p = predict(text)
        abstention.append(p.uncertain)
        if not p.uncertain:
            failures.append({"type": "abstention", "text": text, "got": p.label, "score": p.score})
    out_of_scope = [predict(text) for text in cases["out_of_scope"]]
    confident_oos = rate([not p.uncertain for p in out_of_scope])

    checks = [
        _check(f"Expected labels ({len(expected)} clear cases)", rate(expected),
               f">= {limits['expected_label_pass_rate_min']}",
               rate(expected) >= limits["expected_label_pass_rate_min"]),
        _check(f"Invariance to case, punctuation, spacing ({len(invariance)} variants)",
               rate(invariance), f">= {limits['invariance_pass_rate_min']}",
               rate(invariance) >= limits["invariance_pass_rate_min"]),
        _check(f"Abstention on ambiguous descriptions ({len(abstention)} cases)", rate(abstention),
               f">= {limits['abstention_pass_rate_min']}",
               rate(abstention) >= limits["abstention_pass_rate_min"]),
        _check(f"Out-of-scope inputs answered confidently ({len(out_of_scope)} cases)",
               confident_oos, "reported only; no out-of-distribution detection", True,
               kind="info"),
    ]
    return checks, failures


def latency_check(predict, texts: list[str], limits: dict) -> dict:
    for text in texts[:limits["warmup_requests"]]:
        predict(text)
    timings = []
    for i in range(limits["measured_requests"]):
        started = time.perf_counter()
        predict(texts[i % len(texts)])
        timings.append((time.perf_counter() - started) * 1000)
    p50, p95 = np.percentile(timings, [50, 95])
    return _check("Single-request CPU latency p95 (ms)", float(p95),
                  f"<= {limits['p95_ms_max']} (p50 {p50:.0f} ms)", p95 <= limits["p95_ms_max"])


def run_release_validation(model_dir: Path, evaluation_path: Path, criteria_path: Path,
                           out_dir: Path) -> dict:
    from triage_system.model_loader import load_local_model
    from triage_system.predictor import predict_description

    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cases_path = REPO_ROOT / criteria["behavioural"]["cases"]
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))

    loaded = load_local_model(model_dir)

    def predict(text: str):
        return predict_description(loaded, text)

    behaviour, failures = behavioural_checks(predict, cases, criteria["behavioural"])
    texts = [case["text"] for case in cases["expected_label"]]
    checks = [*offline_checks(evaluation, manifest, criteria["offline_metrics"]),
              *behaviour, latency_check(predict, texts, criteria["latency"])]
    result = {
        "created_utc": utc_timestamp(),
        "decision": "PASS" if all(c["passed"] for c in checks) else "FAIL",
        "model": {"weights_sha256": loaded.weights_sha256, "run": evaluation["model_run"]["run"],
                  "training_commit": manifest.get("git", {}).get("commit")},
        "validated_with": git_state(),
        "checks": checks,
        "behavioural_failures": failures,
    }
    write_json(out_dir / "release_validation.json", result)
    (out_dir / "release_validation.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def _format(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_markdown(result: dict) -> str:
    model = result["model"]
    lines = [
        "# Release Validation",
        "",
        f"Decision: **{result['decision']}**  ",
        f"Model weights: `{model['weights_sha256'][:12]}` from run `{model['run']}` "
        f"(training commit `{str(model['training_commit'])[:7]}`)",
        "",
        "Generated by `python -m triage_training validate-model` against "
        "`configs/release_criteria.yaml`. Offline metrics come from the evaluation report; "
        "behavioural cases (`tests/golden/behavioural_cases.yaml`, synthetic) and latency "
        "are measured on the bundle on CPU.",
        "",
        "| Check | Value | Requirement | Result |",
        "| --- | --- | --- | --- |",
    ]
    for c in result["checks"]:
        outcome = "info" if c["kind"] == "info" else ("pass" if c["passed"] else "**FAIL**")
        lines.append(f"| {c['check']} | {_format(c['value'])} | {c['requirement']} | {outcome} |")
    if result["behavioural_failures"]:
        lines += ["", "## Behavioural case failures", ""]
        for f in result["behavioural_failures"]:
            detail = f"expected {f['want']}, got {f['got']}" if "want" in f else \
                f"got {f['got']} at {f['score']:.2f}, not marked uncertain"
            lines.append(f"- {f['type']}: \"{f['text']}\" — {detail}")
    return "\n".join(lines) + "\n"
