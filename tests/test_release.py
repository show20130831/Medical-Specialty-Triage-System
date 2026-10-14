import copy

import yaml

from triage_system.predictor import Prediction
from triage_training.config import REPO_ROOT
from triage_training.release import (
    behavioural_checks, latency_check, offline_checks, render_markdown,
)

LIMITS = yaml.safe_load((REPO_ROOT / "configs/release_criteria.yaml").read_text(encoding="utf-8"))

EVALUATION = {
    "model_run": {"run": "run-1", "created_utc": "T1", "git_commit": "abc"},
    "models": [{"macro_f1": {"value": 0.89}}],
    "comparison": {"macro_f1_difference": {"ci_low": 0.1}},
    "errors": {"per_class": {"A": {"f1-score": 0.8}, "B": {"f1-score": 0.95}}},
    "calibration": {"test_ece": {"calibrated": 0.05},
                    "uncertain_threshold": {"test": {"accuracy": 0.92, "coverage": 0.9}}},
}
MANIFEST = {"created_utc": "T1", "git": {"commit": "abc"}}


def test_offline_checks_pass_for_matching_good_model():
    checks = offline_checks(EVALUATION, MANIFEST, LIMITS["offline_metrics"])
    assert all(c["passed"] for c in checks)


def test_offline_checks_fail_on_mismatched_report_and_weak_class():
    evaluation = copy.deepcopy(EVALUATION)
    evaluation["errors"]["per_class"]["A"]["f1-score"] = 0.6
    checks = offline_checks(evaluation, {"created_utc": "T2", "git": {"commit": "abc"}},
                            LIMITS["offline_metrics"])
    failed = {c["check"] for c in checks if not c["passed"]}
    assert failed == {"Evaluation report belongs to this bundle", "Lowest per-class F1 (A)"}


CASES = {
    "expected_label": [{"text": "knee", "label": "Ortho"}, {"text": "eye", "label": "ENT"}],
    "invariance": [{"base": "knee", "variants": ["KNEE"]}],
    "abstention": ["mixed"],
    "out_of_scope": ["weather"],
}


def fake_predict(text):
    table = {"knee": ("Ortho", 0.95), "KNEE": ("Ortho", 0.9), "eye": ("Gen", 0.9),
             "mixed": ("Ortho", 0.4), "weather": ("Gen", 0.8)}
    label, score = table[text]
    return Prediction(label, score, uncertain=score < 0.65)


def test_behavioural_checks_report_failures_and_keep_out_of_scope_informational():
    checks, failures = behavioural_checks(fake_predict, CASES, LIMITS["behavioural"])
    by_name = {c["check"].split(" (")[0]: c for c in checks}
    assert by_name["Expected labels"]["value"] == 0.5 and not by_name["Expected labels"]["passed"]
    assert by_name["Invariance to case, punctuation, spacing"]["passed"]
    assert by_name["Abstention on ambiguous descriptions"]["passed"]
    oos = by_name["Out-of-scope inputs answered confidently"]
    assert oos["kind"] == "info" and oos["passed"] and oos["value"] == 1.0
    assert failures == [{"type": "expected_label", "text": "eye", "want": "ENT", "got": "Gen",
                         "score": 0.9, "uncertain": False}]


def test_latency_and_report_rendering():
    check = latency_check(fake_predict, ["knee", "eye"],
                          {"warmup_requests": 1, "measured_requests": 5, "p95_ms_max": 500})
    assert check["passed"]
    result = {"decision": "FAIL", "checks": [check],
              "model": {"weights_sha256": "f" * 64, "run": "run-1", "training_commit": "abc"},
              "behavioural_failures": [{"type": "abstention", "text": "mixed", "got": "Ortho",
                                        "score": 0.9}]}
    markdown = render_markdown(result)
    assert "Decision: **FAIL**" in markdown and "not marked uncertain" in markdown
