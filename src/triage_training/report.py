"""Assemble the statistical evaluation report from saved run predictions.

The report only contains aggregate numbers, so it can be committed; it reads the
prepared splits locally to group test rows but never writes their text.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from triage_system.monitoring import (
    LENGTH_EDGES, SCORE_EDGES, bucket_index, bucket_labels,
)
from triage_training.artifacts import utc_timestamp, write_json
from triage_training.config import TrainingConfig
from triage_training.data import PreparedDataset, description_key, load_dataset
from triage_training.evaluation import classification_metrics
from triage_training.taxonomy import Taxonomy, load_taxonomy
from triage_training.stats import (
    accuracy, bootstrap_ci, brier_score, calibrate, choose_threshold,
    expected_calibration_error, fit_temperature, macro_f1, mcnemar_exact,
    negative_log_likelihood, paired_bootstrap_difference, reliability_table,
    selective_bootstrap, selective_prediction, threshold_stability,
)

NEAR_DUPLICATE_SIMILARITY = 0.8
TARGET_ANSWERED_ACCURACY = 0.90
THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass(frozen=True)
class RunScores:
    run_dir: Path
    display_name: str
    manifest: dict
    splits: dict[str, dict[str, np.ndarray]]  # split -> source_row, y_true, logits

    @property
    def deduplicated(self) -> bool:
        return self.manifest["dataset"]["deduplicated"]

    def predictions(self, split: str = "test") -> np.ndarray:
        return self.splits[split]["logits"].argmax(axis=1)


def load_run(run_dir: Path) -> RunScores:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    is_probability = manifest["model"] == "tfidf_logistic_regression"
    splits = {}
    for name in ("val", "test"):
        saved = np.load(run_dir / f"predictions_{name}.npz")
        scores = saved["scores"].astype(np.float64)
        # Log-probabilities behave as logits for argmax, softmax, and temperature scaling.
        logits = np.log(np.clip(scores, 1e-12, 1.0)) if is_probability else scores
        splits[name] = {"source_row": saved["source_row"], "y_true": saved["y_true"],
                        "logits": logits}
    if is_probability:
        display = "TF-IDF + logistic regression"
    elif "evaluated_export" in manifest:
        display = f"PubMedBERT ({run_dir.name.split('_')[0]} export)"
    else:
        display = f"PubMedBERT (seed {manifest['config']['seed']})"
    return RunScores(run_dir=run_dir, display_name=display, manifest=manifest, splits=splits)


def _require_same_test_rows(runs: list[RunScores]) -> None:
    reference = runs[0].splits["test"]["source_row"]
    for run in runs[1:]:
        if not np.array_equal(run.splits["test"]["source_row"], reference):
            raise ValueError(f"{run.run_dir.name} was not evaluated on the same test rows.")


def _summary(run: RunScores, labels: tuple[str, ...]) -> dict:
    y_true, logits = run.splits["test"]["y_true"], run.splits["test"]["logits"]
    y_pred = logits.argmax(axis=1)
    full = classification_metrics(y_true, calibrate(logits, 1.0), labels)
    return {
        "name": run.display_name,
        "run": run.run_dir.name,
        "n_test": int(len(y_true)),
        "accuracy": bootstrap_ci(y_true, y_pred, accuracy),
        "macro_f1": bootstrap_ci(y_true, y_pred, macro_f1),
        "weighted_f1": full["weighted_f1"],
        "val_macro_f1": macro_f1(run.splits["val"]["y_true"], run.predictions("val")),
    }


def _comparison(a: RunScores, b: RunScores) -> dict:
    y_true = a.splits["test"]["y_true"]
    pred_a, pred_b = a.predictions(), b.predictions()
    return {
        "a": a.display_name,
        "b": b.display_name,
        "mcnemar": mcnemar_exact(pred_a == y_true, pred_b == y_true),
        "macro_f1_difference": paired_bootstrap_difference(y_true, pred_a, pred_b, macro_f1),
        "accuracy_difference": paired_bootstrap_difference(y_true, pred_a, pred_b, accuracy),
    }


def _errors(run: RunScores, labels: tuple[str, ...]) -> dict:
    y_true = run.splits["test"]["y_true"]
    metrics = classification_metrics(y_true, calibrate(run.splits["test"]["logits"], 1.0), labels)
    matrix = np.array(metrics["confusion_matrix"])
    np.fill_diagonal(matrix, 0)
    pairs = sorted(((int(matrix[i, j]), labels[i], labels[j])
                    for i in range(len(labels)) for j in range(len(labels)) if matrix[i, j]),
                   reverse=True)[:5]
    return {
        "per_class": {label: {k: metrics["per_class"][label][k]
                              for k in ("precision", "recall", "f1-score", "support")}
                      for label in labels},
        "top_confusions": [{"true": t, "predicted": p, "count": c} for c, t, p in pairs],
    }


def _calibration(run: RunScores, labels: tuple[str, ...]) -> dict:
    val, test = run.splits["val"], run.splits["test"]
    temperature = fit_temperature(val["logits"], val["y_true"])
    raw_test, calibrated_test = calibrate(test["logits"], 1.0), calibrate(test["logits"], temperature)
    calibrated_val = calibrate(val["logits"], temperature)
    threshold = choose_threshold(calibrated_val, val["y_true"], TARGET_ANSWERED_ACCURACY)
    resampled = np.array(threshold_stability(calibrated_val, val["y_true"],
                                             TARGET_ANSWERED_ACCURACY))
    low, median, high = (float(v) for v in np.percentile(resampled, [2.5, 50, 97.5]))
    abstained = calibrated_test.max(axis=1) < threshold
    correct = calibrated_test.argmax(axis=1) == test["y_true"]
    return {
        "temperature": temperature,
        "test_ece": {"raw": expected_calibration_error(raw_test, test["y_true"]),
                     "calibrated": expected_calibration_error(calibrated_test, test["y_true"])},
        "test_nll": {"raw": negative_log_likelihood(test["logits"], test["y_true"]),
                     "calibrated": negative_log_likelihood(test["logits"], test["y_true"],
                                                           temperature)},
        "test_brier": {"raw": brier_score(raw_test, test["y_true"]),
                       "calibrated": brier_score(calibrated_test, test["y_true"])},
        "reliability_test": reliability_table(calibrated_test, test["y_true"]),
        "selective_test": selective_prediction(calibrated_test, test["y_true"], THRESHOLDS),
        "uncertain_threshold": {
            "target_answered_accuracy": TARGET_ANSWERED_ACCURACY,
            "chosen_on": "val",
            "threshold": threshold,
            "test": selective_prediction(calibrated_test, test["y_true"], [threshold])[0],
            "test_ci": selective_bootstrap(calibrated_test, test["y_true"], threshold),
            "stability": {
                "validation_resamples": int(len(resampled)),
                "threshold_median": median, "threshold_ci_low": low, "threshold_ci_high": high,
                "test_at_ci_low": selective_prediction(calibrated_test, test["y_true"], [low])[0],
                "test_at_ci_high": selective_prediction(calibrated_test, test["y_true"], [high])[0],
            },
            "per_class_test": {
                label: {"n": int((test["y_true"] == i).sum()),
                        "abstention_rate": float(abstained[test["y_true"] == i].mean()),
                        "answered_accuracy": float(correct[(test["y_true"] == i) & ~abstained].mean())
                        if ((test["y_true"] == i) & ~abstained).any() else None}
                for i, label in enumerate(labels)
            },
        },
    }


def _parity(reference: RunScores, other: RunScores, temperature: float, threshold: float) -> dict:
    """Agreement between the evaluated run (GPU) and the same bundle scored elsewhere (CPU)."""
    if not np.array_equal(reference.splits["test"]["source_row"], other.splits["test"]["source_row"]):
        raise ValueError(f"{other.run_dir.name} was not evaluated on the same test rows.")
    a = calibrate(reference.splits["test"]["logits"], temperature)
    b = calibrate(other.splits["test"]["logits"], temperature)
    top3 = lambda p: np.argsort(-p, axis=1)[:, :3]  # noqa: E731
    y_true = other.splits["test"]["y_true"]
    return {
        "reference_run": reference.run_dir.name,
        "compared_run": other.run_dir.name,
        "rows": int(len(a)),
        "label_agreement": float((a.argmax(1) == b.argmax(1)).mean()),
        "top3_order_agreement": float((top3(a) == top3(b)).all(axis=1).mean()),
        "uncertain_agreement": float(((a.max(1) < threshold) == (b.max(1) < threshold)).mean()),
        "max_abs_probability_difference": float(np.abs(a - b).max()),
        "compared_macro_f1": macro_f1(y_true, b.argmax(1)),
        "compared_accuracy": accuracy(y_true, b.argmax(1)),
    }


def _monitoring_reference(run: RunScores, dataset: PreparedDataset, temperature: float,
                          threshold: float) -> dict:
    """Expected production distributions, in the same buckets as the /metrics snapshot."""
    test = dataset.splits["test"].set_index("source_row").loc[run.splits["test"]["source_row"]]
    probabilities = calibrate(run.splits["test"]["logits"], temperature)
    top = probabilities.max(axis=1)
    labels = [dataset.labels[i] for i in probabilities.argmax(axis=1)]

    def histogram(values, edges):
        counts = [0] * (len(edges) + 1)
        for value in values:
            counts[bucket_index(value, edges)] += 1
        return dict(zip(bucket_labels(edges), counts))

    return {
        "source": "calibrated predictions on the deduplicated test split",
        "predictions": int(len(top)),
        "uncertain": int((top < threshold).sum()),
        "label_counts": {label: labels.count(label) for label in dataset.labels},
        "score_histogram": histogram(top, SCORE_EDGES),
        "length_histogram": histogram(test["description"].str.len(), LENGTH_EDGES),
    }


def _accuracy_by(mask: np.ndarray, correct: np.ndarray) -> dict:
    return {"n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None}


def _label_source(runs: list[RunScores], dataset: PreparedDataset) -> dict:
    """Accuracy on rows labeled by direct specialty mapping vs. keyword rules."""
    test = dataset.splits["test"].set_index("source_row")
    out = {}
    for run in runs:
        rows = run.splits["test"]["source_row"]
        source = test.loc[rows, "label_source"].to_numpy()
        correct = run.predictions() == run.splits["test"]["y_true"]
        out[run.display_name] = {kind: _accuracy_by(source == kind, correct)
                                 for kind in ("direct", "keyword")}
    return out


def _keyword_label_overlap(run: RunScores, dataset: PreparedDataset,
                           taxonomy: Taxonomy) -> dict:
    """How often a keyword-rule label is visible in the model input itself."""
    test = dataset.splits["test"].set_index("source_row").loc[run.splits["test"]["source_row"]]
    correct = run.predictions() == run.splits["test"]["y_true"]
    keyword = (test["label_source"] == "keyword").to_numpy()

    def contains(min_weight: int) -> np.ndarray:
        return np.array([
            any(term in text.lower() for term, weight in taxonomy.keyword_weights[label].items()
                if weight >= min_weight)
            for text, label in zip(test["description"], test["label"])
        ])

    any_term, specific = contains(1) & keyword, contains(2) & keyword
    return {
        "model": run.display_name,
        "keyword_rows": int(keyword.sum()),
        "share_with_own_label_term": float(any_term.sum() / keyword.sum()) if keyword.any()
        else None,
        "accuracy_with_specific_term": _accuracy_by(specific, correct),
        "accuracy_without_specific_term": _accuracy_by(keyword & ~specific, correct),
    }


def _near_duplicates(runs: list[RunScores], dataset: PreparedDataset) -> dict:
    """Exact deduplication misses paraphrases; measure how close test rows are to train."""
    train, test = dataset.splits["train"], dataset.splits["test"].set_index("source_row")
    test = test.loc[runs[0].splits["test"]["source_row"]]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vectorizer.fit(list(train["description"]) + list(test["description"]))
    nearest = cosine_similarity(vectorizer.transform(test["description"]),
                                vectorizer.transform(train["description"])).max(axis=1)
    near = nearest >= NEAR_DUPLICATE_SIMILARITY
    return {
        "similarity": "character 3-5 gram TF-IDF cosine to the nearest train description",
        "median_nearest_similarity": float(np.median(nearest)),
        "share_at_least": {str(t): float((nearest >= t).mean()) for t in (0.9, 0.8, 0.7)},
        "accuracy": {run.display_name: {
            "near_duplicate": _accuracy_by(near, run.predictions() == run.splits["test"]["y_true"]),
            "other": _accuracy_by(~near, run.predictions() == run.splits["test"]["y_true"]),
        } for run in runs},
    }


def _leakage(run: RunScores, config: TrainingConfig) -> dict:
    """Split row-level test accuracy into rows whose description was also in train."""
    split_seed = run.manifest["dataset"]["split_seed"]
    dataset = load_dataset(config.paths.data_dir / f"rowlevel_seed{split_seed}")
    keys = {name: part["description"].map(description_key)
            for name, part in dataset.splits.items()}
    labels_by_key: dict[str, set[str]] = {}
    for name, part in dataset.splits.items():
        for key, label in zip(keys[name], part["label"]):
            labels_by_key.setdefault(key, set()).add(label)
    train_keys = set(keys["train"])

    test = dataset.splits["test"].assign(key=keys["test"]).set_index("source_row")
    test = test.loc[run.splits["test"]["source_row"]]
    seen = test["key"].isin(train_keys).to_numpy()
    conflicting = test["key"].map(lambda k: len(labels_by_key[k]) > 1).to_numpy()
    correct = run.predictions() == run.splits["test"]["y_true"]
    return {
        "name": run.display_name,
        "run": run.run_dir.name,
        "all": _accuracy_by(np.ones_like(correct), correct),
        "seen_in_train": _accuracy_by(seen, correct),
        "seen_consistent_label": _accuracy_by(seen & ~conflicting, correct),
        "unseen": _accuracy_by(~seen, correct),
        "conflicting_label": _accuracy_by(conflicting, correct),
        "macro_f1": macro_f1(run.splits["test"]["y_true"], run.predictions()),
    }


def build_report(config: TrainingConfig, model_dir: Path, baseline_dir: Path,
                 seed_dirs: list[Path], rowlevel_dirs: list[Path], out_dir: Path,
                 serving_dir: Path | None = None) -> dict:
    primary, baseline = load_run(model_dir), load_run(baseline_dir)
    seeds = [load_run(d) for d in seed_dirs]
    rowlevel = [load_run(d) for d in rowlevel_dirs]
    main_runs = [primary, baseline, *seeds]
    if not all(run.deduplicated for run in main_runs):
        raise ValueError("--model, --baseline, and --seed-run must use the deduplicated split.")
    if any(run.deduplicated for run in rowlevel):
        raise ValueError("--rowlevel runs must use the row-level split.")
    _require_same_test_rows(main_runs)

    split_seed = primary.manifest["dataset"]["split_seed"]
    dataset = load_dataset(config.paths.data_dir / f"dedup_seed{split_seed}")
    labels = dataset.labels
    finetuned = [primary, *seeds]
    seed_scores = [macro_f1(r.splits["test"]["y_true"], r.predictions()) for r in finetuned]
    calibration = _calibration(primary, labels)
    report = {
        "created_utc": utc_timestamp(),
        "dataset": {k: dataset.report[k] for k in ("audit", "split_sizes", "overlap",
                                                   "raw_csv_sha256", "taxonomy_sha256")},
        "models": [_summary(run, labels) for run in main_runs],
        "seed_variation": {
            "runs": len(finetuned),
            "test_macro_f1_mean": float(np.mean(seed_scores)),
            "test_macro_f1_std": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1
            else None,
        },
        "comparison": _comparison(primary, baseline),
        "errors": _errors(primary, labels),
        "calibration": calibration,
        "model_run": {"run": primary.run_dir.name,
                      "created_utc": primary.manifest.get("created_utc"),
                      "git_commit": primary.manifest.get("git", {}).get("commit")},
        "label_source": _label_source([primary, baseline], dataset),
        "keyword_label_overlap": _keyword_label_overlap(
            primary, dataset, load_taxonomy(config.data.taxonomy)),
        "near_duplicates": _near_duplicates([primary, baseline], dataset),
        "leakage": [_leakage(run, config) for run in rowlevel],
        "serving_parity": _parity(primary, load_run(serving_dir), calibration["temperature"],
                                  calibration["uncertain_threshold"]["threshold"])
        if serving_dir else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model_evaluation.json", report)
    (out_dir / "model_evaluation.md").write_text(render_markdown(report), encoding="utf-8")

    export_dir = model_dir / "model"
    if export_dir.is_dir():
        # Serving reads this to report calibrated scores and withhold low-confidence answers.
        write_json(export_dir / "calibration.json", {
            "temperature": calibration["temperature"],
            "uncertain_threshold": calibration["uncertain_threshold"]["threshold"],
            "target_answered_accuracy": TARGET_ANSWERED_ACCURACY,
        })
        # Baseline for drift checks against the service's /metrics snapshots.
        write_json(export_dir / "monitoring_reference.json", _monitoring_reference(
            primary, dataset, calibration["temperature"],
            calibration["uncertain_threshold"]["threshold"]))
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ci(stat: dict) -> str:
    return f"{stat['value']:.3f} [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}]"


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def render_markdown(report: dict) -> str:
    d = report["dataset"]
    audit = d["audit"]
    lines = [
        "# Model Evaluation",
        "",
        "Generated by `python -m triage_training report`. All numbers are on the "
        "deduplicated test split unless stated otherwise; intervals are 95% percentile "
        "bootstrap intervals over test examples (2,000 resamples).",
        "",
        "## Data",
        "",
        f"- {audit['labeled_rows']:,} labeled MTSamples rows; "
        f"{audit.get('duplicate_rows_removed', 0):,} duplicates removed; "
        f"{audit.get('conflicting_descriptions_dropped', 0):,} descriptions with unresolvable "
        f"conflicting labels dropped.",
        f"- Splits: train {d['split_sizes']['train']:,} / val {d['split_sizes']['val']:,} / "
        f"test {d['split_sizes']['test']:,}; shared descriptions between splits: "
        f"{sum(d['overlap'].values())}.",
        "",
        "## Model comparison",
        "",
        "| Model | Test accuracy | Test macro F1 | Weighted F1 | Val macro F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in report["models"]:
        lines.append(f"| {m['name']} | {_ci(m['accuracy'])} | {_ci(m['macro_f1'])} | "
                     f"{m['weighted_f1']:.3f} | {m['val_macro_f1']:.3f} |")
    s = report["seed_variation"]
    if s["test_macro_f1_std"] is not None:
        lines += ["", f"Across {s['runs']} fine-tuning seeds, test macro F1 is "
                      f"{s['test_macro_f1_mean']:.3f} ± {s['test_macro_f1_std']:.3f} (mean ± SD)."]

    c = report["comparison"]
    f1d, accd, mc = c["macro_f1_difference"], c["accuracy_difference"], c["mcnemar"]
    lines += [
        "",
        f"### {c['a']} vs. {c['b']}",
        "",
        f"- Macro F1 difference: {f1d['difference']:+.3f} "
        f"[{f1d['ci_low']:+.3f}, {f1d['ci_high']:+.3f}], one-sided bootstrap "
        f"p = {_p(f1d['p_value_one_sided'])}.",
        f"- Accuracy difference: {accd['difference']:+.3f} "
        f"[{accd['ci_low']:+.3f}, {accd['ci_high']:+.3f}].",
        f"- Exact McNemar test on paired correctness: only {c['a']} correct on "
        f"{mc['only_a_correct']} rows, only {c['b']} correct on {mc['only_b_correct']}; "
        f"p = {_p(mc['p_value'])}.",
        "",
        "## Errors",
        "",
        "| Specialty | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, row in report["errors"]["per_class"].items():
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | "
                     f"{row['f1-score']:.3f} | {int(row['support'])} |")
    lines += ["", "Most frequent confusions (true → predicted):", ""]
    lines += [f"- {e['true']} → {e['predicted']}: {e['count']}"
              for e in report["errors"]["top_confusions"]]

    lines += ["", "### Label provenance", "",
              "Rows from catch-all specialties (Surgery, Radiology, SOAP notes, discharge "
              "summaries) are labeled by keyword rules rather than a direct specialty mapping.",
              "", "| Model | Direct-mapped rows | Keyword-rule rows |", "| --- | --- | --- |"]
    for name, row in report["label_source"].items():
        lines.append(f"| {name} | {_pct(row['direct']['accuracy'])} (n={row['direct']['n']}) | "
                     f"{_pct(row['keyword']['accuracy'])} (n={row['keyword']['n']}) |")
    ko = report["keyword_label_overlap"]
    with_term, without = ko["accuracy_with_specific_term"], ko["accuracy_without_specific_term"]
    lines += ["", f"For {_pct(ko['share_with_own_label_term'])} of the {ko['keyword_rows']} "
                  "keyword-rule test rows, the description itself contains a keyword of its "
                  f"assigned label. {ko['model']} scores {_pct(with_term['accuracy'])} "
                  f"(n={with_term['n']}) on rows containing a specific (weight 2) keyword and "
                  f"{_pct(without['accuracy'])} (n={without['n']}) on the rest. Part of the "
                  "headline score therefore reflects the labeling rules; accuracy on directly "
                  "mapped rows is the more conservative estimate."]

    nd = report["near_duplicates"]
    lines += ["", "### Near duplicates", "",
              "Deduplication removes exact copies after normalising case, punctuation, and "
              f"whitespace. Measured by {nd['similarity']}, the median test row scores "
              f"{nd['median_nearest_similarity']:.2f}; "
              + ", ".join(f"{_pct(v)} score ≥ {k}" for k, v in nd["share_at_least"].items())
              + ".", "",
              f"| Model | Near duplicate (≥ {NEAR_DUPLICATE_SIMILARITY}) | Other rows |",
              "| --- | --- | --- |"]
    for name, row in nd["accuracy"].items():
        lines.append(f"| {name} | {_pct(row['near_duplicate']['accuracy'])} "
                     f"(n={row['near_duplicate']['n']}) | {_pct(row['other']['accuracy'])} "
                     f"(n={row['other']['n']}) |")

    cal = report["calibration"]
    u = cal["uncertain_threshold"]
    lines += [
        "",
        "## Calibration and abstention",
        "",
        f"Temperature scaling fitted on validation NLL gives T = {cal['temperature']:.2f}. "
        f"Test ECE {cal['test_ece']['raw']:.3f} → {cal['test_ece']['calibrated']:.3f}; "
        f"test NLL {cal['test_nll']['raw']:.3f} → {cal['test_nll']['calibrated']:.3f}.",
        "",
        "| Calibrated confidence ≥ | Coverage | Accuracy on answered |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {r['threshold']:.1f} | {_pct(r['coverage'])} | {_pct(r['accuracy'])} |"
              for r in cal["selective_test"]]
    ci, st = u["test_ci"], u["stability"]
    lines += ["", f"The serving threshold is chosen on validation as the lowest confidence "
                  f"reaching {_pct(u['target_answered_accuracy'])} accuracy: "
                  f"{u['threshold']:.2f}. On test it answers {_pct(u['test']['coverage'])} "
                  f"[{_pct(ci['coverage']['ci_low'])}, {_pct(ci['coverage']['ci_high'])}] of "
                  f"inputs with {_pct(u['test']['accuracy'])} "
                  f"[{_pct(ci['accuracy']['ci_low'])}, {_pct(ci['accuracy']['ci_high'])}] "
                  "accuracy; the rest are returned as uncertain.",
              "",
              f"**Threshold stability.** Re-choosing the threshold on "
              f"{st['validation_resamples']:,} bootstrap resamples of the validation split gives "
              f"a median of {st['threshold_median']:.2f} and a 95% range of "
              f"{st['threshold_ci_low']:.2f}–{st['threshold_ci_high']:.2f}. Across that range, "
              f"test coverage runs from {_pct(st['test_at_ci_high']['coverage'])} to "
              f"{_pct(st['test_at_ci_low']['coverage'])} and answered accuracy from "
              f"{_pct(st['test_at_ci_low']['accuracy'])} to "
              f"{_pct(st['test_at_ci_high']['accuracy'])}. With 318 validation rows the exact "
              "threshold is not stable; the release gate checks the resulting test behaviour "
              "rather than the threshold value.",
              "",
              f"Brier score {cal['test_brier']['raw']:.3f} → {cal['test_brier']['calibrated']:.3f}. "
              "Scores are relative probabilities over eight closed-set groups, not the "
              "probability that a suggestion is clinically correct.",
              "",
              "| Calibrated confidence | Test rows | Mean confidence | Accuracy |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {r['low']:.2f}–{r['high']:.2f} | {r['n']} | "
              f"{'n/a' if r['mean_confidence'] is None else format(r['mean_confidence'], '.3f')} | "
              f"{_pct(r['accuracy'])} |" for r in cal["reliability_test"]]
    lines += ["", "| Specialty (true label) | Test rows | Marked uncertain | Accuracy when answered |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {label} | {row['n']} | {_pct(row['abstention_rate'])} | "
              f"{_pct(row['answered_accuracy'])} |"
              for label, row in u["per_class_test"].items()]

    parity = report.get("serving_parity")
    if parity:
        lines += ["", "## Training and serving parity", "",
                  f"The served bundle was re-scored on CPU (fp32) and compared with the GPU (bf16) "
                  f"evaluation on {parity['rows']} test rows, using the serving calibration.", "",
                  "| Agreement | Value |", "| --- | --- |",
                  f"| Top label | {_pct(parity['label_agreement'])} |",
                  f"| Top-3 order | {_pct(parity['top3_order_agreement'])} |",
                  f"| Uncertain flag | {_pct(parity['uncertain_agreement'])} |",
                  f"| Largest probability difference | {parity['max_abs_probability_difference']:.3f} |",
                  f"| CPU test macro F1 / accuracy | {parity['compared_macro_f1']:.3f} / "
                  f"{_pct(parity['compared_accuracy'])} |"]

    if report["leakage"]:
        lines += [
            "",
            "## Leakage study: row-level split",
            "",
            "Splitting rows instead of unique descriptions puts identical descriptions "
            "in both train and test. Test rows are grouped by whether their "
            "description also occurs in train, and whether the description carries "
            "conflicting labels anywhere in the data. \"Seen, consistent\" isolates "
            "memorisation; \"conflicting\" rows have no single correct answer.",
            "",
            "| Model | All test | Seen in train | Seen, consistent label | Unseen "
            "| Conflicting label | Macro F1 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in report["leakage"]:
            cells = [f"{_pct(r[k]['accuracy'])} (n={r[k]['n']})"
                     for k in ("all", "seen_in_train", "seen_consistent_label", "unseen",
                               "conflicting_label")]
            lines.append(f"| {r['name']} | {' | '.join(cells)} | {r['macro_f1']:.3f} |")
    return "\n".join(lines) + "\n"
