"""Compare models across different data splits of the same source.

Training seeds measure optimiser randomness; repeating the split measures how much the
result depends on which descriptions land in the test set. Both still use one source and
one labeling process, so this is not external validation.
"""

from pathlib import Path

import numpy as np

from triage_training.artifacts import utc_timestamp, write_json
from triage_training.config import TrainingConfig
from triage_training.data import load_dataset
from triage_training.report import load_run
from triage_training.stats import accuracy, macro_f1


def _model_kind(run) -> str:
    return "baseline" if run.manifest["model"] == "tfidf_logistic_regression" else "pubmedbert"


def summarize_splits(config: TrainingConfig, run_dirs: list[Path], out_dir: Path) -> dict:
    rows = []
    for run_dir in run_dirs:
        run = load_run(run_dir)
        if not run.deduplicated:
            raise ValueError(f"{run_dir.name} does not use a deduplicated split.")
        split_seed = run.manifest["dataset"]["split_seed"]
        test = load_dataset(config.paths.data_dir / f"dedup_seed{split_seed}").splits["test"]
        source = test.set_index("source_row").loc[run.splits["test"]["source_row"], "label_source"]
        y_true, y_pred = run.splits["test"]["y_true"], run.predictions()
        direct = source.to_numpy() == "direct"
        rows.append({
            "model": _model_kind(run), "split_seed": split_seed, "run": run_dir.name,
            "n_test": int(len(y_true)), "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "direct_mapped_accuracy": accuracy(y_true[direct], y_pred[direct]),
        })

    summary = {}
    for model in ("baseline", "pubmedbert"):
        subset = [r for r in rows if r["model"] == model]
        if len(subset) > 1:
            summary[model] = {metric: {"mean": float(np.mean([r[metric] for r in subset])),
                                       "sd": float(np.std([r[metric] for r in subset], ddof=1)),
                                       "min": float(min(r[metric] for r in subset)),
                                       "max": float(max(r[metric] for r in subset))}
                              for metric in ("accuracy", "macro_f1", "direct_mapped_accuracy")}
            summary[model]["splits"] = len(subset)
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["split_seed"], {})[r["model"]] = r["macro_f1"]
    gains = [v["pubmedbert"] - v["baseline"] for v in by_seed.values() if len(v) == 2]
    if gains:
        summary["macro_f1_gain"] = {"mean": float(np.mean(gains)), "min": float(min(gains)),
                                    "splits": len(gains)}

    result = {"created_utc": utc_timestamp(), "runs": sorted(rows, key=lambda r: (
        r["split_seed"], r["model"])), "summary": summary}
    write_json(out_dir / "split_variation.json", result)
    (out_dir / "split_variation.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# Variation Across Data Splits",
        "",
        "Each split seed reshuffles which unique descriptions fall into train, validation, "
        "and test; training settings are unchanged. This measures dependence on the split, "
        "not performance on other data: all splits share one source and one labeling "
        "process.",
        "",
        "| Split seed | Model | Test n | Accuracy | Macro F1 | Direct-mapped accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["runs"]:
        lines.append(f"| {r['split_seed']} | {r['model']} | {r['n_test']} | {r['accuracy']:.3f} | "
                     f"{r['macro_f1']:.3f} | {r['direct_mapped_accuracy']:.3f} |")
    summary = result["summary"]
    if any(k in summary for k in ("baseline", "pubmedbert")):
        lines += ["", "| Model | Splits | Macro F1 mean ± SD (min–max) | "
                      "Direct-mapped accuracy mean ± SD (min–max) |", "| --- | --- | --- | --- |"]
        for model in ("baseline", "pubmedbert"):
            if model in summary:
                s = summary[model]
                f1, direct = s["macro_f1"], s["direct_mapped_accuracy"]
                lines.append(f"| {model} | {s['splits']} | {f1['mean']:.3f} ± {f1['sd']:.3f} "
                             f"({f1['min']:.3f}–{f1['max']:.3f}) | {direct['mean']:.3f} ± "
                             f"{direct['sd']:.3f} ({direct['min']:.3f}–{direct['max']:.3f}) |")
    if "macro_f1_gain" in summary:
        g = summary["macro_f1_gain"]
        lines += ["", f"PubMedBERT minus baseline macro F1 across {g['splits']} splits: mean "
                      f"{g['mean']:+.3f}, smallest {g['min']:+.3f}."]
    return "\n".join(lines) + "\n"
