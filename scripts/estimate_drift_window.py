"""Estimate the minimum drift window size from validation predictions (operations plan §8).

The reference profile is built from the test split, so resampling the test split would
understate false alarms. This draws windows with replacement from the validation split,
which comes from the same distribution but was not used for the reference, runs the
service's drift check on each, and reports how often a window with no drift still
reaches WARN or ALERT. Only aggregate rates are written.

    python scripts/estimate_drift_window.py --run <primary run dir> --out reports
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from triage_system.monitoring import (
    LENGTH_EDGES, SCORE_EDGES, DriftThresholds, Window, bucket_index, evaluate_window,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZES = (100, 200, 300, 400, 500, 600)
TARGET_FALSE_ALARM = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="primary fine-tuned run")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    args = parser.parse_args()

    bundle = args.run / "model"
    reference = json.loads((bundle / "monitoring_reference.json").read_text(encoding="utf-8"))
    calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))
    labels = json.loads((bundle / "training_config.json").read_text(encoding="utf-8"))["labels"]
    saved = np.load(args.run / "predictions_val.npz")
    probabilities = softmax(saved["scores"] / calibration["temperature"], axis=1)
    val = pd.read_csv(REPO_ROOT / "outputs/data/dedup_seed42/val.csv", keep_default_na=False)
    lengths = val.set_index("source_row").loc[saved["source_row"], "description"].str.len()
    rows = [(labels[p.argmax()], float(p.max()), int(n))
            for p, n in zip(probabilities, lengths)]

    thresholds = DriftThresholds.from_yaml(REPO_ROOT / "configs/monitoring.yaml")
    rng = random.Random(0)
    results = []
    for size in SIZES:
        statuses = []
        for _ in range(args.resamples):
            window = Window(opened=0)
            for label, score, length in rng.choices(rows, k=size):
                window.labels[label] = window.labels.get(label, 0) + 1
                window.score[bucket_index(score, SCORE_EDGES)] += 1
                window.length[bucket_index(length, LENGTH_EDGES)] += 1
                window.uncertain += int(score < calibration["uncertain_threshold"])
                window.predictions += 1
            statuses.append(evaluate_window(window, reference, thresholds)["status"])
        results.append({"size": size,
                         "false_warn_or_alert": sum(s != "ok" for s in statuses) / len(statuses),
                         "false_alert": statuses.count("alert") / len(statuses)})
    chosen = next((r["size"] for r in results if r["false_warn_or_alert"] < TARGET_FALSE_ALARM),
                  None)

    lines = ["# Drift Window Size Estimate", "",
             f"{args.resamples:,} windows per size drawn with replacement from {len(rows)} "
             "validation predictions (not used to build the test-set reference). Any WARN "
             "or ALERT on these windows is a false alarm. Exploratory: 318 rows cannot "
             "calibrate the rate precisely.", "",
             "| Window size | False WARN or ALERT | False ALERT |", "| --- | --- | --- |"]
    lines += [f"| {r['size']} | {r['false_warn_or_alert']:.1%} | {r['false_alert']:.1%} |"
              for r in results]
    lines += ["", f"Target: fewer than {TARGET_FALSE_ALARM:.0%} false WARN or ALERT. "
              + (f"Smallest size meeting it: **{chosen}**." if chosen else
                 "No tested size meets it; thresholds or sizes need revisiting.")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "drift_window_estimate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "drift_window_estimate.json").write_text(
        json.dumps({"resamples": args.resamples, "validation_rows": len(rows),
                    "results": results, "chosen_min_predictions": chosen}, indent=2) + "\n",
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
