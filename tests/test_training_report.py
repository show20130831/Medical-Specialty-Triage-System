import json

import pytest

from training_fixtures import synthetic_rows, write_config
from triage_training.baseline import run_baseline
from triage_training.data import dataset_name, prepare_dataset, save_dataset
from triage_training.report import build_report


@pytest.fixture
def runs(tmp_path):
    rows = synthetic_rows(per_label=15)
    rows += [row for row in rows if row["medical_specialty"] == "Urology"][:5] * 2
    config = write_config(tmp_path, rows)
    out = {}
    for keep in (False, True):
        dataset = prepare_dataset(config, keep_duplicates=keep)
        name = dataset_name(config, keep)
        save_dataset(dataset, config.paths.data_dir / name)
        for seed in (0, 1):
            run_dir = config.paths.runs_dir / f"{name}_{seed}"
            run_baseline(dataset, config.baseline, seed, run_dir)
            out[(keep, seed)] = run_dir
    return config, out


def test_report_combines_comparison_calibration_and_leakage(runs, tmp_path):
    config, run_dirs = runs
    (run_dirs[(False, 0)] / "model").mkdir()
    out_dir = tmp_path / "reports"
    report = build_report(config, run_dirs[(False, 0)], run_dirs[(False, 1)],
                          [run_dirs[(False, 1)]], [run_dirs[(True, 0)]], out_dir)

    assert {"models", "comparison", "calibration", "leakage", "label_source"} <= report.keys()
    assert report["seed_variation"]["runs"] == 2
    assert report["calibration"]["temperature"] > 0
    leak = report["leakage"][0]
    assert leak["seen_in_train"]["n"] + leak["unseen"]["n"] == leak["all"]["n"]

    markdown = (out_dir / "model_evaluation.md").read_text(encoding="utf-8")
    assert "## Leakage study" in markdown and "McNemar" in markdown
    assert "synthetic" not in markdown
    assert json.loads((out_dir / "model_evaluation.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (run_dirs[(False, 0)] / "model" / "calibration.json").read_text(encoding="utf-8"))
    assert set(calibration) == {"temperature", "uncertain_threshold", "target_answered_accuracy"}


def test_report_rejects_runs_from_the_wrong_split(runs, tmp_path):
    config, run_dirs = runs
    with pytest.raises(ValueError, match="deduplicated split"):
        build_report(config, run_dirs[(True, 0)], run_dirs[(False, 0)], [], [], tmp_path)
    with pytest.raises(ValueError, match="row-level split"):
        build_report(config, run_dirs[(False, 0)], run_dirs[(False, 1)], [],
                     [run_dirs[(False, 1)]], tmp_path)
