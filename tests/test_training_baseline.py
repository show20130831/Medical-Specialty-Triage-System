import json

import numpy as np

from training_fixtures import synthetic_rows, write_config
from triage_training.__main__ import main
from triage_training.baseline import run_baseline
from triage_training.data import prepare_dataset


def test_baseline_writes_metrics_and_text_free_predictions(tmp_path):
    config = write_config(tmp_path, synthetic_rows())
    dataset = prepare_dataset(config)
    run_dir = tmp_path / "run"
    metrics = run_baseline(dataset, config.baseline, seed=0, run_dir=run_dir)

    assert set(metrics) == {"val", "test"}
    assert 0 <= metrics["test"]["macro_f1"] <= 1
    saved = np.load(run_dir / "predictions_test.npz")
    assert set(saved.files) == {"source_row", "y_true", "scores"}
    assert saved["scores"].shape == (len(dataset.splits["test"]), 8)
    np.testing.assert_allclose(saved["scores"].sum(axis=1), 1.0)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["deduplicated"] is True
    assert "synthetic" not in (run_dir / "metrics.json").read_text(encoding="utf-8")


def test_cli_prepares_data_then_trains_baseline(tmp_path):
    write_config(tmp_path, synthetic_rows())
    config_path = str(tmp_path / "train.yaml")
    main(["--config", config_path, "prepare-data"])
    assert (tmp_path / "data" / "dedup_seed42" / "data_report.json").is_file()
    main(["--config", config_path, "baseline"])
    assert len(list((tmp_path / "runs").glob("baseline_dedup_seed42_*/metrics.json"))) == 1
