"""Command line entry point: python -m triage_training <command> [options]."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from triage_training.artifacts import utc_timestamp
from triage_training.config import DEFAULT_CONFIG, load_config
from triage_training.data import dataset_name, load_dataset, prepare_dataset, save_dataset

logger = logging.getLogger("triage_training")


def _summary(metrics: dict) -> str:
    return ", ".join(
        f"{split} acc={metrics[split]['accuracy']:.4f} macro_f1={metrics[split]['macro_f1']:.4f}"
        for split in ("val", "test")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m triage_training", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="clean, label, deduplicate, and split")
    prepare.add_argument("--keep-duplicates", action="store_true",
                         help="split rows instead of unique descriptions (leakage study only)")

    for name, help_text in (("baseline", "train the TF-IDF baseline"),
                            ("finetune", "fine-tune the BERT classifier")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", default="dedup_seed42",
                             help="prepared dataset name under paths.data_dir")
        command.add_argument("--seed", type=int, help="override the training seed")
    commands.choices["finetune"].add_argument(
        "--smoke", action="store_true", help="tiny CPU-friendly run to check the pipeline")

    evaluate = commands.add_parser("evaluate-model", help="score an exported model bundle")
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--dataset", default="dedup_seed42")
    evaluate.add_argument("--name", required=True, help="run name prefix, e.g. servingcpu")

    report = commands.add_parser("report", help="statistical evaluation report")
    report.add_argument("--model", type=Path, required=True, help="primary fine-tuned run")
    report.add_argument("--baseline", type=Path, required=True, help="baseline run, same split")
    report.add_argument("--seed-run", type=Path, action="append", default=[],
                        help="extra fine-tuned runs with other seeds")
    report.add_argument("--rowlevel", type=Path, action="append", default=[],
                        help="runs trained on the row-level split")
    report.add_argument("--out", type=Path, default=Path("reports"))

    gate = commands.add_parser("validate-model", help="release gate for a model bundle")
    gate.add_argument("--model-dir", type=Path, required=True)
    gate.add_argument("--evaluation", type=Path, default=Path("reports/model_evaluation.json"))
    gate.add_argument("--criteria", type=Path, default=Path("configs/release_criteria.yaml"))
    gate.add_argument("--out", type=Path, default=Path("reports"))

    drift = commands.add_parser("monitor-drift", help="compare a /metrics snapshot with reference")
    drift.add_argument("--snapshot", type=Path, required=True, help="JSON saved from GET /metrics")
    drift.add_argument("--reference", type=Path, required=True,
                       help="monitoring_reference.json from the served bundle")
    drift.add_argument("--thresholds", type=Path, default=Path("configs/monitoring.yaml"))
    drift.add_argument("--out", type=Path, default=Path("reports"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "prepare-data":
            prepared = prepare_dataset(config, keep_duplicates=args.keep_duplicates)
            out_dir = config.paths.data_dir / dataset_name(config, args.keep_duplicates)
            save_dataset(prepared, out_dir)
            logger.info("Saved %s to %s", prepared.report["split_sizes"], out_dir)
            logger.info("Audit: %s", prepared.report["audit"])
            logger.info("Overlap: %s", prepared.report["overlap"])
            return

        if args.command == "validate-model":
            from triage_training.release import run_release_validation
            result = run_release_validation(args.model_dir.resolve(), args.evaluation,
                                            args.criteria, args.out)
            logger.info("Release decision: %s (report in %s)", result["decision"], args.out)
            if result["decision"] != "PASS":
                sys.exit(1)
            return

        if args.command == "monitor-drift":
            from triage_training.drift import run_drift_check
            result = run_drift_check(args.snapshot, args.reference, args.thresholds, args.out)
            logger.info("Drift status: %s (report in %s)", result["status"], args.out)
            sys.exit({"WARN": 1, "ALERT": 2}.get(result["status"], 0))

        if args.command == "report":
            from triage_training.report import build_report
            build_report(config, args.model, args.baseline, args.seed_run, args.rowlevel, args.out)
            logger.info("Wrote evaluation report to %s", args.out)
            return

        dataset = load_dataset(config.paths.data_dir / args.dataset)
        if args.command == "evaluate-model":
            from triage_training.finetune import evaluate_export
            run_dir = config.paths.runs_dir / f"{args.name}_{args.dataset}_{utc_timestamp()}"
            metrics = evaluate_export(args.model_dir.resolve(), dataset, run_dir)
        elif args.command == "baseline":
            from triage_training.baseline import run_baseline
            seed = config.split.seed if args.seed is None else args.seed
            run_dir = config.paths.runs_dir / (
                f"baseline_{args.dataset}_trainseed{seed}_{utc_timestamp()}")
            metrics = run_baseline(dataset, config.baseline, seed, run_dir)
        else:
            from triage_training.finetune import run_finetune
            finetune = config.finetune if args.seed is None else replace(
                config.finetune, seed=args.seed)
            tag = "smoke" if args.smoke else "pubmedbert"
            run_dir = config.paths.runs_dir / (
                f"{tag}_{args.dataset}_trainseed{finetune.seed}_{utc_timestamp()}")
            metrics = run_finetune(dataset, finetune, run_dir, smoke=args.smoke)
        logger.info("%s -> %s", _summary(metrics), run_dir)
    except ValueError as exc:  # ConfigError and DataValidationError included
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
