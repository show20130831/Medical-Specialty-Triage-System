"""Fine-tune a BERT encoder on prepared description splits and export a serving bundle."""

import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from triage_training.artifacts import environment, git_state, utc_timestamp, write_json
from triage_training.config import FinetuneConfig
from triage_training.data import PreparedDataset
from triage_training.evaluation import classification_metrics, save_predictions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _precision_flags(torch) -> dict[str, bool]:
    """bf16 on Ampere+ GPUs (e.g. RTX 4090), fp16 on older GPUs (e.g. T4), fp32 on CPU."""
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16 = torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def _resolve_revision(model_name: str) -> str | None:
    """Pin the exact Hub commit of the pretrained weights; local directories have none."""
    if Path(model_name).is_dir():
        return None
    from huggingface_hub import HfApi
    return HfApi().model_info(model_name).sha


def run_finetune(
    dataset: PreparedDataset, config: FinetuneConfig, run_dir: Path, smoke: bool = False
) -> dict:
    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
    )

    set_seed(config.seed)
    splits = dataset.splits
    epochs = config.epochs
    if smoke:
        # A pipeline check that runs on CPU in about a minute; its metrics are meaningless.
        splits = {name: part.groupby("label_id").head(2) for name, part in splits.items()}
        epochs = 1

    labels = dataset.labels
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in id2label.items()}
    revision = _resolve_revision(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, revision=revision, num_labels=len(labels),
        id2label=id2label, label2id=label2id,
    )

    class EncodedSplit(torch.utils.data.Dataset):
        def __init__(self, part):
            self.encodings = tokenizer(part["description"].tolist(), truncation=True,
                                       max_length=config.max_length)
            self.labels = part["label_id"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {key: values[i] for key, values in self.encodings.items()}
            item["labels"] = self.labels[i]
            return item

    encoded = {name: EncodedSplit(part) for name, part in splits.items()}
    token_lengths = np.array([len(ids) for ids in encoded["train"].encodings["input_ids"]])

    def compute_metrics(prediction):
        y_pred = prediction.predictions.argmax(axis=-1)
        y_true = prediction.label_ids
        return {"accuracy": float((y_pred == y_true).mean()),
                "macro_f1": classification_metrics(
                    y_true, np.eye(len(labels))[y_pred], labels)["macro_f1"]}

    checkpoint_dir = run_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoint_dir), num_train_epochs=epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_steps=config.warmup_ratio,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        seed=config.seed, data_seed=config.seed, logging_steps=10, report_to="none",
        **_precision_flags(torch),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()

    metrics, logits = {}, {}
    for name in ("val", "test"):
        output = trainer.predict(encoded[name])
        logits[name] = np.asarray(output.predictions, dtype=np.float32)
        y_true = splits[name]["label_id"].to_numpy()
        metrics[name] = classification_metrics(y_true, _softmax(logits[name]), labels)
        save_predictions(run_dir / f"predictions_{name}.npz",
                         splits[name]["source_row"].to_numpy(), y_true, logits[name])
    best = trainer.state.best_model_checkpoint
    metrics["best_val_macro_f1"] = trainer.state.best_metric
    metrics["best_epoch"] = next(
        (log["epoch"] for log in trainer.state.log_history
         if log.get("eval_macro_f1") == trainer.state.best_metric), None)

    export_dir = run_dir / "model"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))
    # Pickled training arguments are not needed for inference and should not be shipped.
    (export_dir / "training_args.bin").unlink(missing_ok=True)
    manifest = {
        "model": "finetuned_encoder",
        "created_utc": utc_timestamp(),
        "smoke": smoke,
        "base_model": {"name": config.model_name, "revision": revision},
        "config": asdict(config) | {"epochs": epochs},
        "precision": _precision_flags(torch),
        "dataset": {k: dataset.report[k] for k in ("raw_csv_sha256", "taxonomy_sha256",
                                                   "deduplicated", "split_seed", "split_sizes")},
        "train_fraction_truncated": float((token_lengths >= config.max_length).mean()),
        "best_checkpoint": Path(best).name if best else None,
        "git": git_state(),
        "environment": environment(),
    }
    # Serving contract read by triage_system.model_assets.
    write_json(export_dir / "training_config.json", {
        "input_type": "description", "max_length": config.max_length,
        "labels": list(labels), **manifest["config"],
    })
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "trainer_log.json", trainer.state.log_history)
    for name in ("metrics.json", "run_manifest.json"):
        shutil.copy2(run_dir / name, export_dir / name)
    write_json(export_dir / "data_report.json", dataset.report)

    reference = logits["val"][:4]
    del trainer, model
    write_json(export_dir / "reload_check.json",
               _reload_check(export_dir, splits["val"]["description"].tolist()[:4],
                             reference, config.max_length))
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return metrics


def evaluate_export(model_dir: Path, dataset: PreparedDataset, run_dir: Path,
                    batch_size: int = 32) -> dict:
    """Score an exported model on prepared val/test splits, e.g. a model trained elsewhere."""
    import json

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    if tuple(model.config.id2label[i] for i in range(len(dataset.labels))) != dataset.labels:
        raise ValueError("Model labels do not match the prepared dataset labels.")
    max_length = json.loads((model_dir / "training_config.json").read_text())["max_length"]

    metrics = {}
    for name in ("val", "test"):
        part = dataset.splits[name]
        texts = part["description"].tolist()
        batches = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], truncation=True,
                              max_length=max_length, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                batches.append(model(**batch).logits.float().cpu().numpy())
        logits = np.concatenate(batches)
        y_true = part["label_id"].to_numpy()
        metrics[name] = classification_metrics(y_true, _softmax(logits), dataset.labels)
        save_predictions(run_dir / f"predictions_{name}.npz",
                         part["source_row"].to_numpy(), y_true, logits)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_manifest.json", {
        "model": "finetuned_encoder",
        "evaluated_export": str(model_dir),
        "created_utc": utc_timestamp(),
        "dataset": {k: dataset.report[k] for k in ("raw_csv_sha256", "taxonomy_sha256",
                                                   "deduplicated", "split_seed", "split_sizes")},
        "git": git_state(),
        "environment": environment(),
    })
    return metrics


def _reload_check(export_dir: Path, texts: list[str], reference: np.ndarray,
                  max_length: int) -> dict:
    """Reload the export offline on CPU and confirm it reproduces the trained logits."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        export_dir, local_files_only=True, use_safetensors=True).to("cpu").eval()
    batch = tokenizer(texts, truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
    with torch.inference_mode():
        reloaded = model(**batch).logits.float().numpy()
    max_diff = float(np.abs(reloaded - reference).max())
    # Mixed-precision evaluation on GPU differs slightly from fp32 CPU inference.
    passed = bool((reloaded.argmax(1) == reference.argmax(1)).all() and max_diff < 0.1)
    if not passed:
        raise RuntimeError(f"Exported model does not reproduce trained logits (max diff {max_diff}).")
    return {"passed": passed, "device": "cpu", "samples": len(texts), "max_abs_logit_diff": max_diff}
