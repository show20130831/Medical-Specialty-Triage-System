"""Strict YAML configuration for the training pipeline."""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "train.yaml"


class ConfigError(ValueError):
    """The training configuration is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    raw_csv: Path
    encoding: str
    taxonomy: Path
    min_class_size: int


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    val_size: float
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    runs_dir: Path


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    max_df: float
    sublinear_tf: bool


@dataclass(frozen=True)
class LogisticRegressionConfig:
    C: float
    class_weight: str
    max_iter: int


@dataclass(frozen=True)
class BaselineConfig:
    tfidf: TfidfConfig
    logistic_regression: LogisticRegressionConfig


@dataclass(frozen=True)
class FinetuneConfig:
    model_name: str
    max_length: int
    epochs: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig
    split: SplitConfig
    paths: PathsConfig
    baseline: BaselineConfig
    finetune: FinetuneConfig


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping.")
    names = {f.name for f in fields(cls)}
    missing, unknown = names - raw.keys(), raw.keys() - names
    if missing or unknown:
        raise ConfigError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}.")
    hints = get_type_hints(cls)
    values = {}
    for name in names:
        kind, value = hints[name], raw[name]
        if is_dataclass(kind):
            value = _build(kind, value, f"{where}.{name}")
        elif kind is Path:
            value = REPO_ROOT / value
        elif kind == tuple[int, int]:
            value = tuple(value)
        elif kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not is_dataclass(kind) and kind not in (Path, tuple[int, int]) \
                and type(value) is not kind:
            raise ConfigError(f"{where}.{name} must be {kind.__name__}.")
        values[name] = value
    return cls(**values)


def load_config(path: str | Path = DEFAULT_CONFIG) -> TrainingConfig:
    """Load and validate the training configuration."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read training config {path}: {exc}") from None
    config = _build(TrainingConfig, raw, "config")
    for name in ("test_size", "val_size"):
        if not 0 < getattr(config.split, name) < 1:
            raise ConfigError(f"split.{name} must be between 0 and 1.")
    if not 0 <= config.finetune.warmup_ratio < 1:
        raise ConfigError("finetune.warmup_ratio must be in [0, 1).")
    return config
