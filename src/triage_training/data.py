"""Clean, label, deduplicate, split, and validate the MTSamples description data."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from triage_training.artifacts import sha256_file, write_json
from triage_training.config import TrainingConfig
from triage_training.taxonomy import Taxonomy, assign_label, load_taxonomy

REQUIRED_COLUMNS = ("description", "medical_specialty", "keywords")
SPLITS = ("train", "val", "test")
SAVED_COLUMNS = ["source_row", "description", "label", "label_id", "label_source"]


class DataValidationError(ValueError):
    """Prepared data violates an invariant the model or evaluation relies on."""


@dataclass(frozen=True)
class PreparedDataset:
    splits: dict[str, pd.DataFrame]
    labels: tuple[str, ...]
    report: dict


def description_key(text: str) -> str:
    """Normalise case, punctuation, and whitespace so trivially different copies match."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def load_raw(path: Path, encoding: str) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(
            f"Raw dataset not found at {path}. Download mtsamples.csv from Kaggle first."
        )
    header = pd.read_csv(path, encoding=encoding, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise DataValidationError(f"Raw dataset is missing columns: {missing}")
    df = pd.read_csv(path, encoding=encoding, usecols=list(REQUIRED_COLUMNS))
    df.insert(0, "source_row", range(len(df)))
    return df


def clean_and_label(raw: pd.DataFrame, taxonomy: Taxonomy) -> tuple[pd.DataFrame, dict]:
    """Drop unusable rows and assign one of the eight groups to each remaining row."""
    audit = {"raw_rows": len(raw)}
    df = raw.dropna(subset=["medical_specialty"])
    df = df[df["description"].fillna("").str.strip() != ""].copy()
    df["keywords"] = df["keywords"].fillna("")
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    audit["rows_with_description_and_specialty"] = len(df)

    assigned = [
        assign_label(specialty, keywords, taxonomy)
        for specialty, keywords in zip(df["medical_specialty"], df["keywords"])
    ]
    df["label"] = [label for label, _ in assigned]
    df["label_source"] = [source for _, source in assigned]
    audit["unmapped_specialty_rows"] = int((df["label_source"] == "unmapped").sum())
    audit["keyword_rows_without_match"] = int(
        ((df["label_source"] == "keyword") & df["label"].isna()).sum()
    )
    df = df[df["label"].notna()].copy()
    audit["labeled_rows"] = len(df)
    audit["labeled_rows_by_source"] = df["label_source"].value_counts().to_dict()
    return df, audit


def _resolve_group(group: pd.DataFrame) -> pd.Series | None:
    """Pick one label for copies of one description, or None when it stays ambiguous.

    Direct specialty mappings are trusted over keyword rules; a tie among the
    trusted labels means the description cannot be labeled consistently.
    """
    first = group.iloc[0].copy()
    if group["label"].nunique() == 1:
        return first
    direct = group.loc[group["label_source"] == "direct", "label"]
    votes = (direct if len(direct) else group["label"]).value_counts()
    if len(votes) > 1 and votes.iloc[0] == votes.iloc[1]:
        return None
    first["label"] = votes.index[0]
    first["label_source"] = "direct" if len(direct) else "keyword"
    return first


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one row per normalised description and resolve conflicting labels."""
    df = df.assign(key=df["description"].map(description_key))
    groups = df.groupby("key", sort=False)
    label_counts = groups["label"].nunique()
    resolved = [row for _, group in groups if (row := _resolve_group(group)) is not None]
    out = pd.DataFrame(resolved).sort_values("source_row").reset_index(drop=True)
    audit = {
        "duplicate_rows_removed": int(len(df) - df["key"].nunique()),
        "conflicting_descriptions": int((label_counts > 1).sum()),
        "conflicting_descriptions_dropped": int(df["key"].nunique() - len(out)),
        "unique_descriptions": len(out),
    }
    return out.drop(columns="key"), audit


def split_dataset(df: pd.DataFrame, config: TrainingConfig) -> dict[str, pd.DataFrame]:
    s = config.split
    train_val, test = train_test_split(
        df, test_size=s.test_size, random_state=s.seed, stratify=df["label_id"])
    train, val = train_test_split(
        train_val, test_size=s.val_size, random_state=s.seed, stratify=train_val["label_id"])
    return {"train": train, "val": val, "test": test}


def overlap_counts(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    keys = {name: set(part["description"].map(description_key)) for name, part in splits.items()}
    return {
        f"{a}_{b}_shared_descriptions": len(keys[a] & keys[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }


def validate_splits(
    splits: dict[str, pd.DataFrame], labels: tuple[str, ...], require_disjoint: bool
) -> None:
    """Fail loudly on any split that would make training or evaluation misleading."""
    rows = [set(splits[name]["source_row"]) for name in SPLITS]
    if rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2]:
        raise DataValidationError("A source row appears in more than one split.")
    for name in SPLITS:
        part = splits[name]
        if part["description"].isna().any() or (part["description"].str.strip() == "").any():
            raise DataValidationError(f"{name} split contains blank descriptions.")
        if set(part["label"]) != set(labels):
            raise DataValidationError(f"{name} split does not contain all {len(labels)} labels.")
        if not part["label_id"].between(0, len(labels) - 1).all():
            raise DataValidationError(f"{name} split has label ids outside the label range.")
    if require_disjoint and any(overlap_counts(splits).values()):
        raise DataValidationError("Deduplicated splits still share descriptions.")


def prepare_dataset(config: TrainingConfig, keep_duplicates: bool = False) -> PreparedDataset:
    """Run the full data pipeline. keep_duplicates reproduces the legacy row-level split."""
    taxonomy = load_taxonomy(config.data.taxonomy)
    df, audit = clean_and_label(load_raw(config.data.raw_csv, config.data.encoding), taxonomy)
    if not keep_duplicates:
        df, dedup_audit = deduplicate(df)
        audit.update(dedup_audit)

    counts = df["label"].value_counts()
    if set(counts.index) != set(taxonomy.labels):
        raise DataValidationError(f"Expected labels {taxonomy.labels}, found {sorted(counts.index)}.")
    rare = counts[counts < config.data.min_class_size]
    if not rare.empty:
        raise DataValidationError(f"Classes below min_class_size: {rare.to_dict()}")
    df["label_id"] = df["label"].map({label: i for i, label in enumerate(taxonomy.labels)})

    splits = split_dataset(df, config)
    validate_splits(splits, taxonomy.labels, require_disjoint=not keep_duplicates)
    report = {
        "deduplicated": not keep_duplicates,
        "raw_csv_sha256": sha256_file(config.data.raw_csv),
        "taxonomy_sha256": sha256_file(config.data.taxonomy),
        "split_seed": config.split.seed,
        "labels": list(taxonomy.labels),
        "audit": audit,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "class_counts": {
            name: part["label"].value_counts().sort_index().to_dict()
            for name, part in splits.items()
        },
        "label_source_counts": {
            name: part["label_source"].value_counts().to_dict() for name, part in splits.items()
        },
        "description_chars": {
            name: part["description"].str.len().describe(percentiles=[0.5, 0.95])
            .loc[["50%", "95%", "max"]].round(1).to_dict()
            for name, part in splits.items()
        },
        "overlap": overlap_counts(splits),
    }
    return PreparedDataset(splits=splits, labels=taxonomy.labels, report=report)


def dataset_name(config: TrainingConfig, keep_duplicates: bool) -> str:
    return f"{'rowlevel' if keep_duplicates else 'dedup'}_seed{config.split.seed}"


def save_dataset(prepared: PreparedDataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in prepared.splits.items():
        part[SAVED_COLUMNS].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    write_json(out_dir / "data_report.json", prepared.report)


def load_dataset(data_dir: Path) -> PreparedDataset:
    report_path = data_dir / "data_report.json"
    if not report_path.is_file():
        raise DataValidationError(f"No prepared dataset at {data_dir}; run prepare-data first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    splits = {
        name: pd.read_csv(data_dir / f"{name}.csv", encoding="utf-8", keep_default_na=False)
        for name in SPLITS
    }
    labels = tuple(report["labels"])
    validate_splits(splits, labels, require_disjoint=report["deduplicated"])
    return PreparedDataset(splits=splits, labels=labels, report=report)
