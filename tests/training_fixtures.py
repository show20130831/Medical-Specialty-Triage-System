"""Synthetic MTSamples-shaped data so training tests never need the real dataset."""

from pathlib import Path

import pandas as pd
import yaml

from triage_training.config import DEFAULT_CONFIG, load_config

# One directly mapped source specialty per served label.
SPECIALTY_FOR_LABEL = {
    "Cardiovascular & Pulmonary": "Cardiovascular / Pulmonary",
    "ENT & Ophthalmology": "Ophthalmology",
    "Gastroenterology": "Gastroenterology",
    "General Medicine": "Internal Medicine",
    "Neurology": "Neurology",
    "OB-GYN": "Obstetrics / Gynecology",
    "Orthopedics": "Orthopedic",
    "Urology": "Urology",
}


def synthetic_rows(per_label: int = 12) -> list[dict]:
    return [
        {"description": f"synthetic {label} finding number {i}",
         "medical_specialty": specialty, "keywords": ""}
        for label, specialty in SPECIALTY_FOR_LABEL.items()
        for i in range(per_label)
    ]


def write_config(tmp_path: Path, rows: list[dict], **data_overrides):
    raw_csv = tmp_path / "mtsamples.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False, encoding="cp1252")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["data"].update({"raw_csv": str(raw_csv), "min_class_size": 5, **data_overrides})
    raw["paths"] = {"data_dir": str(tmp_path / "data"), "runs_dir": str(tmp_path / "runs")}
    raw["baseline"]["tfidf"]["min_df"] = 1
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)
