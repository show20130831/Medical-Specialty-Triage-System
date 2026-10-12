import pandas as pd
import pytest

from training_fixtures import synthetic_rows, write_config
from triage_training.config import load_config
from triage_training.data import (
    DataValidationError, clean_and_label, deduplicate, description_key, load_dataset,
    overlap_counts, prepare_dataset, save_dataset, validate_splits,
)
from triage_training.taxonomy import assign_label, clean_keywords, load_taxonomy

TAXONOMY = load_taxonomy(load_config().data.taxonomy)


def test_description_key_ignores_case_punctuation_and_spacing():
    assert description_key("  Knee   PAIN, left.") == description_key("knee pain left")


def test_keyword_labeling_strips_disclaimer_and_scores_terms():
    assert clean_keywords("Cholecystectomy, NOTE: thesis disclaimer") == "cholecystectomy,"
    assert assign_label("Surgery", "laparoscopic cholecystectomy", TAXONOMY) == (
        "Gastroenterology", "keyword")
    assert assign_label("Surgery", "nothing relevant", TAXONOMY) == (None, "keyword")
    assert assign_label("Urology", "", TAXONOMY) == ("Urology", "direct")
    assert assign_label("Autopsy", "", TAXONOMY) == (None, "unmapped")


def _labeled(rows):
    raw = pd.DataFrame(rows)
    raw.insert(0, "source_row", range(len(raw)))
    return clean_and_label(raw, TAXONOMY)


def test_clean_and_label_drops_blank_and_unmapped_rows():
    df, audit = _labeled([
        {"description": "x", "medical_specialty": " Urology ", "keywords": None},
        {"description": "  ", "medical_specialty": "Urology", "keywords": ""},
        {"description": "y", "medical_specialty": "Autopsy", "keywords": ""},
    ])
    assert df["label"].tolist() == ["Urology"]
    assert audit["unmapped_specialty_rows"] == 1
    assert audit["rows_with_description_and_specialty"] == 2


def test_deduplicate_prefers_direct_labels_and_drops_ties():
    df, _ = _labeled([
        {"description": "Same text.", "medical_specialty": "Urology", "keywords": ""},
        {"description": "same TEXT", "medical_specialty": "Urology", "keywords": ""},
        {"description": "Gallbladder removal", "medical_specialty": "Surgery",
         "keywords": "cholecystectomy"},
        {"description": "gallbladder removal", "medical_specialty": "Orthopedic", "keywords": ""},
        {"description": "Ambiguous", "medical_specialty": "Neurology", "keywords": ""},
        {"description": "ambiguous", "medical_specialty": "Orthopedic", "keywords": ""},
    ])
    out, audit = deduplicate(df)
    assert out["label"].tolist() == ["Urology", "Orthopedics"]
    assert out["source_row"].tolist() == [0, 2]
    assert audit == {"duplicate_rows_removed": 3, "conflicting_descriptions": 2,
                     "conflicting_descriptions_dropped": 1, "unique_descriptions": 2}


def test_prepare_dataset_produces_disjoint_validated_splits(tmp_path):
    rows = synthetic_rows()
    rows += [dict(rows[0], description=rows[0]["description"].upper())] * 3
    config = write_config(tmp_path, rows)
    prepared = prepare_dataset(config)
    assert prepared.report["audit"]["duplicate_rows_removed"] == 3
    assert not any(prepared.report["overlap"].values())
    assert sum(prepared.report["split_sizes"].values()) == 96
    assert prepared.labels == TAXONOMY.labels

    save_dataset(prepared, tmp_path / "prepared")
    reloaded = load_dataset(tmp_path / "prepared")
    assert reloaded.report == prepared.report
    assert all(reloaded.splits[n]["source_row"].tolist() == prepared.splits[n]["source_row"].tolist()
               for n in ("train", "val", "test"))


def test_keep_duplicates_reports_overlap_instead_of_failing(tmp_path):
    rows = synthetic_rows()
    rows += [row for row in rows if row["medical_specialty"] == "Urology"] * 3
    prepared = prepare_dataset(write_config(tmp_path, rows), keep_duplicates=True)
    assert not prepared.report["deduplicated"]
    assert prepared.report["overlap"]["train_test_shared_descriptions"] > 0


def test_rare_class_fails_validation(tmp_path):
    rows = [r for r in synthetic_rows() if r["medical_specialty"] != "Urology"]
    rows += synthetic_rows(per_label=2)[-2:]
    with pytest.raises(DataValidationError, match="min_class_size"):
        prepare_dataset(write_config(tmp_path, rows))


def test_validate_splits_rejects_shared_descriptions(tmp_path):
    prepared = prepare_dataset(write_config(tmp_path, synthetic_rows()))
    splits = dict(prepared.splits)
    leaked = splits["train"].iloc[[0]].assign(source_row=10_000)
    splits["test"] = pd.concat([splits["test"], leaked])
    assert overlap_counts(splits)["train_test_shared_descriptions"] == 1
    with pytest.raises(DataValidationError, match="share descriptions"):
        validate_splits(splits, prepared.labels, require_disjoint=True)


def test_missing_raw_columns_are_reported(tmp_path):
    config = write_config(tmp_path, synthetic_rows())
    pd.DataFrame({"description": ["x"]}).to_csv(config.data.raw_csv, index=False)
    with pytest.raises(DataValidationError, match="missing columns"):
        prepare_dataset(config)
