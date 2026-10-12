"""TF-IDF + logistic regression baseline on the same prepared splits."""

from dataclasses import asdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from triage_training.artifacts import environment, git_state, utc_timestamp, write_json
from triage_training.config import BaselineConfig
from triage_training.data import PreparedDataset
from triage_training.evaluation import classification_metrics, save_predictions


def run_baseline(
    dataset: PreparedDataset, config: BaselineConfig, seed: int, run_dir: Path
) -> dict:
    """Fit on train, report val and test, and store per-row probabilities for later stats."""
    vectorizer = TfidfVectorizer(lowercase=True, **asdict(config.tfidf))
    classifier = LogisticRegression(random_state=seed, **asdict(config.logistic_regression))
    train = dataset.splits["train"]
    classifier.fit(vectorizer.fit_transform(train["description"]), train["label_id"])

    metrics = {}
    for name in ("val", "test"):
        part = dataset.splits[name]
        probabilities = classifier.predict_proba(vectorizer.transform(part["description"]))
        y_true = part["label_id"].to_numpy()
        metrics[name] = classification_metrics(y_true, probabilities, dataset.labels)
        save_predictions(run_dir / f"predictions_{name}.npz",
                         part["source_row"].to_numpy(), y_true, probabilities)

    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_manifest.json", {
        "model": "tfidf_logistic_regression",
        "created_utc": utc_timestamp(),
        "seed": seed,
        "config": asdict(config),
        "dataset": {k: dataset.report[k] for k in ("raw_csv_sha256", "taxonomy_sha256",
                                                   "deduplicated", "split_seed")},
        "vocabulary_size": len(vectorizer.vocabulary_),
        "git": git_state(),
        "environment": environment(),
    })
    return metrics
