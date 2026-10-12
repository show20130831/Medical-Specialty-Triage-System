import pytest
import yaml

from triage_training.config import DEFAULT_CONFIG, REPO_ROOT, ConfigError, load_config


def _write(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_repository_config_loads_with_resolved_paths():
    config = load_config()
    assert config.data.taxonomy == REPO_ROOT / "configs" / "taxonomy.yaml"
    assert config.baseline.tfidf.ngram_range == (1, 3)
    assert config.finetune.max_length == 512
    assert isinstance(config.finetune.learning_rate, float)


@pytest.mark.parametrize("mutate, message", [
    (lambda raw: raw["finetune"].pop("seed"), "missing"),
    (lambda raw: raw["split"].update(extra=1), "unknown"),
    (lambda raw: raw["finetune"].update(epochs="8"), "must be int"),
    (lambda raw: raw["split"].update(test_size=1.5), "between 0 and 1"),
    (lambda raw: raw["finetune"].update(warmup_ratio=1.0), "warmup_ratio"),
])
def test_invalid_config_fails_loudly(tmp_path, mutate, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, mutate))
