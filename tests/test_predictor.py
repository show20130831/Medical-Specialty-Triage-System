import contextlib
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triage_system.model_assets import ModelAssets
from triage_system.predictor import PredictionError, predict_description, rank_specialties

LABELS = ("A", "B", "C", "D")


def test_ranking_returns_softmax_top_three():
    result = rank_specialties([0.0, 2.0, 1.0, -1.0], LABELS)
    total = sum(math.exp(v) for v in (0.0, 2.0, 1.0, -1.0))
    assert result.label == "B"
    assert result.score == pytest.approx(math.exp(2.0) / total)
    assert [label for label, _ in result.top] == ["B", "C", "A"]
    assert not result.uncertain


def test_temperature_changes_confidence_but_not_ranking():
    sharp = rank_specialties([0.0, 2.0, 1.0, -1.0], LABELS, temperature=0.5)
    flat = rank_specialties([0.0, 2.0, 1.0, -1.0], LABELS, temperature=2.0)
    assert sharp.label == flat.label == "B"
    assert sharp.score > flat.score


def test_low_confidence_is_marked_uncertain():
    result = rank_specialties([0.1, 0.0, 0.0, 0.0], LABELS, uncertain_threshold=0.65)
    assert result.uncertain
    assert result.label == "A"


def test_logit_count_must_match_labels():
    with pytest.raises(PredictionError, match="mapping"):
        rank_specialties([0.0, 1.0], LABELS)


def test_prediction_uses_model_logits_and_calibration(monkeypatch, tmp_path):
    tokenizer = Mock(return_value={"input_ids": Mock(to=lambda _: "ids"),
                                   "attention_mask": Mock(to=lambda _: "mask")})
    row = Mock()
    row.float.return_value.tolist.return_value = [0.0, 3.0, 1.0, 0.0]
    model = Mock()
    model.return_value.logits = [row]
    assets = ModelAssets(tmp_path, LABELS, temperature=1.0, uncertain_threshold=0.5)
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(inference_mode=contextlib.nullcontext))
    result = predict_description(
        SimpleNamespace(tokenizer=tokenizer, model=model, assets=assets), "chest pain")
    assert result.label == "B"
    assert not result.uncertain
    model.assert_called_once_with(input_ids="ids", attention_mask="mask")


def test_blank_description_is_rejected():
    with pytest.raises(PredictionError, match="non-blank"):
        predict_description(SimpleNamespace(), "   ")
