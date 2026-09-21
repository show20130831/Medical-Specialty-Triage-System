from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triage_system.model_assets import ModelAssets
from triage_system.predictor import PredictionError, predict_description


def test_prediction_returns_highest_label_and_score(monkeypatch, tmp_path):
    tokenizer = Mock(return_value={"input_ids": Mock(to=lambda _: Mock()), "attention_mask": Mock(to=lambda _: Mock())})
    model = Mock()
    model.return_value.logits = Mock()
    model.config = SimpleNamespace()
    assets = ModelAssets(tmp_path, ("A", "B"))
    fake_torch = SimpleNamespace(
        inference_mode=lambda: __import__("contextlib").nullcontext(),
        softmax=lambda logits, dim: [[0.2, 0.8]],
        argmax=lambda values: SimpleNamespace(item=lambda: 1),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    result = predict_description(SimpleNamespace(tokenizer=tokenizer, model=model, assets=assets), "chest pain")
    assert result.label == "B"
    assert result.score == 0.8
    tokenizer.assert_called_once()


def test_blank_description_is_rejected():
    with pytest.raises(PredictionError, match="non-blank"):
        predict_description(SimpleNamespace(), "   ")
