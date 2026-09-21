import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triage_system import model_loader
from triage_system.model_assets import SPECIALTIES, ModelAssets


@pytest.fixture
def backend(tmp_path, monkeypatch):
    assets = ModelAssets(tmp_path, tuple(sorted(SPECIALTIES)))
    model = Mock()
    model.config = SimpleNamespace(
        id2label=dict(enumerate(assets.labels)),
        label2id={label: i for i, label in enumerate(assets.labels)},
    )
    model.training = True
    model.to.return_value = model
    model.eval.side_effect = lambda: setattr(model, "training", False)
    model.parameters.return_value = [SimpleNamespace(device=SimpleNamespace(type="cpu"))]
    tokenizer_class, model_class = Mock(), Mock()
    model_class.from_pretrained.return_value = model
    monkeypatch.setattr(model_loader, "validate_model_assets", Mock(return_value=assets))
    monkeypatch.setattr(model_loader, "_get_auto_classes", lambda: (tokenizer_class, model_class))
    return assets, tokenizer_class, model_class, model


def test_offline_cpu_loading_contract(backend):
    assets, tokenizer_class, model_class, model = backend
    result = model_loader.load_local_model(assets.directory)
    tokenizer_class.from_pretrained.assert_called_once_with(
        str(assets.directory), local_files_only=True, trust_remote_code=False,
        model_max_length=512,
    )
    model_class.from_pretrained.assert_called_once_with(
        str(assets.directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True,
    )
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once_with()
    assert result.assets.max_length == 512
    assert result.model is model


def test_loading_failure_is_actionable(backend):
    assets, _, model_class, _ = backend
    model_class.from_pretrained.side_effect = OSError("internal synthetic detail")
    with pytest.raises(model_loader.ModelLoadError, match="asset integrity") as exc:
        model_loader.load_local_model(assets.directory)
    assert "internal synthetic detail" not in str(exc.value)


def test_loaded_mapping_must_match(backend):
    assets, _, _, model = backend
    model.config.id2label = {0: "Unexpected"}
    with pytest.raises(model_loader.ModelLoadError, match="mappings"):
        model_loader.load_local_model(assets.directory)


def test_gpu_model_is_rejected(backend):
    assets, _, _, model = backend
    model.parameters.return_value = [SimpleNamespace(device=SimpleNamespace(type="cuda"))]
    with pytest.raises(model_loader.ModelLoadError, match="CPU"):
        model_loader.load_local_model(assets.directory)


def test_missing_assets_fail_before_ml_import(tmp_path, monkeypatch):
    imports = Mock(side_effect=AssertionError("Must validate first"))
    monkeypatch.setattr(model_loader, "_get_auto_classes", imports)
    with pytest.raises(ValueError, match="Required asset"):
        model_loader.load_local_model(tmp_path)
    imports.assert_not_called()


def test_api_and_loader_import_without_ml_packages():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers'}:
        raise ImportError('ML dependency intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from triage_system.api import health
from triage_system.model_loader import _get_auto_classes, ModelLoadError
assert health() == {'status': 'ok'}
try:
    _get_auto_classes()
except ModelLoadError as exc:
    assert 'inference extra' in str(exc)
else:
    raise AssertionError('Expected actionable dependency error')
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True, timeout=30)
