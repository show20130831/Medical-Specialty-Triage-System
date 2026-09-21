from pathlib import Path

import pytest

from triage_system.config import ConfigurationError, load_settings


def test_settings_use_safe_defaults():
    settings = load_settings({})
    assert settings.model_dir is None
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_settings_parse_environment_values():
    settings = load_settings({"TRIAGE_MODEL_DIR": "~/model", "TRIAGE_HOST": "0.0.0.0", "TRIAGE_PORT": "9000"})
    assert settings.model_dir == Path("~/model").expanduser()
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000


@pytest.mark.parametrize("value", ["0", "65536", "not-a-port"])
def test_invalid_port_is_rejected(value):
    with pytest.raises(ConfigurationError, match="TRIAGE_PORT"):
        load_settings({"TRIAGE_PORT": value})
