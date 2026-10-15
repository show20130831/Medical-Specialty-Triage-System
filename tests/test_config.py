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


def test_preload_and_monitoring_config_are_parsed():
    settings = load_settings({"TRIAGE_PRELOAD": "true", "TRIAGE_MONITORING_CONFIG": "~/m.yaml"})
    assert settings.preload is True
    assert settings.monitoring_config == Path("~/m.yaml").expanduser()
    assert load_settings({}).preload is False


def test_invalid_preload_is_rejected():
    with pytest.raises(ConfigurationError, match="TRIAGE_PRELOAD"):
        load_settings({"TRIAGE_PRELOAD": "maybe"})
