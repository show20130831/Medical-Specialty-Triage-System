"""Environment-backed application settings."""

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """An environment setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    model_dir: Path | None
    host: str = "127.0.0.1"
    port: int = 8000
    preload: bool = False
    monitoring_config: Path | None = None


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    """Read settings without logging secrets or medical input."""
    values = os.environ if environ is None else environ
    model_value = values.get("TRIAGE_MODEL_DIR", "").strip()
    model_dir = Path(model_value).expanduser() if model_value else None
    host = values.get("TRIAGE_HOST", "127.0.0.1").strip()
    if not host:
        raise ConfigurationError("TRIAGE_HOST must not be blank.")
    raw_port = values.get("TRIAGE_PORT", "8000").strip()
    try:
        port = int(raw_port)
    except ValueError:
        raise ConfigurationError("TRIAGE_PORT must be an integer from 1 to 65535.") from None
    if not 1 <= port <= 65535:
        raise ConfigurationError("TRIAGE_PORT must be an integer from 1 to 65535.")
    preload_value = values.get("TRIAGE_PRELOAD", "false").strip().lower()
    if preload_value not in ("true", "false", "1", "0"):
        raise ConfigurationError("TRIAGE_PRELOAD must be true or false.")
    monitoring_value = values.get("TRIAGE_MONITORING_CONFIG", "").strip()
    return Settings(model_dir=model_dir, host=host, port=port,
                    preload=preload_value in ("true", "1"),
                    monitoring_config=Path(monitoring_value).expanduser()
                    if monitoring_value else None)
