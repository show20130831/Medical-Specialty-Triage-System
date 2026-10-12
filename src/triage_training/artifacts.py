"""Run bookkeeping: JSON output, file hashes, environment, and code provenance."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from triage_training.config import REPO_ROOT

TRACKED_PACKAGES = (
    "torch", "transformers", "accelerate", "tokenizers", "safetensors",
    "numpy", "pandas", "scikit-learn",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_state() -> dict[str, object]:
    """Record which commit produced a run, and whether uncommitted edits were present."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment() -> dict[str, object]:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info: dict[str, object] = {"python": platform.python_version(),
                               "platform": platform.platform(), "packages": packages}
    try:
        import torch
    except ImportError:
        return info
    info["cuda"] = torch.version.cuda
    info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return info
