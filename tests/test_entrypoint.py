import sys
from types import SimpleNamespace

import pytest

from triage_system import __main__ as entrypoint


def test_entrypoint_uses_validated_host_and_port(monkeypatch):
    calls = []
    monkeypatch.setenv("TRIAGE_HOST", "0.0.0.0")
    monkeypatch.setenv("TRIAGE_PORT", "9123")
    monkeypatch.setitem(sys.modules, "uvicorn",
                        SimpleNamespace(run=lambda app, **kwargs: calls.append((app, kwargs))))
    entrypoint.main()
    assert calls == [("triage_system.api:app", {"host": "0.0.0.0", "port": 9123})]


def test_entrypoint_rejects_invalid_port(monkeypatch):
    monkeypatch.setenv("TRIAGE_PORT", "not-a-port")
    with pytest.raises(SystemExit, match="TRIAGE_PORT"):
        entrypoint.main()
