from __future__ import annotations

import json
from pathlib import Path

from local_ai_dictation.backend_state import get_backend, set_backend, state_path, toggle_backend
from local_ai_dictation.cli import main
from local_ai_dictation.desktop import bridge_start_command


def test_backend_state_defaults_to_whisper_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    assert get_backend() == "whisper"


def test_backend_state_set_and_toggle_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert set_backend("whisper") == "whisper"
    assert get_backend() == "whisper"
    payload = json.loads(state_path().read_text(encoding="utf-8"))
    assert payload == {"backend": "whisper"}

    assert toggle_backend() == "parakeet"
    assert get_backend() == "parakeet"
    assert toggle_backend() == "willow"
    assert get_backend() == "willow"
    assert toggle_backend() == "whisper"


def test_backend_cli_toggle_and_get_json(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert main(["backend", "toggle", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "parakeet"
    assert payload["state_path"].endswith("backend.json")

    assert main(["backend", "get"]) == 0
    assert capsys.readouterr().out.strip() == "parakeet"


def test_backend_cli_accepts_willow(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert main(["backend", "set", "willow"]) == 0
    assert capsys.readouterr().out.strip() == "willow"
    assert get_backend() == "willow"


def test_bridge_start_command_uses_persisted_backend_when_not_overridden(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    set_backend("whisper")

    assert bridge_start_command("127.0.0.1", 8765) == "local-ai-dictation bridge --host 127.0.0.1 --port 8765 --backend whisper"
