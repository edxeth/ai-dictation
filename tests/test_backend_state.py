from __future__ import annotations

import json
from pathlib import Path

from local_ai_dictation.backend_state import get_backend, set_backend, state_path, toggle_backend
import local_ai_dictation.cli as cli
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


def test_find_bridge_pids_matches_backend_argument_before_endpoint(monkeypatch):
    commands = {
        "local-ai-dictation bridge .*--host 127.0.0.1 --port 40125": "",
        "-m local_ai_dictation.cli bridge .*--host 127.0.0.1 --port 40125": "4321\n",
    }

    class _Completed:
        returncode = 0

        def __init__(self, stdout: str):
            self.stdout = stdout

    def _fake_run(command, **kwargs):
        assert command[:3] == ["pgrep", "-f", "--"]
        return _Completed(commands[command[3]])

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(cli.os, "getpid", lambda: 9999)

    assert cli._find_bridge_pids("127.0.0.1", 40125) == [4321]


def test_restart_bridge_uses_requested_custom_endpoint(monkeypatch, tmp_path: Path):
    launcher = tmp_path / "home" / ".local" / "bin" / "local-ai-dictation-bridge"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(cli, "_find_bridge_pids", lambda host, port: [])
    monkeypatch.setattr(cli, "_wait_for_bridge_ready", lambda host, port: True)

    popen_calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda command, **kwargs: popen_calls.append(command))

    cli._restart_local_bridge("127.0.0.1", 40125)

    assert popen_calls == [[
        cli.sys.executable,
        "-m",
        "local_ai_dictation.cli",
        "bridge",
        "--host",
        "127.0.0.1",
        "--port",
        "40125",
    ]]


def test_bridge_start_command_uses_persisted_backend_when_not_overridden(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    set_backend("whisper")

    assert bridge_start_command("127.0.0.1", 8765) == "local-ai-dictation bridge --host 127.0.0.1 --port 8765 --backend whisper"
