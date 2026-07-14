from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from local_ai_dictation.cli import main
import local_ai_dictation.desktop as desktop


@pytest.fixture(autouse=True)
def _isolate_persisted_backend(monkeypatch):
    monkeypatch.setattr(desktop, "get_backend", lambda: "whisper")


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakePopen:
    def __init__(self, command, **kwargs):
        self.command = command
        self.cwd = kwargs.get("cwd")
        self.env = kwargs.get("env")
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def communicate(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return "", ""


class _PopenFactory:
    def __init__(self):
        self.calls: list[_FakePopen] = []

    def __call__(self, command, **kwargs):
        process = _FakePopen(command, **kwargs)
        self.calls.append(process)
        return process


def test_gui_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_gui(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_command", _fake_gui)

    assert main(["gui", "--host", "127.0.0.1", "--port", "8765", "--hotkey", "Super+R"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui"
    assert calls[0].host == "127.0.0.1"
    assert calls[0].port == 8765
    assert calls[0].hotkey == "Super+R"


def test_bridge_toggle_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_bridge_toggle(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_bridge_toggle_command", _fake_bridge_toggle)

    assert main(["bridge-toggle", "--host", "127.0.0.1", "--port", "8765", "--json"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "bridge-toggle"
    assert calls[0].host == "127.0.0.1"
    assert calls[0].port == 8765
    assert calls[0].json_output is True


def test_gui_stage_subcommand_dispatches_to_desktop_stager(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_stage(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_stage_command", _fake_stage)

    assert main(["gui-stage", "--json"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-stage"
    assert calls[0].json_output is True


def test_gui_package_subcommand_dispatches_to_desktop_packager(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_package(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_command", _fake_package)

    assert main(["gui-package", "--json"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package"
    assert calls[0].json_output is True


def test_gui_package_smoke_subcommand_dispatches_to_desktop_smoke_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_smoke(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_smoke_command", _fake_smoke)

    assert main(["gui-package-smoke", "--json", "--timeout-seconds", "12", "--auto-exit-ms", "900"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-smoke"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 12
    assert calls[0].auto_exit_ms == 900


def test_gui_package_automation_subcommand_dispatches_to_desktop_automation_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_automation(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_automation_command", _fake_automation)

    assert main(["gui-package-automation", "--json", "--timeout-seconds", "20", "--automation-port", "49001"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-automation"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 20
    assert calls[0].automation_port == 49001


def test_gui_package_bridge_recovery_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_bridge_recovery(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_bridge_recovery_command", _fake_bridge_recovery)

    assert main([
        "gui-package-bridge-recovery",
        "--json",
        "--timeout-seconds",
        "21",
        "--automation-port",
        "49002",
        "--host",
        "127.0.0.1",
        "--bridge-port",
        "40125",
    ]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-bridge-recovery"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 21
    assert calls[0].automation_port == 49002
    assert calls[0].host == "127.0.0.1"
    assert calls[0].bridge_port == 40125


def test_gui_package_main_window_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_main_window(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_main_window_command", _fake_main_window)

    assert main([
        "gui-package-main-window",
        "--json",
        "--timeout-seconds",
        "22",
        "--automation-port",
        "49003",
        "--host",
        "127.0.0.1",
        "--bridge-port",
        "40126",
    ]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-main-window"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 22
    assert calls[0].automation_port == 49003
    assert calls[0].host == "127.0.0.1"
    assert calls[0].bridge_port == 40126


def test_gui_package_tray_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_tray(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_tray_command", _fake_tray)

    assert main([
        "gui-package-tray",
        "--json",
        "--timeout-seconds",
        "23",
        "--automation-port",
        "49004",
        "--host",
        "127.0.0.1",
        "--bridge-port",
        "40127",
    ]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-tray"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 23
    assert calls[0].automation_port == 49004
    assert calls[0].host == "127.0.0.1"
    assert calls[0].bridge_port == 40127


def test_gui_package_hotkey_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_hotkey(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_hotkey_command", _fake_hotkey)

    assert main([
        "gui-package-hotkey",
        "--json",
        "--timeout-seconds",
        "24",
        "--automation-port",
        "49005",
        "--host",
        "127.0.0.1",
        "--bridge-port",
        "40128",
    ]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-hotkey"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 24
    assert calls[0].automation_port == 49005
    assert calls[0].host == "127.0.0.1"
    assert calls[0].bridge_port == 40128


def test_gui_package_verify_subcommand_dispatches_to_desktop_runner(monkeypatch):
    calls: list[SimpleNamespace] = []

    def _fake_verify(namespace):
        calls.append(namespace)
        return 0

    monkeypatch.setattr("local_ai_dictation.desktop.run_gui_package_verify_command", _fake_verify)

    assert main(["gui-package-verify", "--json", "--timeout-seconds", "25"]) == 0
    assert len(calls) == 1
    assert calls[0].command == "gui-package-verify"
    assert calls[0].json_output is True
    assert calls[0].timeout_seconds == 25


def test_build_gui_environment_sets_native_log_paths(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    env = desktop.build_gui_environment("127.0.0.1", 8765, hotkey="Super+R")

    assert env["LOCAL_AI_DICTATION_BRIDGE_URL"] == "http://127.0.0.1:8765"
    assert env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == "local-ai-dictation bridge --host 127.0.0.1 --port 8765 --backend whisper"
    assert env["LOCAL_AI_DICTATION_HOTKEY"] == "Super+R"
    assert env["LOCAL_AI_DICTATION_GUI_LOG_PATH"].endswith(".local/state/local-ai-dictation/desktop/startup.log")
    assert env["LOCAL_AI_DICTATION_GUI_STARTUP_DIAGNOSTICS_PATH"].endswith(".local/state/local-ai-dictation/desktop/startup-diagnostics.json")


def test_build_gui_environment_includes_whisper_backend_in_default_bridge_command(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    env = desktop.build_gui_environment("127.0.0.1", 8765, backend="whisper")

    assert env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == "local-ai-dictation bridge --host 127.0.0.1 --port 8765 --backend whisper"


def test_run_bridge_toggle_command_posts_to_bridge(monkeypatch, capsys):
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"state":"recording"}'

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(desktop, "urlopen", _fake_urlopen)

    namespace = SimpleNamespace(host="127.0.0.1", port=8765, json_output=False)
    assert desktop.run_bridge_toggle_command(namespace) == 0
    assert captured == {
        "url": "http://127.0.0.1:8765/session/toggle",
        "method": "POST",
        "body": b"{}",
        "timeout": 5.0,
    }
    assert capsys.readouterr().out.strip() == "recording"


def test_build_bridge_command_includes_whisper_backend():
    namespace = SimpleNamespace(host="127.0.0.1", port=8765, backend="whisper", max_silence_ms=1200, min_speech_ms=300, vad_mode=2, log_file="transcriber.debug.log", cpu=False, input_device=None, vad=False, clipboard=True, debug=False)

    command = desktop.build_bridge_command(namespace)

    assert command[:7] == [sys.executable, "-m", "local_ai_dictation.cli", "bridge", "--backend", "whisper", "--host"]


def test_run_gui_command_launches_native_electrobun_app(monkeypatch, tmp_path: Path):
    app_dir = tmp_path / "desktop" / "electrobun"
    app_dir.mkdir(parents=True)
    dependency_installs: list[tuple[Path, str]] = []
    runs: list[tuple[list[str], Path, dict[str, str] | None]] = []

    monkeypatch.setattr(desktop, "ensure_desktop_app_available", lambda: app_dir)
    monkeypatch.setattr(desktop, "ensure_bun_available", lambda: "bun")
    monkeypatch.setattr(desktop, "ensure_gui_dependencies", lambda path, bun_path: dependency_installs.append((path, bun_path)))

    def _fake_run(command, **kwargs):
        runs.append((command, kwargs.get("cwd"), kwargs.get("env")))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(desktop.subprocess, "run", _fake_run)

    namespace = SimpleNamespace(host="127.0.0.1", port=8765, hotkey="Super+R", bridge_command="uv run local-ai-dictation bridge --host 127.0.0.1 --port 8765")
    assert desktop.run_gui_command(namespace) == 0
    assert dependency_installs == [(app_dir, "bun")]
    assert len(runs) == 6
    command, cwd, env = runs[-1]
    assert command == ["bun", "run", "start"]
    assert cwd == app_dir
    assert env is not None
    assert env["LOCAL_AI_DICTATION_BRIDGE_URL"] == "http://127.0.0.1:8765"
    assert env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == "uv run local-ai-dictation bridge --host 127.0.0.1 --port 8765"
    assert env["LOCAL_AI_DICTATION_HOTKEY"] == "Super+R"


def test_stage_windows_desktop_app_copies_desktop_folder(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    app_dir = repo_root / "desktop" / "electrobun"
    (app_dir / "src").mkdir(parents=True)
    (app_dir / "package.json").write_text('{"name":"local-ai-dictation-electrobun"}')
    (app_dir / "src" / "index.ts").write_text("console.log('hello');")
    (app_dir / "node_modules").mkdir()
    (app_dir / "node_modules" / "ignored.txt").write_text("ignore me")
    (app_dir / ".tmp-check").mkdir()
    (app_dir / ".tmp-check" / "ignored.txt").write_text("ignore me")

    stage_root = tmp_path / "mnt" / "c" / "Users" / "dev" / "AppData" / "Local" / "LocalAIDictation" / "staging"
    monkeypatch.setattr(desktop, "windows_stage_root", lambda: stage_root)
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")

    payload = desktop.stage_windows_desktop_app(app_dir)
    staged_app_dir = Path(payload["desktop_app_dir"])

    assert staged_app_dir == next(stage_root.glob("*/desktop/electrobun"))
    assert (staged_app_dir / "package.json").exists()
    assert (staged_app_dir / "src" / "index.ts").read_text() == "console.log('hello');"
    assert not (staged_app_dir / "node_modules").exists()
    assert not (staged_app_dir / ".tmp-check").exists()
    assert payload["windows_desktop_app_dir"].startswith("C:\\stage\\")


def test_run_gui_package_command_stages_build_and_reports_artifacts(monkeypatch, tmp_path: Path):
    app_dir = tmp_path / "desktop" / "electrobun"
    app_dir.mkdir(parents=True)
    build_dir = app_dir / "build" / "stable-win-x64"
    artifacts_dir = app_dir / "artifacts"

    monkeypatch.setattr(desktop, "ensure_windows_bun_available", lambda: "C:\\Users\\dev\\scoop\\shims\\bun.exe")
    monkeypatch.setattr(
        desktop,
        "stage_windows_desktop_app",
        lambda: {
            "desktop_app_dir": str(app_dir),
            "windows_desktop_app_dir": "C:\\stage\\electrobun",
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")

    commands: list[tuple[str, list[str]]] = []

    def _fake_run_windows_in_dir(windows_dir: str, command_list: list[str]) -> None:
        commands.append((windows_dir, command_list))
        build_dir.mkdir(parents=True)
        (build_dir / "local-ai-dictation-desktop" / "bin").mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)
        (build_dir / "local-ai-dictation-desktop" / "bin" / "launcher").write_text("launcher")
        (build_dir / "local-ai-dictation-desktop-Setup.exe").write_text("setup")
        (build_dir / "local-ai-dictation-desktop-Setup.tar.zst").write_text("archive")
        (build_dir / "local-ai-dictation-desktop-Setup.metadata.json").write_text('{"identifier":"local-ai-dictation.desktop.local","channel":"stable","name":"local-ai-dictation-desktop"}')
        (artifacts_dir / "stable-win-x64-local-ai-dictation-desktop-Setup.zip").write_text("zip")

    monkeypatch.setattr(desktop, "run_windows_in_dir", _fake_run_windows_in_dir)

    namespace = SimpleNamespace(json_output=False)
    assert desktop.run_gui_package_command(namespace) == 0
    assert commands == [(
        "C:\\stage\\electrobun",
        ["bun install", "bunx electrobun build --env=stable"],
    )]


def test_apply_windows_packaged_icon_workaround_updates_all_windows_exes(monkeypatch, tmp_path: Path):
    app_dir = tmp_path / "desktop" / "electrobun"
    icon_path = app_dir / "src" / "mainview" / "assets" / "local-ai-dictation-icon.ico"
    rcedit_path = app_dir / "node_modules" / "rcedit" / "bin" / "rcedit-x64.exe"
    build_dir = app_dir / "build" / "stable-win-x64"
    bin_dir = build_dir / "local-ai-dictation-desktop" / "bin"
    bin_dir.mkdir(parents=True)
    icon_path.parent.mkdir(parents=True)
    rcedit_path.parent.mkdir(parents=True)
    icon_path.write_text("icon")
    rcedit_path.write_text("rcedit")

    launcher_path = bin_dir / "launcher.exe"
    bun_path = bin_dir / "bun.exe"
    helper_path = bin_dir / "process_helper.exe"
    setup_path = build_dir / "local-ai-dictation-desktop-Setup.exe"
    for target in (launcher_path, bun_path, helper_path, setup_path):
        target.write_text(target.name)

    embedded: list[Path] = []

    def _fake_embed(executable_path: Path, icon_file: Path, rcedit_file: Path) -> None:
        assert icon_file == icon_path
        assert rcedit_file == rcedit_path
        embedded.append(executable_path)

    monkeypatch.setattr(desktop, "embed_windows_exe_icon", _fake_embed)

    desktop.apply_windows_packaged_icon_workaround(app_dir)

    assert embedded == sorted([launcher_path, bun_path, helper_path, setup_path])


def test_run_gui_package_bridge_recovery_command_verifies_offline_then_online(monkeypatch, tmp_path: Path):
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir(parents=True)
    installed_root = tmp_path / "installed" / "stable"
    launcher_path = installed_root / "app" / "bin" / "launcher.exe"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("launcher")
    diagnostics_path = installed_root / "logs" / "startup-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    log_path = installed_root / "logs" / "startup.log"

    monkeypatch.setattr(
        desktop,
        "build_windows_package_payload",
        lambda: {
            "stage_root": str(stage_root),
            "installer_path": str(stage_root / "local-ai-dictation-desktop-Setup.exe"),
            "app_identifier": "local-ai-dictation.desktop.local",
            "app_channel": "stable",
        },
    )
    monkeypatch.setattr(
        desktop,
        "installed_windows_app_paths",
        lambda identifier, channel: {
            "app_root": installed_root,
            "app_dir": installed_root / "app",
            "launcher_path": launcher_path,
            "logs_dir": installed_root / "logs",
            "diagnostics_path": diagnostics_path,
            "log_path": log_path,
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")
    monkeypatch.setattr(desktop, "prepare_installed_windows_app_for_reinstall", lambda installed_paths: None)
    monkeypatch.setattr(desktop, "reserve_localhost_port", lambda: 49012)
    monkeypatch.setattr(desktop, "reserve_socket_localhost_port", lambda: 40125)

    installer_calls: list[tuple[Path, float]] = []

    def _fake_run_wsl_windows_executable_capture(executable_path: Path, *, timeout_seconds: float, env=None, cwd=None):
        installer_calls.append((executable_path, timeout_seconds))
        return _FakeCompletedProcess(returncode=0, stdout="installer ok", stderr="")

    monkeypatch.setattr(desktop, "run_wsl_windows_executable_capture", _fake_run_wsl_windows_executable_capture)

    launcher_factory = _PopenFactory()

    def _fake_launch_wsl_windows_executable(executable_path: Path, *, env=None, cwd=None):
        diagnostics_path.write_text(
            json.dumps(
                {
                    "bunReady": True,
                    "rendererReady": True,
                    "rendererRpcReady": True,
                    "automationReady": True,
                    "shutdownReason": "automation:quit",
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("automation ready", encoding="utf-8")
        process = launcher_factory([str(executable_path)], env=env, cwd=cwd)
        process.returncode = 0
        return process

    monkeypatch.setattr(desktop, "launch_wsl_windows_executable", _fake_launch_wsl_windows_executable)
    monkeypatch.setattr(desktop, "wait_for_bridge", lambda host, port, timeout_seconds=10.0, poll_interval=0.25: True)

    state_requests: list[str] = []
    expected_bridge_url = "http://127.0.0.1:40125"
    expected_bridge_command = "local-ai-dictation bridge --host 127.0.0.1 --port 40125 --backend whisper"
    offline_state = {
        "bridge": {"connected": False},
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Bridge offline",
                "bridgeUrl": expected_bridge_url,
                "bridgeCommand": expected_bridge_command,
                "statusLine": "Start the bridge command below, then use the button or hotkey.",
            }
        },
    }
    recovered_state = {
        "bridge": {"connected": True},
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Start recording",
                "bridgeUrl": expected_bridge_url,
                "bridgeCommand": expected_bridge_command,
                "statusLine": "Model ready. Press the button or the hotkey to begin.",
            }
        },
    }

    def _fake_wait_for_gui_e2e_state(port: int, predicate, *, timeout_seconds: float, poll_interval: float = 0.25):
        state = offline_state if not state_requests else recovered_state
        state_requests.append(f"state:{port}:{timeout_seconds}")
        assert predicate(state) is True
        return state

    monkeypatch.setattr(desktop, "wait_for_gui_e2e_state", _fake_wait_for_gui_e2e_state)

    action_requests: list[str] = []

    def _fake_invoke_gui_e2e_action(port: int, action: str, *, timeout_seconds: float = desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS):
        action_requests.append(f"{port}:{action}:{timeout_seconds}")
        return {"diagnostics": {"lastAutomationAction": action}}

    monkeypatch.setattr(desktop, "invoke_gui_e2e_action", _fake_invoke_gui_e2e_action)

    bridge_factory = _PopenFactory()
    monkeypatch.setattr(desktop.subprocess, "Popen", bridge_factory)

    namespace = SimpleNamespace(json_output=False, timeout_seconds=12.0, automation_port=0, host="127.0.0.1", bridge_port=0)
    assert desktop.run_gui_package_bridge_recovery_command(namespace) == 0
    assert installer_calls == [(
        stage_root / "local-ai-dictation-desktop-Setup.exe",
        12.0,
    )]
    assert len(launcher_factory.calls) == 1
    assert launcher_factory.calls[0].env is not None
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E_PORT"] == "49012"
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_URL"] == expected_bridge_url
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == expected_bridge_command
    assert len(bridge_factory.calls) == 1
    assert bridge_factory.calls[0].command[:4] == [desktop.sys.executable, "-m", "local_ai_dictation.cli", "bridge"]
    assert bridge_factory.calls[0].env is not None
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_MODE"] == "1"
    assert state_requests == ["state:49012:12.0", "state:49012:12.0"]
    assert action_requests == [f"49012:quit:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}"]
    assert (stage_root / "smoke" / "bridge.stdout.log").exists()
    assert (stage_root / "smoke" / "startup-diagnostics.json").exists()


def test_run_gui_package_main_window_command_verifies_renderer_toggle_flow(monkeypatch, tmp_path: Path):
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir(parents=True)
    installed_root = tmp_path / "installed" / "stable"
    launcher_path = installed_root / "app" / "bin" / "launcher.exe"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("launcher")
    diagnostics_path = installed_root / "logs" / "startup-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    log_path = installed_root / "logs" / "startup.log"

    monkeypatch.setattr(
        desktop,
        "build_windows_package_payload",
        lambda: {
            "stage_root": str(stage_root),
            "installer_path": str(stage_root / "local-ai-dictation-desktop-Setup.exe"),
            "app_identifier": "local-ai-dictation.desktop.local",
            "app_channel": "stable",
        },
    )
    monkeypatch.setattr(
        desktop,
        "installed_windows_app_paths",
        lambda identifier, channel: {
            "app_root": installed_root,
            "app_dir": installed_root / "app",
            "launcher_path": launcher_path,
            "logs_dir": installed_root / "logs",
            "diagnostics_path": diagnostics_path,
            "log_path": log_path,
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")
    monkeypatch.setattr(desktop, "prepare_installed_windows_app_for_reinstall", lambda installed_paths: None)
    monkeypatch.setattr(desktop, "reserve_localhost_port", lambda: 49013)
    monkeypatch.setattr(desktop, "reserve_socket_localhost_port", lambda: 40126)
    monkeypatch.setattr(desktop, "wait_for_bridge", lambda host, port, timeout_seconds=10.0, poll_interval=0.25: True)

    installer_calls: list[tuple[Path, float]] = []

    def _fake_run_wsl_windows_executable_capture(executable_path: Path, *, timeout_seconds: float, env=None, cwd=None):
        installer_calls.append((executable_path, timeout_seconds))
        return _FakeCompletedProcess(returncode=0, stdout="installer ok", stderr="")

    monkeypatch.setattr(desktop, "run_wsl_windows_executable_capture", _fake_run_wsl_windows_executable_capture)

    launcher_factory = _PopenFactory()

    def _fake_launch_wsl_windows_executable(executable_path: Path, *, env=None, cwd=None):
        diagnostics_path.write_text(
            json.dumps(
                {
                    "bunReady": True,
                    "rendererReady": True,
                    "rendererRpcReady": True,
                    "automationReady": True,
                    "shutdownReason": "automation:quit",
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("automation ready", encoding="utf-8")
        process = launcher_factory([str(executable_path)], env=env, cwd=cwd)
        process.returncode = 0
        return process

    monkeypatch.setattr(desktop, "launch_wsl_windows_executable", _fake_launch_wsl_windows_executable)

    state_requests: list[str] = []
    expected_bridge_url = "http://127.0.0.1:40126"
    expected_bridge_command = "local-ai-dictation bridge --host 127.0.0.1 --port 40126 --backend whisper"
    expected_transcript = "main window deterministic transcript"
    initial_state = {
        "bridge": {
            "connected": True,
            "session": {
                "state": "idle",
            },
        },
        "renderer": {
            "snapshot": {
                "bridgeUrl": expected_bridge_url,
                "bridgeCommand": expected_bridge_command,
                "toggleButtonText": "Start recording",
                "historyCount": 0,
            }
        },
    }
    recording_state = {
        "bridge": {
            "connected": True,
            "session": {
                "state": "recording",
            },
        },
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Stop recording",
                "historyCount": 0,
            }
        },
    }
    completed_state = {
        "bridge": {
            "connected": True,
            "session": {
                "state": "idle",
                "last_transcript": {"transcript": expected_transcript},
            },
        },
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Start recording",
                "historyCount": 1,
                "historyTexts": [expected_transcript],
            }
        },
    }

    def _fake_wait_for_gui_e2e_state(port: int, predicate, *, timeout_seconds: float, poll_interval: float = 0.25):
        state = [initial_state, recording_state, completed_state][len(state_requests)]
        state_requests.append(f"state:{port}:{timeout_seconds}")
        assert predicate(state) is True
        return state

    monkeypatch.setattr(desktop, "wait_for_gui_e2e_state", _fake_wait_for_gui_e2e_state)

    action_requests: list[str] = []

    def _fake_invoke_gui_e2e_action(port: int, action: str, *, timeout_seconds: float = desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS):
        action_requests.append(f"{port}:{action}:{timeout_seconds}")
        return {"diagnostics": {"lastAutomationAction": action}}

    monkeypatch.setattr(desktop, "invoke_gui_e2e_action", _fake_invoke_gui_e2e_action)

    bridge_factory = _PopenFactory()
    monkeypatch.setattr(desktop.subprocess, "Popen", bridge_factory)

    namespace = SimpleNamespace(json_output=False, timeout_seconds=12.0, automation_port=0, host="127.0.0.1", bridge_port=0)
    assert desktop.run_gui_package_main_window_command(namespace) == 0
    assert installer_calls == [(
        stage_root / "local-ai-dictation-desktop-Setup.exe",
        12.0,
    )]
    assert len(launcher_factory.calls) == 1
    assert launcher_factory.calls[0].env is not None
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E_PORT"] == "49013"
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_URL"] == expected_bridge_url
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == expected_bridge_command
    assert len(bridge_factory.calls) == 1
    assert bridge_factory.calls[0].env is not None
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_MODE"] == "1"
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_TRANSCRIPT"] == expected_transcript
    assert state_requests == ["state:49013:12.0", "state:49013:12.0", "state:49013:12.0"]
    assert action_requests == [
        f"49013:show-window:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49013:window/toggle-recording:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49013:window/toggle-recording:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49013:quit:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
    ]
    assert (stage_root / "smoke" / "bridge.stdout.log").exists()
    assert (stage_root / "smoke" / "startup-diagnostics.json").exists()


def test_run_gui_package_tray_command_verifies_tray_toggle_flow(monkeypatch, tmp_path: Path):
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir(parents=True)
    installed_root = tmp_path / "installed" / "stable"
    launcher_path = installed_root / "app" / "bin" / "launcher.exe"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("launcher")
    diagnostics_path = installed_root / "logs" / "startup-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    log_path = installed_root / "logs" / "startup.log"

    monkeypatch.setattr(
        desktop,
        "build_windows_package_payload",
        lambda: {
            "stage_root": str(stage_root),
            "installer_path": str(stage_root / "local-ai-dictation-desktop-Setup.exe"),
            "app_identifier": "local-ai-dictation.desktop.local",
            "app_channel": "stable",
        },
    )
    monkeypatch.setattr(
        desktop,
        "installed_windows_app_paths",
        lambda identifier, channel: {
            "app_root": installed_root,
            "app_dir": installed_root / "app",
            "launcher_path": launcher_path,
            "logs_dir": installed_root / "logs",
            "diagnostics_path": diagnostics_path,
            "log_path": log_path,
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")
    monkeypatch.setattr(desktop, "prepare_installed_windows_app_for_reinstall", lambda installed_paths: None)
    monkeypatch.setattr(desktop, "reserve_localhost_port", lambda: 49014)
    monkeypatch.setattr(desktop, "reserve_socket_localhost_port", lambda: 40127)
    monkeypatch.setattr(desktop, "wait_for_bridge", lambda host, port, timeout_seconds=10.0, poll_interval=0.25: True)

    installer_calls: list[tuple[Path, float]] = []

    def _fake_run_wsl_windows_executable_capture(executable_path: Path, *, timeout_seconds: float, env=None, cwd=None):
        installer_calls.append((executable_path, timeout_seconds))
        return _FakeCompletedProcess(returncode=0, stdout="installer ok", stderr="")

    monkeypatch.setattr(desktop, "run_wsl_windows_executable_capture", _fake_run_wsl_windows_executable_capture)

    launcher_factory = _PopenFactory()

    def _fake_launch_wsl_windows_executable(executable_path: Path, *, env=None, cwd=None):
        diagnostics_path.write_text(
            json.dumps(
                {
                    "bunReady": True,
                    "rendererReady": True,
                    "rendererRpcReady": True,
                    "automationReady": True,
                    "shutdownReason": "automation:tray/quit",
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("automation ready", encoding="utf-8")
        process = launcher_factory([str(executable_path)], env=env, cwd=cwd)
        process.returncode = 0
        return process

    monkeypatch.setattr(desktop, "launch_wsl_windows_executable", _fake_launch_wsl_windows_executable)

    state_requests: list[str] = []
    expected_bridge_url = "http://127.0.0.1:40127"
    expected_bridge_command = "local-ai-dictation bridge --host 127.0.0.1 --port 40127 --backend whisper"
    expected_transcript = "tray deterministic transcript"
    initial_state = {
        "tray": {"created": True, "actions": ["open", "toggle", "quit"]},
        "bridge": {
            "connected": True,
            "session": {
                "state": "idle",
            },
        },
        "renderer": {
            "snapshot": {
                "bridgeUrl": expected_bridge_url,
                "bridgeCommand": expected_bridge_command,
                "toggleButtonText": "Start recording",
                "historyCount": 0,
            }
        },
    }
    recording_state = {
        "tray": {"created": True, "actions": ["open", "toggle", "quit"]},
        "bridge": {
            "connected": True,
            "session": {
                "state": "recording",
            },
        },
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Stop recording",
                "historyCount": 0,
            }
        },
    }
    completed_state = {
        "tray": {"created": True, "actions": ["open", "toggle", "quit"]},
        "bridge": {
            "connected": True,
            "session": {
                "state": "idle",
                "last_transcript": {"transcript": expected_transcript},
            },
        },
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Start recording",
                "historyCount": 1,
                "historyTexts": [expected_transcript],
            }
        },
    }

    def _fake_wait_for_gui_e2e_state(port: int, predicate, *, timeout_seconds: float, poll_interval: float = 0.25):
        state = [initial_state, recording_state, completed_state][len(state_requests)]
        state_requests.append(f"state:{port}:{timeout_seconds}")
        assert predicate(state) is True
        return state

    monkeypatch.setattr(desktop, "wait_for_gui_e2e_state", _fake_wait_for_gui_e2e_state)

    action_requests: list[str] = []

    def _fake_invoke_gui_e2e_action(port: int, action: str, *, timeout_seconds: float = desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS):
        action_requests.append(f"{port}:{action}:{timeout_seconds}")
        return {"diagnostics": {"lastAutomationAction": action}}

    monkeypatch.setattr(desktop, "invoke_gui_e2e_action", _fake_invoke_gui_e2e_action)

    bridge_factory = _PopenFactory()
    monkeypatch.setattr(desktop.subprocess, "Popen", bridge_factory)

    namespace = SimpleNamespace(json_output=False, timeout_seconds=12.0, automation_port=0, host="127.0.0.1", bridge_port=0)
    assert desktop.run_gui_package_tray_command(namespace) == 0
    assert installer_calls == [(
        stage_root / "local-ai-dictation-desktop-Setup.exe",
        12.0,
    )]
    assert len(launcher_factory.calls) == 1
    assert launcher_factory.calls[0].env is not None
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E_PORT"] == "49014"
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_URL"] == expected_bridge_url
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == expected_bridge_command
    assert len(bridge_factory.calls) == 1
    assert bridge_factory.calls[0].env is not None
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_MODE"] == "1"
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_TRANSCRIPT"] == expected_transcript
    assert state_requests == ["state:49014:12.0", "state:49014:12.0", "state:49014:12.0"]
    assert action_requests == [
        f"49014:tray/open:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49014:tray/toggle:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49014:tray/toggle:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49014:tray/quit:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
    ]
    assert (stage_root / "smoke" / "bridge.stdout.log").exists()
    assert (stage_root / "smoke" / "startup-diagnostics.json").exists()


def test_run_gui_package_hotkey_command_verifies_hotkey_toggle_flow(monkeypatch, tmp_path: Path):
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir(parents=True)
    installed_root = tmp_path / "installed" / "stable"
    launcher_path = installed_root / "app" / "bin" / "launcher.exe"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("launcher")
    diagnostics_path = installed_root / "logs" / "startup-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    log_path = installed_root / "logs" / "startup.log"

    monkeypatch.setattr(
        desktop,
        "build_windows_package_payload",
        lambda: {
            "stage_root": str(stage_root),
            "installer_path": str(stage_root / "local-ai-dictation-desktop-Setup.exe"),
            "app_identifier": "local-ai-dictation.desktop.local",
            "app_channel": "stable",
        },
    )
    monkeypatch.setattr(
        desktop,
        "installed_windows_app_paths",
        lambda identifier, channel: {
            "app_root": installed_root,
            "app_dir": installed_root / "app",
            "launcher_path": launcher_path,
            "logs_dir": installed_root / "logs",
            "diagnostics_path": diagnostics_path,
            "log_path": log_path,
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")
    monkeypatch.setattr(desktop, "prepare_installed_windows_app_for_reinstall", lambda installed_paths: None)
    monkeypatch.setattr(desktop, "reserve_localhost_port", lambda: 49015)
    monkeypatch.setattr(desktop, "reserve_socket_localhost_port", lambda: 40128)
    monkeypatch.setenv("LOCAL_AI_DICTATION_HOTKEY", "CommandOrControl+Alt+R")
    monkeypatch.setattr(desktop, "wait_for_bridge", lambda host, port, timeout_seconds=10.0, poll_interval=0.25: True)

    installer_calls: list[tuple[Path, float]] = []

    def _fake_run_wsl_windows_executable_capture(executable_path: Path, *, timeout_seconds: float, env=None, cwd=None):
        installer_calls.append((executable_path, timeout_seconds))
        return _FakeCompletedProcess(returncode=0, stdout="installer ok", stderr="")

    monkeypatch.setattr(desktop, "run_wsl_windows_executable_capture", _fake_run_wsl_windows_executable_capture)

    launcher_factory = _PopenFactory()

    def _fake_launch_wsl_windows_executable(executable_path: Path, *, env=None, cwd=None):
        diagnostics_path.write_text(
            json.dumps(
                {
                    "bunReady": True,
                    "rendererReady": True,
                    "rendererRpcReady": True,
                    "automationReady": True,
                    "hotkeyRegistered": True,
                    "shutdownReason": "automation:quit",
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("automation ready", encoding="utf-8")
        process = launcher_factory([str(executable_path)], env=env, cwd=cwd)
        process.returncode = 0
        return process

    monkeypatch.setattr(desktop, "launch_wsl_windows_executable", _fake_launch_wsl_windows_executable)

    state_requests: list[str] = []
    expected_bridge_url = "http://127.0.0.1:40128"
    expected_bridge_command = "local-ai-dictation bridge --host 127.0.0.1 --port 40128 --backend whisper"
    expected_hotkey = "CommandOrControl+Alt+R"
    expected_transcript = "hotkey deterministic transcript"
    initial_state = {
        "hotkey": {"accelerator": expected_hotkey, "registered": True},
        "bridge": {
            "connected": True,
            "session": {
                "state": "idle",
            },
        },
        "renderer": {
            "snapshot": {
                "bridgeUrl": expected_bridge_url,
                "bridgeCommand": expected_bridge_command,
                "toggleButtonText": "Start recording",
                "historyCount": 0,
            }
        },
    }
    recording_state = {
        "hotkey": {"accelerator": expected_hotkey, "registered": True},
        "bridge": {
            "connected": True,
            "session": {
                "state": "recording",
            },
        },
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Stop recording",
                "historyCount": 0,
            }
        },
    }
    completed_state = {
        "hotkey": {"accelerator": expected_hotkey, "registered": True},
        "bridge": {
            "connected": True,
            "session": {
                "state": "idle",
                "last_transcript": {"transcript": expected_transcript},
            },
        },
        "renderer": {
            "snapshot": {
                "toggleButtonText": "Start recording",
                "historyCount": 1,
                "historyTexts": [expected_transcript],
            }
        },
    }

    def _fake_wait_for_gui_e2e_state(port: int, predicate, *, timeout_seconds: float, poll_interval: float = 0.25):
        state = [initial_state, recording_state, completed_state][len(state_requests)]
        state_requests.append(f"state:{port}:{timeout_seconds}")
        assert predicate(state) is True
        return state

    monkeypatch.setattr(desktop, "wait_for_gui_e2e_state", _fake_wait_for_gui_e2e_state)

    action_requests: list[str] = []

    def _fake_invoke_gui_e2e_action(port: int, action: str, *, timeout_seconds: float = desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS):
        action_requests.append(f"{port}:{action}:{timeout_seconds}")
        return {"diagnostics": {"lastAutomationAction": action}}

    monkeypatch.setattr(desktop, "invoke_gui_e2e_action", _fake_invoke_gui_e2e_action)

    bridge_factory = _PopenFactory()
    monkeypatch.setattr(desktop.subprocess, "Popen", bridge_factory)

    namespace = SimpleNamespace(json_output=False, timeout_seconds=12.0, automation_port=0, host="127.0.0.1", bridge_port=0)
    assert desktop.run_gui_package_hotkey_command(namespace) == 0
    assert installer_calls == [(
        stage_root / "local-ai-dictation-desktop-Setup.exe",
        12.0,
    )]
    assert len(launcher_factory.calls) == 1
    assert launcher_factory.calls[0].env is not None
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E_PORT"] == "49015"
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_URL"] == expected_bridge_url
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] == expected_bridge_command
    assert launcher_factory.calls[0].env["LOCAL_AI_DICTATION_HOTKEY"] == expected_hotkey
    assert len(bridge_factory.calls) == 1
    assert bridge_factory.calls[0].env is not None
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_MODE"] == "1"
    assert bridge_factory.calls[0].env["LOCAL_AI_DICTATION_E2E_TRANSCRIPT"] == expected_transcript
    assert state_requests == ["state:49015:12.0", "state:49015:12.0", "state:49015:12.0"]
    assert action_requests == [
        f"49015:hotkey/trigger:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49015:hotkey/trigger:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49015:quit:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
    ]
    assert (stage_root / "smoke" / "bridge.stdout.log").exists()
    assert (stage_root / "smoke" / "startup-diagnostics.json").exists()


def test_run_gui_package_smoke_command_launches_packaged_app_and_validates_diagnostics(monkeypatch, tmp_path: Path):
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir(parents=True)
    installed_root = tmp_path / "installed" / "stable"
    launcher_path = installed_root / "app" / "bin" / "launcher.exe"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("launcher")
    diagnostics_path = installed_root / "logs" / "startup-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    log_path = installed_root / "logs" / "startup.log"

    monkeypatch.setattr(
        desktop,
        "build_windows_package_payload",
        lambda: {
            "stage_root": str(stage_root),
            "installer_path": str(stage_root / "local-ai-dictation-desktop-Setup.exe"),
            "app_identifier": "local-ai-dictation.desktop.local",
            "app_channel": "stable",
        },
    )
    monkeypatch.setattr(
        desktop,
        "installed_windows_app_paths",
        lambda identifier, channel: {
            "app_root": installed_root,
            "app_dir": installed_root / "app",
            "launcher_path": launcher_path,
            "logs_dir": installed_root / "logs",
            "diagnostics_path": diagnostics_path,
            "log_path": log_path,
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")
    monkeypatch.setattr(desktop, "prepare_installed_windows_app_for_reinstall", lambda installed_paths: None)

    installer_calls: list[tuple[Path, float]] = []

    def _fake_run_wsl_windows_executable_capture(executable_path: Path, *, timeout_seconds: float, env=None, cwd=None):
        installer_calls.append((executable_path, timeout_seconds))
        return _FakeCompletedProcess(returncode=0, stdout="installer ok", stderr="")

    monkeypatch.setattr(desktop, "run_wsl_windows_executable_capture", _fake_run_wsl_windows_executable_capture)

    popen_factory = _PopenFactory()

    def _fake_launch_wsl_windows_executable(executable_path: Path, *, env=None, cwd=None):
        diagnostics_path.write_text(
            json.dumps(
                {
                    "bunReady": True,
                    "rendererReady": True,
                    "rendererRpcReady": True,
                    "shutdownReason": "e2e-auto-exit",
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("renderer ready", encoding="utf-8")
        process = popen_factory([str(executable_path)], env=env, cwd=cwd)
        process.returncode = 0
        return process

    monkeypatch.setattr(desktop, "launch_wsl_windows_executable", _fake_launch_wsl_windows_executable)
    monkeypatch.setattr(desktop, "request_installed_gui_shutdown", lambda installed_paths: None)

    namespace = SimpleNamespace(json_output=False, timeout_seconds=12.0, auto_exit_ms=900)
    assert desktop.run_gui_package_smoke_command(namespace) == 0
    assert installer_calls == [(
        stage_root / "local-ai-dictation-desktop-Setup.exe",
        12.0,
    )]
    assert len(popen_factory.calls) == 1
    assert popen_factory.calls[0].command == [str(launcher_path)]
    assert popen_factory.calls[0].env is not None
    assert popen_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E"] == "1"
    assert popen_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_AUTO_EXIT_MS"] == "900"
    assert (stage_root / "smoke" / "installer.stdout.log").read_text(encoding="utf-8") == "installer ok"
    assert (stage_root / "smoke" / "startup-diagnostics.json").exists()


def test_run_gui_package_verify_command_runs_existing_checks(monkeypatch, capsys):
    commands: list[list[str]] = []
    payloads = {
        "gui-package-smoke": {"installer_path": "/tmp/setup.exe", "windows_installer_path": "C:\\setup.exe", "windows_diagnostics_path": "C:\\smoke.json"},
        "gui-package-automation": {"windows_diagnostics_path": "C:\\automation.json"},
        "gui-package-bridge-recovery": {"windows_diagnostics_path": "C:\\bridge-recovery.json"},
        "gui-package-main-window": {"windows_diagnostics_path": "C:\\main-window.json"},
        "gui-package-tray": {"windows_diagnostics_path": "C:\\tray.json"},
        "gui-package-hotkey": {"windows_diagnostics_path": "C:\\hotkey.json"},
    }

    monkeypatch.setattr(desktop, "repo_root", lambda: Path("/tmp/local-ai-dictation"))

    def _fake_run(command, **kwargs):
        commands.append(command)
        subcommand = command[3]
        assert kwargs == {
            "cwd": Path("/tmp/local-ai-dictation"),
            "capture_output": True,
            "text": True,
            "check": False,
        }
        return _FakeCompletedProcess(
            returncode=0,
            stdout=f"running {subcommand}\n{json.dumps(payloads[subcommand])}\n",
            stderr="",
        )

    monkeypatch.setattr(desktop.subprocess, "run", _fake_run)

    namespace = SimpleNamespace(json_output=True, timeout_seconds=15.0)
    assert desktop.run_gui_package_verify_command(namespace) == 0

    summary = json.loads(capsys.readouterr().out)
    assert [command[3] for command in commands] == [
        "gui-package-smoke",
        "gui-package-automation",
        "gui-package-bridge-recovery",
        "gui-package-main-window",
        "gui-package-tray",
        "gui-package-hotkey",
    ]
    assert all(command[:3] == [desktop.sys.executable, "-m", "local_ai_dictation.cli"] for command in commands)
    assert all(command[-1] == "--json" for command in commands)
    assert all(command[-3:-1] == ["--timeout-seconds", "15.0"] for command in commands)
    assert summary["command"] == "gui-package-verify"
    assert summary["installer_path"] == "/tmp/setup.exe"
    assert summary["windows_installer_path"] == "C:\\setup.exe"
    assert summary["checks"]["smoke"] == payloads["gui-package-smoke"]
    assert summary["checks"]["hotkey"] == payloads["gui-package-hotkey"]


def test_invoke_gui_e2e_action_tolerates_quit_connection_close(monkeypatch):
    def _fake_request_gui_e2e_json(*args, **kwargs):
        raise desktop.DesktopAppError("GUI automation request failed for /actions/quit: Remote end closed connection without response")

    monkeypatch.setattr(desktop, "request_gui_e2e_json", _fake_request_gui_e2e_json)

    assert desktop.invoke_gui_e2e_action(49016, "quit") == {
        "automationAction": "quit",
        "shutdownRequested": True,
    }


def test_run_gui_package_automation_command_launches_packaged_app_and_uses_localhost_hooks(monkeypatch, tmp_path: Path):
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir(parents=True)
    installed_root = tmp_path / "installed" / "stable"
    launcher_path = installed_root / "app" / "bin" / "launcher.exe"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("launcher")
    diagnostics_path = installed_root / "logs" / "startup-diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True)
    log_path = installed_root / "logs" / "startup.log"

    monkeypatch.setattr(
        desktop,
        "build_windows_package_payload",
        lambda: {
            "stage_root": str(stage_root),
            "installer_path": str(stage_root / "local-ai-dictation-desktop-Setup.exe"),
            "app_identifier": "local-ai-dictation.desktop.local",
            "app_channel": "stable",
        },
    )
    monkeypatch.setattr(
        desktop,
        "installed_windows_app_paths",
        lambda identifier, channel: {
            "app_root": installed_root,
            "app_dir": installed_root / "app",
            "launcher_path": launcher_path,
            "logs_dir": installed_root / "logs",
            "diagnostics_path": diagnostics_path,
            "log_path": log_path,
        },
    )
    monkeypatch.setattr(desktop, "windows_path_from_wsl", lambda path: f"C:\\stage\\{path.name}")
    monkeypatch.setattr(desktop, "prepare_installed_windows_app_for_reinstall", lambda installed_paths: None)

    installer_calls: list[tuple[Path, float]] = []

    def _fake_run_wsl_windows_executable_capture(executable_path: Path, *, timeout_seconds: float, env=None, cwd=None):
        installer_calls.append((executable_path, timeout_seconds))
        return _FakeCompletedProcess(returncode=0, stdout="installer ok", stderr="")

    monkeypatch.setattr(desktop, "run_wsl_windows_executable_capture", _fake_run_wsl_windows_executable_capture)

    popen_factory = _PopenFactory()

    def _fake_launch_wsl_windows_executable(executable_path: Path, *, env=None, cwd=None):
        diagnostics_path.write_text(
            json.dumps(
                {
                    "bunReady": True,
                    "rendererReady": True,
                    "rendererRpcReady": True,
                    "automationReady": True,
                    "shutdownReason": "automation:quit",
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("automation ready", encoding="utf-8")
        process = popen_factory([str(executable_path)], env=env, cwd=cwd)
        process.returncode = 0
        return process

    monkeypatch.setattr(desktop, "launch_wsl_windows_executable", _fake_launch_wsl_windows_executable)
    monkeypatch.setattr(desktop, "reserve_localhost_port", lambda: 49011)

    state_requests: list[str] = []
    action_requests: list[str] = []

    def _fake_wait_for_gui_e2e_state(port: int, predicate, *, timeout_seconds: float, poll_interval: float = 0.25):
        state = {
            "diagnostics": {"automationReady": True},
            "tray": {"created": True, "actions": ["open", "toggle", "quit"]},
            "hotkey": {"accelerator": "CommandOrControl+Alt+R", "registered": True},
            "renderer": {"ready": True, "rpcReady": True, "userAgent": "test-agent"},
            "bridge": {"connected": False},
        }
        state_requests.append(f"state:{port}:{timeout_seconds}")
        assert predicate(state) is True
        return state

    def _fake_invoke_gui_e2e_action(port: int, action: str, *, timeout_seconds: float = desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS):
        action_requests.append(f"{port}:{action}:{timeout_seconds}")
        return {"diagnostics": {"lastAutomationAction": action}}

    monkeypatch.setattr(desktop, "wait_for_gui_e2e_state", _fake_wait_for_gui_e2e_state)
    monkeypatch.setattr(desktop, "invoke_gui_e2e_action", _fake_invoke_gui_e2e_action)

    namespace = SimpleNamespace(json_output=False, timeout_seconds=12.0, automation_port=0)
    assert desktop.run_gui_package_automation_command(namespace) == 0
    assert installer_calls == [(
        stage_root / "local-ai-dictation-desktop-Setup.exe",
        12.0,
    )]
    assert len(popen_factory.calls) == 1
    assert popen_factory.calls[0].command == [str(launcher_path)]
    assert popen_factory.calls[0].env is not None
    assert popen_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E"] == "1"
    assert popen_factory.calls[0].env["LOCAL_AI_DICTATION_GUI_E2E_PORT"] == "49011"
    assert state_requests == ["state:49011:12.0"]
    assert action_requests == [
        f"49011:show-window:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49011:tray/open:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
        f"49011:quit:{desktop.DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS}",
    ]
    assert (stage_root / "smoke" / "startup-diagnostics.json").exists()
