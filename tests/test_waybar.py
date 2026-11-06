from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from local_ai_dictation.backend_state import BACKEND_LABELS

ROOT = Path(__file__).resolve().parents[1]
STATUS_SCRIPT = ROOT / "integrations/waybar/local-ai-dictation-status.py"
SWITCH_SCRIPT = ROOT / "integrations/hyprland/local-ai-dictation-switch-model.sh"
MENU_SCRIPT = ROOT / "integrations/hyprland/local-ai-dictation-model-menu.sh"


def _run_status(
    tmp_path: Path, payload, preferred: str, extra_env: dict[str, str] | None = None
) -> dict:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    state_home = tmp_path / "state"
    backend_file = state_home / "local-ai-dictation/backend.json"
    backend_file.parent.mkdir(parents=True, exist_ok=True)
    backend_file.write_text(json.dumps({"backend": preferred}), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, str(STATUS_SCRIPT)],
            env={
                **os.environ,
                "XDG_STATE_HOME": str(state_home),
                "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{server.server_port}",
                **(extra_env or {}),
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    return json.loads(result.stdout)


# Hyprland 0.56 answers `hyprctl eval 'return 1'` with "ok" under the Lua config
# manager; the legacy manager rejects eval with this message.
LEGACY_EVAL_REPLY = "eval is only supported with the lua config manager"


@pytest.mark.parametrize(
    ("eval_reply", "window_class", "dispatch"),
    [
        pytest.param(
            "ok",
            "chromium",
            [
                "dispatch",
                'hl.dsp.send_shortcut({ mods = "CTRL", key = "V", window = "activewindow" })',
            ],
            id="lua-app",
        ),
        pytest.param(
            LEGACY_EVAL_REPLY,
            "com.mitchellh.ghostty",
            ["dispatch", "sendshortcut", "CTRL SHIFT,V,activewindow"],
            id="legacy-terminal",
        ),
    ],
)
def test_waybar_pastes_a_new_transcript_in_the_active_config_dialect(
    tmp_path, eval_reply, window_class, dispatch
):
    commands = tmp_path / "bin"
    commands.mkdir()
    hyprctl = commands / "hyprctl"
    hyprctl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['STUB_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(args) + '\\n')\n"
        "if args[:1] == ['eval']:\n"
        "    print(os.environ['STUB_EVAL_REPLY'])\n"
        "elif args == ['-j', 'activewindow']:\n"
        "    print(json.dumps({'class': os.environ['STUB_WINDOW_CLASS']}))\n"
        "else:\n"
        "    print('ok')\n",
        encoding="utf-8",
    )
    hyprctl.chmod(0o755)
    calls_log = tmp_path / "hyprctl.calls"
    # A previous completion is on record, so the new one is a fresh transcript.
    last_completed = tmp_path / "state/local-ai-dictation-waybar/last-completed-at"
    last_completed.parent.mkdir(parents=True)
    last_completed.write_text("100.0", encoding="utf-8")
    payload = {
        "bridge": {"model_loaded": True},
        "session": {
            "state": "idle",
            "config": {"backend": "whisper"},
            "last_completed_at": 200.0,
            "paste_dispatched_at": None,
            "last_transcript": {"transcript": "hello world"},
        },
    }

    _run_status(
        tmp_path,
        payload,
        "whisper",
        extra_env={
            "PATH": f"{commands}:{os.environ['PATH']}",
            "STUB_LOG": str(calls_log),
            "STUB_EVAL_REPLY": eval_reply,
            "STUB_WINDOW_CLASS": window_class,
        },
    )

    calls = [json.loads(line) for line in calls_log.read_text().splitlines()]
    assert [call for call in calls if call[0] == "dispatch"] == [dispatch]
    assert last_completed.read_text(encoding="utf-8") == "200.0"


@pytest.mark.parametrize("backend", list(BACKEND_LABELS))
def test_waybar_names_every_backend_saved_or_running(tmp_path, backend):
    label = BACKEND_LABELS[backend]
    running = {
        "bridge": {"model_loaded": True},
        "session": {"state": "idle", "config": {"backend": backend}},
    }

    offline = _run_status(tmp_path / "offline", None, backend)
    connected = _run_status(tmp_path / "connected", running, "whisper")

    assert offline["text"] == f"󰍬 {label}"
    assert offline["class"] == "offline"
    assert connected["text"] == f"󰍬 {label}"
    assert connected["tooltip"].startswith(f"{label} ready")


@pytest.mark.parametrize("connected", [False, True])
def test_waybar_displays_ultra_for_the_existing_parakeet_key(tmp_path, connected):
    payload = (
        {
            "bridge": {"model_loaded": True},
            "session": {"state": "idle", "config": {"backend": "parakeet"}},
        }
        if connected
        else None
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    state_home = tmp_path / "state"
    backend_file = state_home / "local-ai-dictation/backend.json"
    backend_file.parent.mkdir(parents=True)
    # When connected, the running bridge takes precedence over the saved choice.
    preferred = {"backend": "willow" if connected else "parakeet"}
    backend_file.write_text(json.dumps(preferred), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, str(STATUS_SCRIPT)],
            env={
                **os.environ,
                "XDG_STATE_HOME": str(state_home),
                "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{server.server_port}",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    status = json.loads(result.stdout)
    assert status["text"] == "󰍬 Parakeet Ultra"
    assert "Parakeet Ultra" in status["tooltip"]
    assert status["class"] == ("idle" if connected else "offline")
    assert json.loads(backend_file.read_text()) == preferred


def test_waybar_switch_preserves_the_key_restarts_bridge_and_names_ultra(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    backend_file = tmp_path / "backend"
    backend_file.write_text("whisper\n", encoding="utf-8")
    cli = commands / "dictation-cli"
    cli.write_text(
        "#!/bin/bash\n"
        'case "$1 $2" in\n'
        '  "backend get") cat "$STUB_LOG/backend" ;;\n'
        '  "backend toggle") printf "%s\\n" "$*" > "$STUB_LOG/toggle.args"; '
        'printf "parakeet\\n" | tee "$STUB_LOG/backend" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    for name in ("notify-send", "pkill"):
        command = commands / name
        command.write_text(
            f'#!/bin/bash\nprintf "%s\\n" "$@" > "$STUB_LOG/{name}.args"\n',
            encoding="utf-8",
        )
        command.chmod(0o755)

    subprocess.run(
        ["/bin/bash", str(SWITCH_SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{commands}:/bin",
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "STUB_LOG": str(tmp_path),
            "LOCAL_AI_DICTATION_CLI": str(cli),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    assert backend_file.read_text().strip() == "parakeet"
    assert (
        tmp_path / "toggle.args"
    ).read_text().strip() == "backend toggle --restart-bridge"
    assert (
        "Model switched to Parakeet Ultra"
        in (tmp_path / "notify-send.args").read_text()
    )
    assert (tmp_path / "pkill.args").read_text().splitlines() == ["-RTMIN+8", "waybar"]


@pytest.mark.parametrize(
    ("after", "label"),
    [
        ("parakeet-redux", "Parakeet Redux"),
        ("bark", "Bark 3.0"),
        ("bark-pro", "Bark 3.0 Pro"),
    ],
)
def test_waybar_switch_names_the_new_local_models(tmp_path, after, label):
    commands = tmp_path / "bin"
    commands.mkdir()
    cli = commands / "dictation-cli"
    cli.write_text(
        "#!/bin/bash\n"
        'case "$1 $2" in\n'
        '  "backend get") cat "$STUB_LOG/backend" 2>/dev/null || echo whisper ;;\n'
        f'  "backend toggle") printf "{after}\\n" > "$STUB_LOG/backend" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    for name in ("notify-send", "pkill"):
        command = commands / name
        command.write_text(
            f'#!/bin/bash\nprintf "%s\\n" "$@" > "$STUB_LOG/{name}.args"\n',
            encoding="utf-8",
        )
        command.chmod(0o755)

    subprocess.run(
        ["/bin/bash", str(SWITCH_SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{commands}:/bin",
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "STUB_LOG": str(tmp_path),
            "LOCAL_AI_DICTATION_CLI": str(cli),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    assert f"Model switched to {label}" in (tmp_path / "notify-send.args").read_text()


# System tools the menu script and its stubs call. The test PATH holds only
# these and the stubs, so a menu tool installed on the host is never found.
MENU_SYSTEM_TOOLS = ("cat", "date", "head", "mkdir", "sed")


def _menu_commands(
    tmp_path: Path, *, choice: str, cli_status: int = 0, pickers: tuple[str, ...]
) -> Path:
    commands = tmp_path / "bin"
    commands.mkdir()
    for tool in MENU_SYSTEM_TOOLS:
        (commands / tool).symlink_to(shutil.which(tool))
    # Each picker stub records which picker ran, its arguments and its stdin,
    # then exits 1 without output when dismissed, as dmenu-style pickers do.
    for picker in pickers:
        (commands / picker).write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" {picker} > "$STUB_LOG/menu.name"\n'
            'printf "%s\\n" "$@" > "$STUB_LOG/menu.args"\n'
            'cat > "$STUB_LOG/menu.stdin"\n'
            '[ -n "$STUB_CHOICE" ] || exit 1\n'
            'printf "%s\\n" "$STUB_CHOICE"\n',
            encoding="utf-8",
        )
    cli = commands / "dictation-cli"
    cli.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$STUB_LOG/cli.calls"\n'
        f"exit {cli_status}\n",
        encoding="utf-8",
    )
    for name in ("notify-send", "pkill"):
        command = commands / name
        command.write_text(
            f'#!/bin/bash\nprintf "%s\\n" "$*" >> "$STUB_LOG/{name}.calls"\n',
            encoding="utf-8",
        )
    for command in commands.iterdir():
        if not command.is_symlink():
            command.chmod(0o755)
    return commands


def _run_menu(
    tmp_path: Path,
    *,
    saved: str | None,
    choice: str,
    cli_status: int = 0,
    pickers: tuple[str, ...] = ("vicinae",),
    env: dict[str, str] | None = None,
):
    commands = _menu_commands(
        tmp_path, choice=choice, cli_status=cli_status, pickers=pickers
    )
    base_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("LOCAL_AI_DICTATION_MENU_CMD", "LOCAL_AI_DICTATION_WALKER_THEME")
    }
    state_home = tmp_path / "state"
    if saved is not None:
        backend_file = state_home / "local-ai-dictation/backend.json"
        backend_file.parent.mkdir(parents=True)
        backend_file.write_text(json.dumps({"backend": saved}) + "\n", encoding="utf-8")
    return subprocess.run(
        ["/bin/bash", str(MENU_SCRIPT)],
        env={
            **base_env,
            "PATH": str(commands),
            "XDG_STATE_HOME": str(state_home),
            "STUB_LOG": str(tmp_path),
            "STUB_CHOICE": choice,
            "LOCAL_AI_DICTATION_CLI": str(commands / "dictation-cli"),
            **(env or {}),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )


def _calls(tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / f"{name}.calls"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _menu_call(tmp_path: Path) -> tuple[str, list[str]]:
    name = (tmp_path / "menu.name").read_text(encoding="utf-8").strip()
    return name, (tmp_path / "menu.args").read_text(encoding="utf-8").splitlines()


def test_model_menu_lists_every_backend_and_switches_straight_to_the_pick(tmp_path):
    result = _run_menu(
        tmp_path, saved="willow", choice="󰍬 Bark 3.0 Pro · English · most accurate"
    )

    assert result.returncode == 0, result.stderr
    menu = (tmp_path / "menu.stdin").read_text(encoding="utf-8").splitlines()
    assert [line.split(" · ")[0][2:] for line in menu] == list(BACKEND_LABELS.values())
    assert menu[-1] == "● Willow · current"
    # Hints name only what matters for choosing on a GPU; no download sizes.
    assert menu[:-1] == [
        "󰍬 Whisper · Distil large-v3.5",
        "󰍬 Parakeet Ultra · 25 languages",
        "󰍬 Parakeet Redux · 25 languages · less accurate",
        "󰍬 Bark 3.0 · English · less accurate",
        "󰍬 Bark 3.0 Pro · English · most accurate",
    ]
    # One direct switch: a single restart that loads only the chosen model.
    assert _calls(tmp_path, "cli") == ["backend set bark-pro --restart-bridge"]
    assert "Model switched to Bark 3.0 Pro" in _calls(tmp_path, "notify-send")[-1]
    assert _calls(tmp_path, "pkill") == ["-RTMIN+8 waybar"]


@pytest.mark.parametrize(
    ("saved", "choice"),
    [
        ("bark", "● Bark 3.0 · current"),  # the running choice needs no reload
        ("bark", ""),  # menu dismissed with Escape
        ("bark", "something typed into the search box"),
    ],
)
def test_model_menu_does_nothing_without_a_new_choice(tmp_path, saved, choice):
    result = _run_menu(tmp_path, saved=saved, choice=choice)

    assert result.returncode == 0, result.stderr
    assert _calls(tmp_path, "cli") == []
    assert _calls(tmp_path, "pkill") == []


def test_model_menu_marks_whisper_current_when_nothing_was_saved(tmp_path):
    _run_menu(tmp_path, saved=None, choice="")

    menu = (tmp_path / "menu.stdin").read_text(encoding="utf-8").splitlines()
    assert menu[0] == "● Whisper · current"
    assert sum(line.startswith("● ") for line in menu) == 1


def test_model_menu_reports_a_failed_switch(tmp_path):
    result = _run_menu(
        tmp_path,
        saved="whisper",
        choice="󰍬 Parakeet Redux · 25 languages · less accurate",
        cli_status=1,
    )

    assert result.returncode == 1
    assert _calls(tmp_path, "cli") == ["backend set parakeet-redux --restart-bridge"]
    assert "Could not switch to Parakeet Redux" in _calls(tmp_path, "notify-send")[-1]
    assert _calls(tmp_path, "pkill") == ["-RTMIN+8 waybar"]


# Each picker's dmenu-mode flags, from its documentation. Walker's flags are the
# ones the 1.3.0 menu used.
@pytest.mark.parametrize(
    ("picker", "args"),
    [
        ("vicinae", ["dmenu", "-p", "Dictation model"]),
        ("walker", ["-d", "-e", "-p", "Dictation model"]),
        ("fuzzel", ["--dmenu", "--prompt", "Dictation model: "]),
        ("rofi", ["-dmenu", "-p", "Dictation model"]),
        ("wofi", ["--dmenu", "-p", "Dictation model"]),
    ],
)
def test_model_menu_opens_the_installed_picker(tmp_path, picker, args):
    result = _run_menu(
        tmp_path, saved="whisper", choice="󰍬 Willow · Cloud", pickers=(picker,)
    )

    assert result.returncode == 0, result.stderr
    assert _menu_call(tmp_path) == (picker, args)
    assert _calls(tmp_path, "cli") == ["backend set willow --restart-bridge"]


def test_model_menu_prefers_vicinae_over_other_installed_pickers(tmp_path):
    _run_menu(
        tmp_path, saved="whisper", choice="", pickers=("wofi", "walker", "vicinae")
    )

    assert _menu_call(tmp_path)[0] == "vicinae"


def test_model_menu_keeps_the_walker_theme_setting(tmp_path):
    _run_menu(
        tmp_path,
        saved="whisper",
        choice="",
        pickers=("walker",),
        env={"LOCAL_AI_DICTATION_WALKER_THEME": "pi"},
    )

    assert _menu_call(tmp_path) == (
        "walker",
        ["-d", "-e", "-p", "Dictation model", "-t", "pi"],
    )


# A blank setting counts as unset, as with other shell variables: the script
# picks an installed picker instead of failing.
@pytest.mark.parametrize("command", ["", "   "])
def test_model_menu_treats_a_blank_command_as_unset(tmp_path, command):
    result = _run_menu(
        tmp_path,
        saved="whisper",
        choice="",
        pickers=("vicinae",),
        env={"LOCAL_AI_DICTATION_MENU_CMD": command},
    )

    assert result.returncode == 0, result.stderr
    assert _menu_call(tmp_path)[0] == "vicinae"
    assert _calls(tmp_path, "notify-send") == []


def test_model_menu_runs_the_configured_command_as_given(tmp_path):
    result = _run_menu(
        tmp_path,
        saved="whisper",
        choice="󰍬 Bark 3.0 · English · less accurate",
        pickers=("vicinae", "mypicker"),
        env={"LOCAL_AI_DICTATION_MENU_CMD": "mypicker --dmenu --lines 6"},
    )

    assert result.returncode == 0, result.stderr
    assert _menu_call(tmp_path) == ("mypicker", ["--dmenu", "--lines", "6"])
    assert _calls(tmp_path, "cli") == ["backend set bark --restart-bridge"]


@pytest.mark.parametrize(
    ("pickers", "env", "message"),
    [
        ((), {}, "No menu tool found"),
        (
            ("vicinae",),
            {"LOCAL_AI_DICTATION_MENU_CMD": "missing-picker --dmenu"},
            "missing-picker",
        ),
    ],
)
def test_model_menu_reports_a_missing_picker(tmp_path, pickers, env, message):
    result = _run_menu(tmp_path, saved="whisper", choice="", pickers=pickers, env=env)

    assert result.returncode == 1
    assert message in _calls(tmp_path, "notify-send")[-1]
    assert _calls(tmp_path, "cli") == []
    assert _calls(tmp_path, "pkill") == []
