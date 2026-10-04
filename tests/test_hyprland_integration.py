from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOGGLE_SCRIPT = ROOT / "integrations" / "hyprland" / "local-ai-dictation-toggle.sh"


class _FailedToggleHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler interface
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler interface
        body = json.dumps(
            {
                "state": "error",
                "last_error": "MODEL_TRANSCRIBE_FAILED: Willow result timeout",
            }
        ).encode("utf-8")
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


class _AuthFailureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler interface
        body = json.dumps({"ok": True, "bridge": {"model_loading": False}}).encode(
            "utf-8"
        )
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler interface
        body = json.dumps(
            {
                "state": "error",
                "last_error": (
                    "WILLOW_AUTH_REQUIRED: Willow session expired. Run `local-ai-dictation willow-session login` "
                    "(HTTP 400: Invalid Refresh Token)"
                ),
                "last_error_code": "WILLOW_AUTH_REQUIRED",
                "auth_required": True,
            }
        ).encode("utf-8")
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


class _LoadingBridgeState:
    def __init__(self) -> None:
        self.health_requests = 0
        self.post_before_ready = False


class _LoadingThenReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler interface
        state = self.server.state
        state.health_requests += 1
        loading = state.health_requests < 3
        body = json.dumps(
            {
                "ok": True,
                "bridge": {"model_loading": loading, "model_loaded": not loading},
                "session": {"state": "stopped"},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler interface
        state = self.server.state
        if state.health_requests < 3:
            state.post_before_ready = True
            body = b'{"error":"invalid_state","detail":"Model is still loading or warming"}'
            self.send_response(409)
        else:
            body = b'{"state":"starting"}'
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def test_hyprland_toggle_surfaces_bridge_failure(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FailedToggleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    state_home = tmp_path / "state"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": "/bin",
            "XDG_STATE_HOME": str(state_home),
            "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{server.server_address[1]}",
            "LOCAL_AI_DICTATION_BRIDGE_LAUNCHER": str(tmp_path / "missing-bridge"),
        }
    )

    try:
        result = subprocess.run(
            ["/bin/bash", str(TOGGLE_SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.returncode == 1
    assert "MODEL_TRANSCRIBE_FAILED: Willow result timeout" in result.stderr
    log = (state_home / "local-ai-dictation" / "hotkey.log").read_text(encoding="utf-8")
    assert "MODEL_TRANSCRIBE_FAILED: Willow result timeout" in log


def test_hyprland_toggle_waits_for_bridge_warmup(tmp_path: Path):
    state = _LoadingBridgeState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LoadingThenReadyHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    state_home = tmp_path / "state"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": "/bin",
            "XDG_STATE_HOME": str(state_home),
            "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{server.server_address[1]}",
            "LOCAL_AI_DICTATION_BRIDGE_LAUNCHER": str(tmp_path / "missing-bridge"),
        }
    )

    try:
        result = subprocess.run(
            ["/bin/bash", str(TOGGLE_SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.returncode == 0
    assert state.health_requests >= 3
    assert state.post_before_ready is False


def test_hyprland_toggle_health_hang_returns(tmp_path: Path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = threading.Event()

    def accept_and_hang() -> None:
        connection, _address = listener.accept()
        accepted.set()
        time.sleep(5)
        connection.close()

    accepter = threading.Thread(target=accept_and_hang, daemon=True)
    accepter.start()
    state_home = tmp_path / "state"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": "/bin",
            "XDG_STATE_HOME": str(state_home),
            "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{port}",
            "LOCAL_AI_DICTATION_BRIDGE_LAUNCHER": str(tmp_path / "missing-bridge"),
            "LOCAL_AI_DICTATION_HEALTH_CURL_MAX_TIME": "1",
            "LOCAL_AI_DICTATION_BRIDGE_START_ATTEMPTS": "1",
            "LOCAL_AI_DICTATION_TOGGLE_CURL_MAX_TIME": "1",
        }
    )

    started = time.monotonic()
    try:
        result = subprocess.run(
            ["/bin/bash", str(TOGGLE_SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    finally:
        listener.close()

    assert time.monotonic() - started < 6
    assert result.returncode == 1
    assert accepted.is_set()


def test_hyprland_toggle_connection_failure_uses_plain_notification(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FailedToggleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    notify_stub = stub_bin / "notify-send"
    notify_stub.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$*" >> "$STUB_LOG/notify-send.args"\n',
        encoding="utf-8",
    )
    notify_stub.chmod(0o755)

    state_home = tmp_path / "state"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{stub_bin}:/bin",
            "XDG_STATE_HOME": str(state_home),
            "STUB_LOG": str(tmp_path),
            "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{server.server_address[1]}",
            "LOCAL_AI_DICTATION_BRIDGE_LAUNCHER": str(tmp_path / "missing-bridge"),
        }
    )

    try:
        result = subprocess.run(
            ["/bin/bash", str(TOGGLE_SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.returncode == 1
    notify_args = (tmp_path / "notify-send.args").read_text(encoding="utf-8")
    assert "The connection to Willow failed, so the text was not pasted." in notify_args
    assert "Press Super+R to try again, or switch the model." in notify_args
    assert "schema_version" not in notify_args


def test_hyprland_toggle_auth_failure_notification_copies_login_command(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthFailureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    notify_stub = stub_bin / "notify-send"
    notify_stub.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                'printf "%s " "$@" >> "$STUB_LOG/notify-send.args"',
                'echo >> "$STUB_LOG/notify-send.args"',
                "echo copy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    notify_stub.chmod(0o755)
    wl_copy_stub = stub_bin / "wl-copy"
    wl_copy_stub.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                'cat > "$STUB_LOG/clipboard"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wl_copy_stub.chmod(0o755)

    state_home = tmp_path / "state"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{stub_bin}:/bin",
            "XDG_STATE_HOME": str(state_home),
            "STUB_LOG": str(tmp_path),
            "LOCAL_AI_DICTATION_BRIDGE_URL": f"http://127.0.0.1:{server.server_address[1]}",
            "LOCAL_AI_DICTATION_BRIDGE_LAUNCHER": str(tmp_path / "missing-bridge"),
        }
    )

    try:
        result = subprocess.run(
            ["/bin/bash", str(TOGGLE_SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.returncode == 1
    assert "WILLOW_AUTH_REQUIRED" in result.stderr

    clipboard = tmp_path / "clipboard"
    deadline = time.time() + 2.0
    while time.time() < deadline and not clipboard.exists():
        time.sleep(0.05)
    assert (
        clipboard.read_text(encoding="utf-8")
        == "local-ai-dictation willow-session login"
    )

    notify_args = (tmp_path / "notify-send.args").read_text(encoding="utf-8")
    assert "--action" in notify_args
    assert "copy=Copy sign-in command" in notify_args
    assert "Willow session expired" in notify_args
    assert (
        "Sign in again. Click to copy: local-ai-dictation willow-session login"
        in notify_args
    )
