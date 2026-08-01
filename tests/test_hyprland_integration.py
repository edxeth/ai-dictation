from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading


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
