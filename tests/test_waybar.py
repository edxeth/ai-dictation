from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS_SCRIPT = ROOT / "integrations/waybar/local-ai-dictation-status.py"
SWITCH_SCRIPT = ROOT / "integrations/hyprland/local-ai-dictation-switch-model.sh"


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
