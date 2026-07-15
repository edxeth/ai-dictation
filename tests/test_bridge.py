from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

bridge_module = importlib.import_module("local_ai_dictation.bridge")
BridgeStateError = bridge_module.BridgeStateError
DictationBridgeController = bridge_module.DictationBridgeController
TranscriptionResult = importlib.import_module("local_ai_dictation.types").TranscriptionResult


class _ClipboardSuccess:
    def __init__(self):
        self.calls: list[str] = []

    def copy(self, text: str) -> None:
        self.calls.append(text)


class _ClipboardFailure:
    def copy(self, text: str) -> None:
        raise RuntimeError("clipboard backend unavailable")


def _runtime_loader(debug: bool):
    return object(), object(), _ClipboardSuccess(), object()


def _make_runtime_loader(clipboard_module: Any):
    def _loader(debug: bool):
        return object(), object(), clipboard_module, object()

    return _loader


def _make_model_loader(delay: float = 0.0, calls: list[int] | None = None):
    def _loader(config, nemo_asr, torch_module, *, status_stream=None):
        if calls is not None:
            calls.append(1)
        if status_stream is not None:
            status_stream.write("⏳ Loading model\n")
        if delay:
            time.sleep(delay)
        return object(), False, 0.0, 0.0

    return _loader


def _recorder(config, pyaudio_module, sample_rate=16000, *, stop_requested=None, status_stream=None):
    if status_stream is not None:
        status_stream.write("🎤 Recording...\n")
    deadline = time.time() + 2.0
    while stop_requested is not None and not stop_requested():
        if time.time() >= deadline:
            break
        time.sleep(0.01)
    return b"audio"


def _transcriber(config, model, audio_data, sample_rate, *, status_stream=None):
    if status_stream is not None:
        status_stream.write("🤖 Generating...\n")
    return TranscriptionResult(text="fake transcript", device="cpu"), "/tmp/fake.wav", 0.0, 0.0


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)


def _wait_for_json(url: str, predicate, *, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.load(response)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        last_payload = payload
        if predicate(payload):
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for expected payload from {url}: {last_payload}")


def test_bridge_start_rejects_double_start():
    controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
    )

    started = controller.start_session()
    assert started["state"] == "starting"

    try:
        controller.start_session()
        raise AssertionError("expected BridgeStateError")
    except BridgeStateError as exc:
        assert "starting" in str(exc)
    finally:
        controller.shutdown()


def test_bridge_stop_during_model_load_cancels_cleanly():
    controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(delay=0.15),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
    )

    controller.start_session()
    stopped = controller.stop_session()

    assert stopped["state"] == "idle"
    assert stopped["last_transcript"]["transcript"] == ""
    assert stopped["last_transcript"]["metadata"]["cancelled_before_recording"] is True

    controller.shutdown()


def test_bridge_stop_cancels_willow_handshake_before_recording():
    session_created = threading.Event()
    cancelled = threading.Event()
    recorder_called = False

    class StreamingSession:
        def wait_ready(self) -> None:
            assert cancelled.wait(timeout=1)
            raise bridge_module.ModelError("MODEL_TRANSCRIBE_FAILED", "cancelled")

        def send_audio(self, pcm: bytes) -> None:
            raise AssertionError("audio must not be sent before readiness")

        def finish(self) -> str:
            raise AssertionError("cancelled session must not finish")

        def cancel(self) -> None:
            cancelled.set()

    class StreamingModel:
        def start_streaming(self) -> StreamingSession:
            session_created.set()
            return StreamingSession()

    def model_loader(config, runtime_module, torch_module, *, status_stream=None):
        return StreamingModel(), False, 0.0, 0.0

    def recorder(*args, **kwargs):
        nonlocal recorder_called
        recorder_called = True
        return b"audio"

    controller = DictationBridgeController(
        backend="willow",
        runtime_loader=_runtime_loader,
        model_loader=model_loader,
        recorder=recorder,
        transcript_timeout=1.0,
    )

    controller.start_session()
    assert session_created.wait(timeout=1)
    stopped = controller.stop_session()

    assert cancelled.is_set()
    assert recorder_called is False
    assert stopped["state"] == "idle"
    assert stopped["last_transcript"]["metadata"]["cancelled_before_recording"] is True

    controller.shutdown()


def test_bridge_warms_whisper_before_accepting_recording(monkeypatch):
    warmup_started = threading.Event()
    release_warmup = threading.Event()

    def warmup(model) -> None:
        warmup_started.set()
        assert release_warmup.wait(timeout=1)

    monkeypatch.setattr("local_ai_dictation.whisper.warmup", warmup)
    controller = DictationBridgeController(
        backend="whisper",
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
    )

    controller.start_model_warmup()
    assert warmup_started.wait(timeout=1)
    try:
        controller.start_session()
        raise AssertionError("recording must wait for model warmup")
    except BridgeStateError as exc:
        assert "warming" in str(exc)

    release_warmup.set()
    deadline = time.monotonic() + 1
    while controller.get_session_payload()["model_loading"] and time.monotonic() < deadline:
        time.sleep(0.01)

    started = controller.start_session()
    assert started["state"] == "starting"
    controller.stop_session()
    controller.shutdown()


def test_bridge_reuses_loaded_model_across_sessions():
    model_calls: list[int] = []
    controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(calls=model_calls),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
    )

    controller.start_session()
    time.sleep(0.05)
    first = controller.stop_session()
    controller.start_session()
    time.sleep(0.05)
    second = controller.stop_session()

    assert first["last_transcript"]["transcript"] == "fake transcript"
    assert second["last_transcript"]["transcript"] == "fake transcript"
    assert len(model_calls) == 1
    assert second["model_loaded"] is True
    assert len(second["history"]) == 2

    cleared = controller.clear_history()
    assert cleared["history"] == []
    assert cleared["last_transcript"] is None

    controller.shutdown()


def test_willow_bridge_streams_audio_while_recording(monkeypatch):
    sent_audio: list[bytes] = []
    session_created = threading.Event()
    allow_ready = threading.Event()
    audio_sent = threading.Event()
    finish_called = threading.Event()
    batch_transcriber_called = False

    class StreamingSession:
        def wait_ready(self) -> None:
            assert allow_ready.wait(timeout=1)

        def send_audio(self, pcm: bytes) -> None:
            sent_audio.append(pcm)
            audio_sent.set()

        def finish(self) -> str:
            finish_called.set()
            return "streamed transcript"

        def cancel(self) -> None:
            pass

    class StreamingModel:
        def start_streaming(self) -> StreamingSession:
            session_created.set()
            return StreamingSession()

    def model_loader(config, runtime_module, torch_module, *, status_stream=None):
        return StreamingModel(), False, 0.0, 0.0

    def recorder(config, pyaudio_module, sample_rate=16000, *, stop_requested=None, status_stream=None, on_audio_chunk=None):
        assert sample_rate == 16000
        assert on_audio_chunk is not None
        if status_stream is not None:
            status_stream.write("🎤 Recording...\n")
        pcm = b"\x01\x00" * 1600
        on_audio_chunk(pcm)
        deadline = time.monotonic() + 2.0
        while stop_requested is not None and not stop_requested() and time.monotonic() < deadline:
            time.sleep(0.01)
        return pcm

    def batch_transcriber(*args, **kwargs):
        nonlocal batch_transcriber_called
        batch_transcriber_called = True
        raise AssertionError("Willow should not use batch transcription")

    monkeypatch.setattr(bridge_module, "has_probable_speech", lambda *args, **kwargs: True)
    controller = DictationBridgeController(
        backend="willow",
        runtime_loader=_runtime_loader,
        model_loader=model_loader,
        recorder=recorder,
        transcriber=batch_transcriber,
        transcript_timeout=1.0,
    )

    controller.start_session()
    assert session_created.wait(timeout=1)
    assert controller.get_session_payload()["state"] == "starting"
    assert audio_sent.is_set() is False

    allow_ready.set()
    assert audio_sent.wait(timeout=1)
    assert controller.get_session_payload()["state"] == "recording"

    completed = controller.stop_session()

    assert b"".join(sent_audio) == b"\x01\x00" * 1600
    assert finish_called.is_set()
    assert batch_transcriber_called is False
    assert completed["state"] == "idle"
    assert completed["last_transcript"]["transcript"] == "streamed transcript"
    assert completed["last_transcript"]["metadata"]["backend"] == "willow"

    controller.shutdown()


def test_whisper_bridge_uses_streaming_model(monkeypatch):
    audio_sent = threading.Event()
    finish_called = threading.Event()
    batch_called = False

    class StreamingSession:
        def wait_ready(self) -> None:
            pass

        def send_audio(self, pcm: bytes) -> None:
            audio_sent.set()

        def finish(self) -> str:
            finish_called.set()
            return "rolling transcript"

        def cancel(self) -> None:
            pass

    class StreamingModel:
        _parakeet_device = "cuda"

        def start_streaming(self) -> StreamingSession:
            return StreamingSession()

    def model_loader(config, runtime_module, torch_module, *, status_stream=None):
        return StreamingModel(), True, 0.0, 0.0

    def recorder(config, pyaudio_module, sample_rate=16000, *, stop_requested=None, status_stream=None, on_audio_chunk=None):
        assert sample_rate == 16000
        assert on_audio_chunk is not None
        if status_stream is not None:
            status_stream.write("🎤 Recording...\n")
        pcm = b"\x01\x00" * 1600
        on_audio_chunk(pcm)
        while stop_requested is not None and not stop_requested():
            time.sleep(0.01)
        return pcm

    def batch_transcriber(*args, **kwargs):
        nonlocal batch_called
        batch_called = True
        raise AssertionError("Whisper should use rolling transcription")

    monkeypatch.setattr(bridge_module, "has_probable_speech", lambda *args, **kwargs: True)
    controller = DictationBridgeController(
        backend="whisper",
        runtime_loader=_runtime_loader,
        model_loader=model_loader,
        recorder=recorder,
        transcriber=batch_transcriber,
        transcript_timeout=1.0,
    )

    controller.start_session()
    assert audio_sent.wait(timeout=1)
    completed = controller.stop_session()

    assert finish_called.is_set()
    assert batch_called is False
    assert completed["last_transcript"]["transcript"] == "rolling transcript"
    assert completed["last_transcript"]["device"] == "cuda"
    assert completed["last_transcript"]["metadata"]["backend"] == "whisper"

    controller.shutdown()


def test_bridge_timeout_requests_cancel_without_waiting_for_whisper_worker(monkeypatch):
    finish_started = threading.Event()
    cancel_requested = threading.Event()
    release_inference = threading.Event()
    audio_sent = threading.Event()

    class StreamingSession:
        def wait_ready(self) -> None:
            pass

        def send_audio(self, pcm: bytes) -> None:
            audio_sent.set()

        def finish(self) -> str:
            finish_started.set()
            assert release_inference.wait(timeout=2)
            if cancel_requested.is_set():
                raise bridge_module.ModelError("MODEL_TRANSCRIBE_FAILED", "cancelled")
            return "late transcript"

        def cancel(self) -> None:
            cancel_requested.set()

        def wait_finished(self, timeout=None) -> bool:
            return release_inference.wait(timeout=timeout)

    class StreamingModel:
        _parakeet_device = "cuda"

        def start_streaming(self) -> StreamingSession:
            return StreamingSession()

    def model_loader(config, runtime_module, torch_module, *, status_stream=None):
        return StreamingModel(), True, 0.0, 0.0

    def recorder(config, pyaudio_module, sample_rate=16000, *, stop_requested=None, status_stream=None, on_audio_chunk=None):
        if status_stream is not None:
            status_stream.write("🎤 Recording...\n")
        pcm = b"\x01\x00" * 1600
        on_audio_chunk(pcm)
        while stop_requested is not None and not stop_requested():
            time.sleep(0.01)
        return pcm

    monkeypatch.setattr(bridge_module, "has_probable_speech", lambda *args, **kwargs: True)
    controller = DictationBridgeController(
        backend="whisper",
        runtime_loader=_runtime_loader,
        model_loader=model_loader,
        recorder=recorder,
        transcript_timeout=0.05,
    )

    controller.start_session()
    assert audio_sent.wait(timeout=1)
    stop_started = time.monotonic()
    try:
        controller.stop_session()
        raise AssertionError("stop should time out while inference is blocked")
    except TimeoutError:
        pass
    stop_elapsed = time.monotonic() - stop_started

    assert finish_started.is_set()
    assert cancel_requested.is_set()
    assert stop_elapsed < 1.5
    assert controller._session_finished.is_set() is False

    release_inference.set()
    assert controller._session_finished.wait(timeout=1)
    controller.shutdown()


def test_bridge_retains_audio_only_when_explicitly_enabled(tmp_path: Path):
    def even_pcm_recorder(config, pyaudio_module, sample_rate=16000, **kwargs):
        return b"\x00\x00" * 200

    default_dir = tmp_path / "default"
    default_controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(),
        recorder=even_pcm_recorder,
        transcriber=_transcriber,
        diagnostic_audio_dir=default_dir,
    )
    default_controller.start_session()
    assert default_controller._session_finished.wait(timeout=2)
    default_controller.shutdown()
    assert not default_dir.exists()

    retained_dir = tmp_path / "retained"
    retained_controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(),
        recorder=even_pcm_recorder,
        transcriber=_transcriber,
        retain_audio=True,
        diagnostic_audio_dir=retained_dir,
    )
    retained_controller.start_session()
    assert retained_controller._session_finished.wait(timeout=2)
    retained_controller.shutdown()

    retained = sorted(retained_dir.glob("*.wav"))
    assert [path.name for path in retained] == ["last-capture-16k.wav", "last-capture-raw.wav"]
    assert retained_dir.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in retained)


def test_bridge_records_recording_clipboard_and_paste_timestamps():
    clipboard = _ClipboardSuccess()
    paste_calls: list[str] = []

    def paste_notifier() -> bool:
        paste_calls.append(clipboard.calls[-1])
        return True

    controller = DictationBridgeController(
        runtime_loader=_make_runtime_loader(clipboard),
        model_loader=_make_model_loader(),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
        paste_notifier=paste_notifier,
    )

    controller.start_session()
    time.sleep(0.05)
    in_progress = controller.get_session_payload()
    assert in_progress["state"] == "recording"
    assert in_progress["recording_started_at"] is not None
    assert in_progress["clipboard_copied_at"] is None
    assert in_progress["paste_dispatched_at"] is None

    completed = controller.stop_session()
    assert completed["last_transcript"]["transcript"] == "fake transcript"
    assert completed["last_completed_at"] is not None
    assert completed["clipboard_copied_at"] is not None
    assert completed["paste_dispatched_at"] is not None
    assert completed["clipboard_copied_at"] <= completed["paste_dispatched_at"] <= completed["last_completed_at"]
    assert clipboard.calls == ["fake transcript"]
    assert paste_calls == ["fake transcript"]

    controller.shutdown()



def test_bridge_leaves_clipboard_timestamp_unset_when_copy_fails():
    controller = DictationBridgeController(
        runtime_loader=_make_runtime_loader(_ClipboardFailure()),
        model_loader=_make_model_loader(),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
    )

    controller.start_session()
    time.sleep(0.05)
    completed = controller.stop_session()

    assert completed["last_transcript"]["transcript"] == "fake transcript"
    assert completed["clipboard_copied_at"] is None
    assert any("Clipboard warning:" in line for line in completed["stderr_tail"])

    controller.shutdown()



def test_bridge_status_notifier_fires_on_state_changes():
    notifications: list[str] = []
    controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
        status_notifier=lambda: notifications.append(controller.get_session_payload()["state"]),
    )

    started = controller.start_session()
    assert started["state"] == "starting"
    time.sleep(0.05)
    completed = controller.stop_session()

    assert completed["state"] == "idle"
    assert notifications[0] == "starting"
    assert "recording" in notifications
    assert "transcribing" in notifications
    assert notifications[-1] == "idle"

    controller.shutdown()



def test_bridge_server_health_and_session_endpoints():
    controller = DictationBridgeController(
        runtime_loader=_runtime_loader,
        model_loader=_make_model_loader(),
        recorder=_recorder,
        transcriber=_transcriber,
        transcript_timeout=1.0,
    )
    server = bridge_module.make_bridge_server("127.0.0.1", 0, controller=controller)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            health = json.load(response)
        assert health["schema_version"] == 1
        assert health["session"]["state"] == "stopped"
        assert health["bridge"]["model_loaded"] is False

        start_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/session/start",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(start_request, timeout=2) as response:
            started = json.load(response)
        assert started["state"] == "starting"

        time.sleep(0.05)
        stop_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/session/stop",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(stop_request, timeout=3) as response:
            stopped = json.load(response)
        assert stopped["state"] == "idle"
        assert stopped["last_transcript"]["transcript"] == "fake transcript"
    finally:
        server.shutdown()
        server.server_close()
        controller.shutdown()
        thread.join(timeout=1)


def test_hyprland_paste_notifier_uses_terminal_shortcut(monkeypatch):
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == "activewindow":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"class": "com.mitchellh.ghostty"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bridge_module.shutil, "which", lambda name: "/usr/bin/hyprctl" if name == "hyprctl" else None)
    monkeypatch.setattr(bridge_module.subprocess, "run", run)

    notifier = bridge_module._build_hyprland_paste_notifier({"LOCAL_AI_DICTATION_HYPRLAND_PASTE": "1"})

    assert notifier is not None
    assert notifier() is True
    assert calls == [
        ["/usr/bin/hyprctl", "-j", "activewindow"],
        ["/usr/bin/hyprctl", "dispatch", "sendshortcut", "CTRL SHIFT,V,activewindow"],
    ]


def test_hyprland_paste_notifier_times_out_without_blocking(monkeypatch):
    monkeypatch.setattr(bridge_module.shutil, "which", lambda name: "/usr/bin/hyprctl")

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bridge_module.subprocess, "run", run)
    notifier = bridge_module._build_hyprland_paste_notifier(
        {"LOCAL_AI_DICTATION_HYPRLAND_PASTE": "1"}
    )

    assert notifier is not None
    assert notifier() is False


def test_build_bridge_controller_from_namespace_reads_e2e_env():
    namespace = type("Namespace", (), {"cpu": False, "clipboard": True})()

    controller = bridge_module.build_bridge_controller_from_namespace(
        namespace,
        env={
            "LOCAL_AI_DICTATION_E2E_MODE": "1",
            "LOCAL_AI_DICTATION_E2E_TRANSCRIPT": "scripted transcript",
            "LOCAL_AI_DICTATION_E2E_START_DELAY_MS": "25",
            "LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS": "35",
        },
    )

    payload = controller.health_payload()
    assert payload["bridge"]["e2e_mode"] is True
    assert payload["bridge"]["model_loaded"] is True
    assert payload["session"]["model_loading"] is False
    controller.shutdown()



def test_build_bridge_controller_from_namespace_parses_numeric_input_device():
    namespace = type("Namespace", (), {"cpu": False, "clipboard": True, "input_device": "7"})()

    controller = bridge_module.build_bridge_controller_from_namespace(namespace, env={})

    assert controller.get_session_payload()["config"]["input_device"] == 7
    controller.shutdown()


def test_bridge_cli_e2e_mode_exposes_deterministic_session_flow():
    port = _reserve_local_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["LOCAL_AI_DICTATION_E2E_MODE"] = "1"
    env["LOCAL_AI_DICTATION_E2E_TRANSCRIPT"] = "deterministic transcript"
    env["LOCAL_AI_DICTATION_E2E_START_DELAY_MS"] = "30"
    env["LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS"] = "40"

    process = subprocess.Popen(
        [sys.executable, "-m", "local_ai_dictation.cli", "bridge", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        health = _wait_for_json(
            f"http://127.0.0.1:{port}/health",
            lambda payload: bool(payload.get("ok")),
            timeout=5.0,
        )
        assert health["bridge"]["e2e_mode"] is True
        assert health["bridge"]["model_loaded"] is True
        assert health["session"]["state"] == "stopped"

        started = _post_json(f"http://127.0.0.1:{port}/session/start")
        assert started["state"] == "starting"

        recording = _wait_for_json(
            f"http://127.0.0.1:{port}/session",
            lambda payload: payload.get("state") == "recording",
        )
        assert recording["stderr_tail"][-1] == "🎤 Recording..."
        assert recording["recording_started_at"] is not None

        stopped = _post_json(f"http://127.0.0.1:{port}/session/stop")
        assert stopped["state"] == "idle"
        assert stopped["last_transcript"]["transcript"] == "deterministic transcript"
        assert stopped["last_transcript"]["metadata"]["e2e_mode"] is True
        assert stopped["clipboard_copied_at"] is not None
        assert len(stopped["history"]) == 1

        session = _wait_for_json(
            f"http://127.0.0.1:{port}/session",
            lambda payload: len(payload.get("history", [])) == 1,
        )
        assert session["history"][0]["payload"]["transcript"] == "deterministic transcript"

        toggle_started = _post_json(f"http://127.0.0.1:{port}/session/toggle")
        assert toggle_started["state"] == "starting"
        _wait_for_json(
            f"http://127.0.0.1:{port}/session",
            lambda payload: payload.get("state") == "recording",
        )
        toggle_stopped = _post_json(f"http://127.0.0.1:{port}/session/toggle")
        assert toggle_stopped["state"] == "idle"
        assert len(toggle_stopped["history"]) == 2

        cleared = _post_json(f"http://127.0.0.1:{port}/session/clear-history")
        assert cleared["history"] == []
        assert cleared["last_transcript"] is None
    finally:
        stdout = ""
        if process.poll() is None:
            process.terminate()
            try:
                stdout, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate(timeout=5)
        assert process.returncode == 0, stdout
