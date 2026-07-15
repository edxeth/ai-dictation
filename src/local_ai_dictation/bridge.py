"""Opt-in localhost bridge for controlling Local AI Dictation from a desktop app."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse

from local_ai_dictation.audio import has_probable_speech, list_input_devices, resample_pcm16_mono, resolve_input_sample_rate
from local_ai_dictation.backend_state import get_backend
from local_ai_dictation.dictation import (
    _load_model,
    _load_runtime_dependencies,
    _transcribe_once,
    record_audio_interruptible,
    save_audio,
)
from local_ai_dictation.doctor import collect_doctor_report
from local_ai_dictation.errors import ExitCode, ModelError
from local_ai_dictation.output import copy_transcript_to_clipboard, render_transcription
from local_ai_dictation.types import DictationConfig, TranscriptionResult


BRIDGE_SCHEMA_VERSION = 1
DEFAULT_E2E_TRANSCRIPT = "Local AI Dictation deterministic E2E transcript"
DIAGNOSTIC_RAW_FILENAME = "last-capture-raw.wav"
DIAGNOSTIC_MODEL_INPUT_FILENAME = "last-capture-16k.wav"
HYPRCTL_TIMEOUT_SECONDS = 0.25


def _build_hyprland_paste_notifier(env: Mapping[str, str]) -> Callable[[], bool] | None:
    enabled = str(env.get("LOCAL_AI_DICTATION_HYPRLAND_PASTE", "")).strip().lower()
    hyprctl = shutil.which("hyprctl")
    if enabled not in {"1", "true", "yes", "on"} or hyprctl is None:
        return None

    def _paste() -> bool:
        try:
            active_window = subprocess.run(
                [hyprctl, "-j", "activewindow"],
                capture_output=True,
                text=True,
                timeout=HYPRCTL_TIMEOUT_SECONDS,
                check=False,
            )
            window = json.loads(active_window.stdout) if active_window.returncode == 0 else {}
            window_class = str(window.get("class") or "") if isinstance(window, dict) else ""
            shortcut = "CTRL SHIFT,V,activewindow" if window_class == "com.mitchellh.ghostty" else "CTRL,V,activewindow"
            dispatched = subprocess.run(
                [hyprctl, "dispatch", "sendshortcut", shortcut],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=HYPRCTL_TIMEOUT_SECONDS,
                check=False,
            )
            return dispatched.returncode == 0
        except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return False

    return _paste


def _build_status_notifier(env: Mapping[str, str]) -> Callable[[], None] | None:
    raw_signal = str(env.get("LOCAL_AI_DICTATION_WAYBAR_SIGNAL", "")).strip()
    if not raw_signal:
        return None
    if not raw_signal.isdigit():
        return None

    process_name = str(env.get("LOCAL_AI_DICTATION_WAYBAR_PROCESS", "waybar")).strip() or "waybar"

    def _notify() -> None:
        def _signal_waybar() -> None:
            subprocess.run(
                ["pkill", f"-RTMIN+{raw_signal}", process_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        threading.Thread(target=_signal_waybar, daemon=True).start()

    return _notify


class BridgeStateError(RuntimeError):
    """Raised when a bridge request is invalid for the current session state."""


class _DiagnosticStream:
    def __init__(self, controller: "DictationBridgeController") -> None:
        self._controller = controller

    def write(self, text: str) -> int:
        normalized = text.replace("\x1b[2K", "").replace("\r", "\n")
        if "Loading model" in normalized:
            self._controller._append_diagnostic("⏳ Loading model")
            return len(text)
        if "Generating" in normalized:
            self._controller._append_diagnostic("🤖 Generating...")
            return len(text)
        for line in normalized.splitlines():
            stripped = line.strip()
            if stripped:
                self._controller._append_diagnostic(stripped)
        return len(text)

    def flush(self) -> None:
        return None


class DictationBridgeController:
    def __init__(
        self,
        *,
        backend: str = "whisper",
        cpu: bool = False,
        input_device: int | str | None = None,
        vad: bool = False,
        max_silence_ms: int = 1200,
        min_speech_ms: int = 300,
        vad_mode: int = 2,
        debug: bool = False,
        log_file: str = "transcriber.debug.log",
        clipboard: bool = True,
        transcript_timeout: float = 300.0,
        stderr_tail_limit: int = 200,
        e2e_mode: bool = False,
        e2e_transcript: str = DEFAULT_E2E_TRANSCRIPT,
        e2e_start_delay_ms: int = 0,
        e2e_stop_delay_ms: int = 0,
        status_notifier: Callable[[], None] | None = None,
        paste_notifier: Callable[[], bool] | None = None,
        retain_audio: bool = False,
        diagnostic_audio_dir: Path | None = None,
        runtime_loader: Callable[..., tuple[Any, Any, Any, Any]] = _load_runtime_dependencies,
        model_loader: Callable[..., tuple[Any, bool, float, float]] = _load_model,
        recorder: Callable[..., bytes | None] = record_audio_interruptible,
        transcriber: Callable[..., tuple[TranscriptionResult, str, float, float]] = _transcribe_once,
    ) -> None:
        self.backend = backend
        self.cpu = cpu
        self.input_device = input_device
        self.vad = vad
        self.max_silence_ms = max_silence_ms
        self.min_speech_ms = min_speech_ms
        self.vad_mode = vad_mode
        self.debug = debug
        self.log_file = log_file
        self.clipboard = clipboard
        self._transcript_timeout = transcript_timeout
        self._stderr_tail_limit = stderr_tail_limit
        self._e2e_mode = e2e_mode
        self._e2e_transcript = e2e_transcript
        self._e2e_start_delay_ms = max(0, e2e_start_delay_ms)
        self._e2e_stop_delay_ms = max(0, e2e_stop_delay_ms)
        self._status_notifier = status_notifier
        self._paste_notifier = paste_notifier
        self._retain_audio = retain_audio
        self._diagnostic_audio_dir = diagnostic_audio_dir or (
            Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
            / "local-ai-dictation"
            / "diagnostic-audio"
        )
        self._runtime_loader = runtime_loader
        self._model_loader = model_loader
        self._recorder = recorder
        self._transcriber = transcriber

        self._lock = threading.RLock()
        self._event_condition = threading.Condition(self._lock)
        self._event_version = 0
        self._runtime_ready = threading.Event()
        self._session_finished = threading.Event()
        self._stop_requested = threading.Event()
        self._shutdown_requested = threading.Event()
        self._session_thread: threading.Thread | None = None
        self._active_streaming_session: Any | None = None
        self._state = "stopped"
        self._started_at: float | None = None
        self._recording_started_at: float | None = None
        self._last_completed_at: float | None = None
        self._clipboard_copied_at: float | None = None
        self._paste_dispatched_at: float | None = None
        self._last_transcript: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._stderr_tail: list[str] = []
        self._history: list[dict[str, Any]] = []

        self._runtime_module: Any | None = None
        self._pyaudio_module: Any | None = None
        self._pyperclip_module: Any | None = None
        self._torch_module: Any | None = None
        self._model: Any | None = None
        self._model_loaded = self._e2e_mode
        self._model_loading = False
        self._warmup_thread: threading.Thread | None = None
        self._diagnostic_stream = _DiagnosticStream(self)

        if self._e2e_mode:
            self._append_diagnostic("Deterministic E2E bridge mode enabled")

    def _notify_status(self) -> None:
        with self._lock:
            self._event_version += 1
            self._event_condition.notify_all()
        if self._status_notifier is None:
            return
        try:
            self._status_notifier()
        except Exception:
            return

    def _append_diagnostic(self, text: str) -> None:
        should_echo = False
        should_notify = False
        with self._lock:
            if not text:
                return
            if self._stderr_tail and self._stderr_tail[-1] == text:
                return
            self._stderr_tail.append(text)
            should_echo = True
            if len(self._stderr_tail) > self._stderr_tail_limit:
                self._stderr_tail = self._stderr_tail[-self._stderr_tail_limit :]
            if text == "🎤 Recording...":
                self._recording_started_at = time.time()
                if self._state == "starting":
                    self._state = "recording"
                    should_notify = True
        if should_echo:
            print(text, file=sys.stdout, flush=True)
        if should_notify:
            self._notify_status()

    def _config(self) -> DictationConfig:
        return DictationConfig(
            backend=self.backend,
            cpu=self.cpu,
            input_device=self.input_device,
            vad=self.vad,
            max_silence_ms=self.max_silence_ms,
            min_speech_ms=self.min_speech_ms,
            vad_mode=self.vad_mode,
            format="json",
            output_file=None,
            clipboard=self.clipboard,
            debug=self.debug,
            log_file=self.log_file,
            list_devices=False,
        )

    def _ensure_runtime_loaded(self) -> None:
        if self._e2e_mode or self._runtime_ready.is_set():
            return
        self._append_diagnostic("Starting...")
        try:
            runtime_module, pyaudio_module, pyperclip_module, torch_module = self._runtime_loader(self.backend, self.debug)
        except TypeError:
            runtime_module, pyaudio_module, pyperclip_module, torch_module = self._runtime_loader(self.debug)
        self._runtime_module = runtime_module
        self._pyaudio_module = pyaudio_module
        self._pyperclip_module = pyperclip_module
        self._torch_module = torch_module
        self._runtime_ready.set()

    def _ensure_model_loaded(self, config: DictationConfig) -> None:
        if self._e2e_mode or self._model_loaded:
            return
        if self._runtime_module is None or self._torch_module is None:
            raise RuntimeError("Bridge runtime dependencies are not loaded")
        model, _use_cuda, _load_start, _load_end = self._model_loader(
            config,
            self._runtime_module,
            self._torch_module,
            status_stream=self._diagnostic_stream,
        )
        self._model = model
        self._model_loaded = True
        self._notify_status()

    def _warmup_worker(self) -> None:
        try:
            config = self._config()
            self._ensure_runtime_loaded()
            if self._shutdown_requested.is_set():
                return
            self._ensure_model_loaded(config)
            if self._shutdown_requested.is_set() or self._model is None:
                return

            if self.backend == "whisper":
                from local_ai_dictation.whisper import warmup as warmup_model
            elif self.backend == "parakeet":
                from local_ai_dictation.model import warmup as warmup_model
            else:
                return

            self._append_diagnostic("Warming model")
            warmup_model(self._model)
        except Exception as exc:  # pragma: no cover - defensive warmup guard
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._model_loading = False
            self._notify_status()

    def start_model_warmup(self) -> None:
        with self._lock:
            if self._e2e_mode or self._model_loaded or self._model_loading:
                return
            self._model_loading = True
            self._last_error = None
            self._warmup_thread = threading.Thread(target=self._warmup_worker, daemon=True)
            self._warmup_thread.start()
        self._notify_status()

    def _complete_session(self, transcription: TranscriptionResult) -> None:
        payload = json.loads(render_transcription(transcription, "json"))
        completed_at = time.time()
        history_item = {
            "id": f"tx-{int(completed_at * 1000)}",
            "completed_at": completed_at,
            "payload": payload,
        }
        with self._lock:
            self._last_transcript = payload
            self._last_completed_at = completed_at
            self._last_error = None
            self._state = "idle"
            self._started_at = None
            self._history.append(history_item)
        self._notify_status()

    def _complete_cancelled_before_recording(self) -> None:
        should_notify = False
        with self._lock:
            if self._last_transcript and self._last_transcript.get("metadata", {}).get("cancelled_before_recording"):
                self._state = "idle"
                self._started_at = None
                self._last_error = None
                should_notify = True
        if should_notify:
            self._notify_status()
            return
        self._complete_session(
            TranscriptionResult(
                text="",
                metadata={"cancelled_before_recording": True},
            )
        )

    def _copy_to_clipboard(self, transcription: TranscriptionResult) -> bool:
        if not self.clipboard or self._pyperclip_module is None:
            return False
        warning = copy_transcript_to_clipboard(transcription, self._pyperclip_module)
        if warning is not None:
            self._append_diagnostic(f"Clipboard warning: {warning}")
            return False
        with self._lock:
            self._clipboard_copied_at = time.time()
        return True

    def _dispatch_paste(self, transcription: TranscriptionResult) -> bool:
        if self._paste_notifier is None or not transcription.text.strip():
            return False
        try:
            dispatched = self._paste_notifier()
        except Exception:
            return False
        if not dispatched:
            return False
        with self._lock:
            self._paste_dispatched_at = time.time()
        return True

    def _wait_for_e2e_delay(self, delay_ms: int) -> bool:
        if delay_ms <= 0:
            return not (self._stop_requested.is_set() or self._shutdown_requested.is_set())
        deadline = time.monotonic() + (delay_ms / 1000.0)
        while time.monotonic() < deadline:
            if self._stop_requested.is_set() or self._shutdown_requested.is_set():
                return False
            time.sleep(0.01)
        return not (self._stop_requested.is_set() or self._shutdown_requested.is_set())

    def _run_e2e_session(self) -> None:
        if not self._wait_for_e2e_delay(self._e2e_start_delay_ms):
            self._complete_cancelled_before_recording()
            return

        self._append_diagnostic("🎤 Recording...")
        with self._lock:
            self._state = "recording"
        self._notify_status()

        while not self._stop_requested.is_set() and not self._shutdown_requested.is_set():
            time.sleep(0.01)

        if self._shutdown_requested.is_set():
            with self._lock:
                self._state = "stopped"
                self._started_at = None
            self._notify_status()
            return

        with self._lock:
            self._state = "transcribing"
        self._notify_status()
        self._append_diagnostic("🤖 Generating...")

        if not self._wait_for_e2e_delay(self._e2e_stop_delay_ms):
            if self._shutdown_requested.is_set():
                with self._lock:
                    self._state = "stopped"
                    self._started_at = None
                self._notify_status()
                return

        if self.clipboard:
            with self._lock:
                self._clipboard_copied_at = time.time()

        self._complete_session(
            TranscriptionResult(
                text=self._e2e_transcript,
                device="e2e",
                metadata={"e2e_mode": True},
            )
        )

    def _session_worker(self) -> None:
        config = self._config()
        temp_path: str | None = None
        streaming_session: Any | None = None
        try:
            if self._e2e_mode:
                self._run_e2e_session()
                return

            self._ensure_runtime_loaded()
            if self._stop_requested.is_set() or self._shutdown_requested.is_set():
                self._complete_cancelled_before_recording()
                return

            self._ensure_model_loaded(config)
            if self._stop_requested.is_set() or self._shutdown_requested.is_set():
                self._complete_cancelled_before_recording()
                return

            if self._pyaudio_module is None:
                raise RuntimeError("PyAudio runtime is unavailable")

            if self._model is None:
                raise RuntimeError("Model is not loaded")

            start_streaming = getattr(self._model, "start_streaming", None)
            if config.backend in {"whisper", "willow"} and callable(start_streaming):
                capture_sample_rate = 16000
                streaming_session = start_streaming()
                with self._lock:
                    self._active_streaming_session = streaming_session
                if self._stop_requested.is_set() or self._shutdown_requested.is_set():
                    streaming_session.cancel()
                    self._complete_cancelled_before_recording()
                    return
                try:
                    streaming_session.wait_ready()
                except ModelError:
                    if self._stop_requested.is_set() or self._shutdown_requested.is_set():
                        self._complete_cancelled_before_recording()
                        return
                    raise
                if self._stop_requested.is_set() or self._shutdown_requested.is_set():
                    streaming_session.cancel()
                    self._complete_cancelled_before_recording()
                    return
                on_audio_chunk = streaming_session.send_audio
            else:
                capture_sample_rate = (
                    16000
                    if self.vad
                    else resolve_input_sample_rate(self.input_device, self._pyaudio_module)
                )
                on_audio_chunk = None

            with self._lock:
                self._state = "recording"
            self._notify_status()

            audio_data = self._recorder(
                config,
                self._pyaudio_module,
                sample_rate=capture_sample_rate,
                stop_requested=lambda: self._stop_requested.is_set() or self._shutdown_requested.is_set(),
                status_stream=self._diagnostic_stream,
                **({"on_audio_chunk": on_audio_chunk} if on_audio_chunk is not None else {}),
            )

            if self._shutdown_requested.is_set():
                with self._lock:
                    self._state = "stopped"
                    self._started_at = None
                self._notify_status()
                return

            if not audio_data:
                self._complete_cancelled_before_recording()
                return

            if self._retain_audio and len(audio_data) % 2 == 0:
                self._save_diagnostic_audio(
                    audio_data,
                    self._diagnostic_audio_dir / DIAGNOSTIC_RAW_FILENAME,
                    capture_sample_rate,
                )

            with self._lock:
                self._state = "transcribing"
            self._notify_status()

            infer_audio_data = (
                resample_pcm16_mono(audio_data, capture_sample_rate, 16000)
                if len(audio_data) % 2 == 0
                else audio_data
            )
            if self._retain_audio and len(infer_audio_data) % 2 == 0:
                self._save_diagnostic_audio(
                    infer_audio_data,
                    self._diagnostic_audio_dir / DIAGNOSTIC_MODEL_INPUT_FILENAME,
                    16000,
                )

            if streaming_session is not None:
                if has_probable_speech(infer_audio_data, 16000, vad_mode=config.vad_mode):
                    self._append_diagnostic("🤖 Generating...")
                    transcript_text = streaming_session.finish()
                    streaming_session = None
                    transcription = TranscriptionResult(
                        text=transcript_text,
                        device=(
                            "cloud"
                            if config.backend == "willow"
                            else getattr(self._model, "_parakeet_device", None)
                        ),
                        metadata={"backend": config.backend},
                    )
                else:
                    streaming_session.cancel()
                    self._wait_for_streaming_session(streaming_session)
                    streaming_session = None
                    transcription = TranscriptionResult(
                        text="",
                        metadata={"backend": config.backend, "silence_filtered": True},
                    )
            else:
                transcription, temp_path, _infer_start, _infer_end = self._transcriber(
                    config,
                    self._model,
                    infer_audio_data,
                    16000,
                    status_stream=self._diagnostic_stream,
                )
            if self._copy_to_clipboard(transcription):
                self._dispatch_paste(transcription)
            self._complete_session(transcription)
        except ModelError as exc:
            with self._lock:
                self._last_error = str(exc)
                self._state = "error"
                self._started_at = None
            self._notify_status()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            with self._lock:
                self._last_error = str(exc)
                self._state = "error"
                self._started_at = None
            self._notify_status()
        finally:
            if streaming_session is not None:
                streaming_session.cancel()
                self._wait_for_streaming_session(streaming_session)
            with self._lock:
                self._active_streaming_session = None
            if temp_path:
                try:
                    import os
                    os.unlink(temp_path)
                except Exception:
                    pass
            self._session_finished.set()

    @staticmethod
    def _wait_for_streaming_session(streaming_session: Any) -> None:
        wait_finished = getattr(streaming_session, "wait_finished", None)
        if callable(wait_finished):
            wait_finished()

    @staticmethod
    def _save_diagnostic_audio(audio_data: bytes, path: Path, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        save_audio(audio_data, str(path), sample_rate=sample_rate)
        os.chmod(path, 0o600)

    def shutdown(self) -> None:
        self._shutdown_requested.set()
        self._stop_requested.set()
        with self._lock:
            streaming_session = self._active_streaming_session
        if streaming_session is not None:
            streaming_session.cancel()
        thread = self._session_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._state = "stopped"
            self._started_at = None
        self._notify_status()

    def start_session(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"starting", "recording", "transcribing"}:
                raise BridgeStateError(f"Cannot start while session is {self._state}")
            if self._model_loading:
                raise BridgeStateError("Model is still loading or warming")
            if self._session_thread is not None and self._session_thread.is_alive():
                raise BridgeStateError("Cannot start while the previous session is still winding down")
            self._stop_requested.clear()
            self._session_finished.clear()
            self._state = "starting"
            self._started_at = time.time()
            self._recording_started_at = None
            self._clipboard_copied_at = None
            self._paste_dispatched_at = None
            self._last_error = None
            self._session_thread = threading.Thread(target=self._session_worker, daemon=True)
            self._session_thread.start()
            payload = self.get_session_payload()
        self._notify_status()
        return payload

    def stop_session(self) -> dict[str, Any]:
        streaming_session: Any | None = None
        with self._lock:
            if self._state not in {"starting", "recording"}:
                raise BridgeStateError(f"Cannot stop while session is {self._state}")
            current_state = self._state
            self._stop_requested.set()
            if current_state == "starting" and not self._model_loaded:
                self._complete_cancelled_before_recording()
                return self.get_session_payload()
            if current_state == "starting":
                streaming_session = self._active_streaming_session
        if streaming_session is not None:
            streaming_session.cancel()

        thread = self._session_thread
        if thread is not None:
            thread.join(timeout=min(self._transcript_timeout, 30.0))
        if not self._session_finished.is_set():
            with self._lock:
                streaming_session = self._active_streaming_session
            if streaming_session is not None:
                streaming_session.cancel()
            if thread is not None:
                thread.join(timeout=1.0)
        if not self._session_finished.is_set():
            with self._lock:
                self._state = "error"
                self._last_error = "Timed out waiting for session to stop"
            self._notify_status()
            raise TimeoutError(self._last_error)
        return self.get_session_payload()

    def toggle_session(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
        if state in {"starting", "recording"}:
            return self.stop_session()
        if state == "transcribing":
            raise BridgeStateError("Session is still transcribing")
        return self.start_session()

    def clear_history(self) -> dict[str, Any]:
        with self._lock:
            self._history.clear()
            self._last_transcript = None
            self._last_completed_at = None
            payload = self.get_session_payload()
        self._notify_status()
        return payload

    def _health_payload_locked(self) -> dict[str, Any]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "ok": True,
            "bridge": {
                "backend": "local-ai-dictation-bridge",
                "model_backend": self.backend,
                "model_loaded": self._model_loaded,
                "model_loading": self._model_loading,
                "e2e_mode": self._e2e_mode,
            },
            "session": self.get_session_payload(),
        }

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._health_payload_locked()

    def wait_for_status_update(self, after_version: int | None, *, timeout: float = 15.0) -> tuple[int, dict[str, Any] | None]:
        with self._lock:
            if after_version is None:
                return self._event_version, self._health_payload_locked()
            updated = self._event_condition.wait_for(
                lambda: self._event_version != after_version or self._shutdown_requested.is_set(),
                timeout=timeout,
            )
            if not updated or self._shutdown_requested.is_set():
                return self._event_version, None
            return self._event_version, self._health_payload_locked()

    def get_session_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": BRIDGE_SCHEMA_VERSION,
                "state": self._state,
                "started_at": self._started_at,
                "recording_started_at": self._recording_started_at,
                "last_completed_at": self._last_completed_at,
                "clipboard_copied_at": self._clipboard_copied_at,
                "paste_dispatched_at": self._paste_dispatched_at,
                "last_transcript": self._last_transcript,
                "last_error": self._last_error,
                "model_loaded": self._model_loaded,
                "model_loading": self._model_loading,
                "history": list(self._history),
                "config": {
                    "backend": self.backend,
                    "cpu": self.cpu,
                    "input_device": self.input_device,
                    "vad": self.vad,
                    "max_silence_ms": self.max_silence_ms,
                    "min_speech_ms": self.min_speech_ms,
                    "vad_mode": self.vad_mode,
                    "debug": self.debug,
                    "clipboard": self.clipboard,
                    "retain_audio": self._retain_audio,
                },
                "stderr_tail": list(self._stderr_tail[-20:]),
            }


class _BridgeHandler(BaseHTTPRequestHandler):
    controller: DictationBridgeController

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler interface
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._write_json(200, self.controller.health_payload())
            return
        if parsed.path == "/events":
            self._write_events()
            return
        if parsed.path == "/session":
            self._write_json(200, self.controller.get_session_payload())
            return
        if parsed.path == "/devices":
            devices = list_input_devices()
            self._write_json(
                200,
                {
                    "schema_version": 1,
                    "devices": [asdict(device) for device in devices],
                },
            )
            return
        if parsed.path == "/doctor":
            report = collect_doctor_report(
                check_model_cache=_truthy_query(query.get("check_model_cache")),
                backend=self.controller.backend,
            )
            self._write_json(200, asdict(report))
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler interface
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/session/start":
                self._write_json(200, self.controller.start_session())
                return
            if parsed.path == "/session/stop":
                self._write_json(200, self.controller.stop_session())
                return
            if parsed.path == "/session/toggle":
                self._write_json(200, self.controller.toggle_session())
                return
            if parsed.path == "/session/clear-history":
                self._write_json(200, self.controller.clear_history())
                return
        except BridgeStateError as exc:
            self._write_json(409, {"error": "invalid_state", "detail": str(exc)})
            return
        except TimeoutError as exc:
            self._write_json(504, {"error": "timeout", "detail": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive transport layer
            self._write_json(500, {"error": "internal_error", "detail": str(exc)})
            return

        self._write_json(404, {"error": "not_found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        version, payload = self.controller.wait_for_status_update(None)
        if payload is not None:
            self._write_sse("session", payload)

        while True:
            version, payload = self.controller.wait_for_status_update(version, timeout=15.0)
            if payload is None:
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                continue
            try:
                self._write_sse("session", payload)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _write_sse(self, event: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(b"data: ")
        self.wfile.write(encoded)
        self.wfile.write(b"\n\n")
        self.wfile.flush()


def _truthy_query(values: list[str] | None) -> bool:
    if not values:
        return False
    return values[0].strip().lower() in {"1", "true", "yes", "on"}


def _env_truthy(name: str, env: Mapping[str, str]) -> bool:
    value = env.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, env: Mapping[str, str], default: int) -> int:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    return int(value)



def _parse_input_device(value: Any) -> int | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def make_bridge_server(
    host: str,
    port: int,
    *,
    controller: DictationBridgeController,
) -> ThreadingHTTPServer:
    handler = type("LocalAIDictationBridgeHandler", (_BridgeHandler,), {"controller": controller})
    return ThreadingHTTPServer((host, port), handler)


def build_bridge_controller_from_namespace(
    namespace: Any,
    *,
    env: Mapping[str, str] | None = None,
) -> DictationBridgeController:
    source_env = os.environ if env is None else env
    backend = getattr(namespace, "backend", None)
    return DictationBridgeController(
        backend=get_backend(source_env) if backend in {None, ""} else str(backend),
        cpu=bool(getattr(namespace, "cpu", False)),
        input_device=_parse_input_device(getattr(namespace, "input_device", None)),
        vad=bool(getattr(namespace, "vad", False)),
        max_silence_ms=int(getattr(namespace, "max_silence_ms", 1200)),
        min_speech_ms=int(getattr(namespace, "min_speech_ms", 300)),
        vad_mode=int(getattr(namespace, "vad_mode", 2)),
        debug=bool(getattr(namespace, "debug", False)),
        log_file=str(getattr(namespace, "log_file", "transcriber.debug.log")),
        clipboard=bool(getattr(namespace, "clipboard", True)),
        e2e_mode=_env_truthy("LOCAL_AI_DICTATION_E2E_MODE", source_env),
        e2e_transcript=source_env.get("LOCAL_AI_DICTATION_E2E_TRANSCRIPT", DEFAULT_E2E_TRANSCRIPT),
        e2e_start_delay_ms=_env_int("LOCAL_AI_DICTATION_E2E_START_DELAY_MS", source_env, 0),
        e2e_stop_delay_ms=_env_int("LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS", source_env, 0),
        status_notifier=_build_status_notifier(source_env),
        paste_notifier=_build_hyprland_paste_notifier(source_env),
        retain_audio=_env_truthy("LOCAL_AI_DICTATION_RETAIN_AUDIO", source_env),
        diagnostic_audio_dir=(
            Path(source_env.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
            / "local-ai-dictation"
            / "diagnostic-audio"
        ),
    )


def run_bridge_server(namespace: Any) -> int:
    host = str(getattr(namespace, "host", "127.0.0.1"))
    port = int(getattr(namespace, "port", 8765))
    controller = build_bridge_controller_from_namespace(namespace)
    controller.start_model_warmup()
    server = make_bridge_server(host, port, controller=controller)

    shutdown_requested = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_shutdown(signum: int, frame: Any) -> None:
        shutdown_requested.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    print(f"Local AI Dictation bridge listening on http://{host}:{port} ({controller.backend})")
    print("Run the Windows app, then use its button or hotkey to start/stop recording.")

    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if not shutdown_requested.is_set():
            server.shutdown()
        server.server_close()
        controller.shutdown()
    return 0
