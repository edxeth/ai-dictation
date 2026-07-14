#!/usr/bin/env python3
"""Interactive dictation flow for Local AI Dictation."""

from __future__ import annotations

import argparse
import atexit
import importlib
import io
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from contextlib import nullcontext
from typing import Any, Callable, cast

from local_ai_dictation.audio import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    build_vad_backend,
    downmix_pcm16_to_mono,
    fallback_input_device_id,
    has_probable_speech,
    list_input_devices,
    pulse_default_source_spec,
    record_until_vad_stop,
    resample_pcm16_mono,
    resolve_input_device_id,
    resolve_input_sample_rate,
)
from local_ai_dictation.config import resolve_config
from local_ai_dictation.errors import (
    AUDIO_BACKEND_UNREACHABLE,
    MODEL_IMPORT_FAILED,
    MODEL_TRANSCRIBE_FAILED,
    AudioError,
    ExitCode,
    ModelError,
)
from local_ai_dictation.output import emit_transcription_result
from local_ai_dictation.types import DictationConfig, TranscriptionEngine, TranscriptionResult


_shutdown_event = threading.Event()
_bridge_stop_event = threading.Event()
_old_stdout = None
_stderr_fd = None
_devnull_fd = None


def _cleanup_handler() -> None:
    _shutdown_event.set()


atexit.register(_cleanup_handler)


def _signal_handler(signum, frame) -> None:  # pragma: no cover - signal plumbing
    if hasattr(signal, "SIGUSR1") and signum == signal.SIGUSR1:
        _bridge_stop_event.set()
        return
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, _signal_handler)


def _no_kbi_traceback(exc_type, exc, tb) -> None:
    if exc_type is KeyboardInterrupt:
        return
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _no_kbi_traceback


HELP_DESC = "Local AI Dictation with selectable Whisper, Parakeet, or Willow backends."
HELP_EPILOG = """Examples:
  local-ai-dictation dictation
  local-ai-dictation dictation --backend whisper
  local-ai-dictation dictation --debug
  local-ai-dictation dictation --cpu
  local-ai-dictation dictation --list-devices
  local-ai-dictation dictation --input-device 2
"""


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--backend",
        choices=["parakeet", "whisper", "willow"],
        default=None,
        help="Select the transcription backend",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=None,
        help="Verbose diagnostics: device, timings, GPU memory (logs to file unless --log-file=-)",
    )
    parser.add_argument("--cpu", action="store_true", default=None, help="Force CPU inference")
    parser.add_argument(
        "--vad",
        action="store_true",
        default=None,
        help="Enable VAD-driven auto-stop when supported by the runtime",
    )
    parser.add_argument(
        "--max-silence-ms",
        type=int,
        default=None,
        help="Silence duration required before VAD auto-stop becomes eligible",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=None,
        help="Minimum cumulative voiced duration before VAD stop can trigger",
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="WebRTC-VAD aggressiveness",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default=None,
        help="Transcript output format",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional transcript output file path",
    )
    parser.add_argument(
        "--clipboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable clipboard copy",
    )
    parser.add_argument(
        "--log-file",
        default="transcriber.debug.log",
        help="Debug log file (use '-' for stderr)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio input devices and exit"
    )
    parser.add_argument(
        "--input-device",
        type=str,
        default=None,
        help="PyAudio input device index or exact device name",
    )
    parser.add_argument(
        "--bridge-mode",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-ai-dictation dictation",
        description=HELP_DESC,
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    return add_cli_arguments(parser)


def configure_logging(config: DictationConfig, *, status_stream=None) -> None:
    if config.debug:
        if config.log_file != "-":
            try:
                if os.path.exists(config.log_file):
                    os.remove(config.log_file)
            except Exception:
                pass
            logging.basicConfig(
                filename=config.log_file,
                filemode="w",
                level=logging.DEBUG,
                force=True,
            )
            print(f"Debug logs -> {config.log_file}", file=status_stream or sys.stdout)
        else:
            logging.basicConfig(level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)


def redirect_library_loggers_to_root_file() -> None:
    names = [
        "",
        "nemo_logger",
        "urllib3",
        "datasets",
        "matplotlib",
        "graphviz",
        "huggingface_hub",
        "transformers",
    ]
    for name in names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.DEBUG)


def _silence_start() -> None:
    global _old_stdout, _stderr_fd, _devnull_fd
    _old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    _stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)


def _silence_stop() -> None:
    global _old_stdout, _stderr_fd, _devnull_fd
    if _old_stdout is not None:
        sys.stdout = _old_stdout
        _old_stdout = None
    if _stderr_fd is not None:
        os.dup2(_stderr_fd, 2)
        os.close(_stderr_fd)
        _stderr_fd = None
    if _devnull_fd is not None:
        os.close(_devnull_fd)
        _devnull_fd = None


class SilentSTDERR:
    def __enter__(self):
        self.old_stderr = os.dup(2)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 2)
        return self

    def __exit__(self, *args):
        os.dup2(self.old_stderr, 2)
        os.close(self.old_stderr)
        os.close(self.devnull)


class _silence_context:
    def __enter__(self):
        _silence_start()
        return self

    def __exit__(self, *exc):
        _silence_stop()
        return False


def _status_stream(config: DictationConfig):
    return sys.stderr if config.format == "json" else sys.stdout


def spinner_animation(stop_event: threading.Event, prefix: str, stream=None) -> None:
    stream = stream or sys.stdout
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    try:
        while not stop_event.is_set() and not _shutdown_event.is_set():
            stream.write(f"\r{prefix} {chars[idx % len(chars)]}")
            stream.flush()
            idx += 1
            time.sleep(0.1)
    finally:
        stream.write("\033[2K\r")
        stream.flush()


def wait_for_enter_interruptible(show_prompt: bool = False, *, stream=None) -> bool:
    prompt_stream = stream or sys.stdout
    if show_prompt:
        prompt_stream.write("Press ENTER to start recording (Ctrl+C to exit)...\n")
        prompt_stream.flush()
    while not _shutdown_event.is_set():
        readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        if readable:
            sys.stdin.readline()
            return True
    return False


def _manual_stop_requested(timeout: float = 0.0) -> bool:
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if readable:
        sys.stdin.readline()
        return True
    return False


def save_audio(audio_data: bytes, filename: str, sample_rate: int = 16000) -> None:
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


def _load_runtime_dependencies(backend: str, debug: bool):
    if not debug:
        if backend == "parakeet":
            os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
        _silence_start()
    try:
        if backend == "parakeet":
            runtime_module = importlib.import_module("nemo.collections.asr")
        elif backend == "whisper":
            runtime_module = importlib.import_module("faster_whisper")
        elif backend == "willow":
            runtime_module = importlib.import_module("local_ai_dictation.willow")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        pyaudio = importlib.import_module("pyaudio")
        pyperclip = importlib.import_module("pyperclip")
        torch = importlib.import_module("torch")
    finally:
        if not debug:
            _silence_stop()

    warnings.filterwarnings("ignore")
    return runtime_module, pyaudio, pyperclip, torch


def list_devices(pyaudio_module: Any) -> int:
    try:
        devices = list_input_devices(pyaudio_module)
        print("Input devices:")
        for device in devices:
            default_marker = " default" if device.is_default_candidate else ""
            print(
                f"- id={device.id} name='{device.name}' rate={device.default_sample_rate}Hz host_api={device.host_api}{default_marker}"
            )
        return int(ExitCode.OK)
    except Exception as exc:
        error = AudioError(AUDIO_BACKEND_UNREACHABLE, str(exc))
        print(f"Error listing devices: {error}")
        return int(ExitCode.ERROR)


def _looks_like_missing_default_output_device(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "no default output device" in lowered or "default output device" in lowered


STOP_CAPTURE_DRAIN_SECONDS = 1.0
PAREC_LATENCY_MSEC = 40
PAREC_PROCESS_TIME_MSEC = 20


def _record_audio_with_parec(
    sample_rate: int,
    *,
    stop_requested: Callable[[], bool],
    status_stream=None,
) -> bytes | None:
    parec_path = shutil.which("parec")
    if parec_path is None:
        raise RuntimeError("parec is unavailable")

    default_spec = pulse_default_source_spec()
    capture_sample_rate = default_spec[0] if default_spec is not None else sample_rate
    capture_channels = default_spec[1] if default_spec is not None else 1

    process = subprocess.Popen(
        [
            parec_path,
            "--device=@DEFAULT_SOURCE@",
            "--raw",
            "--format=s16le",
            "--rate",
            str(capture_sample_rate),
            "--channels",
            str(capture_channels),
            f"--latency-msec={PAREC_LATENCY_MSEC}",
            f"--process-time-msec={PAREC_PROCESS_TIME_MSEC}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        process.kill()
        process.wait(timeout=5)
        raise RuntimeError("parec stdout is unavailable")

    chunks: list[bytes] = []

    def _reader() -> None:
        while True:
            data = process.stdout.read(4096)
            if not data:
                return
            chunks.append(data)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    print("🎤 Recording...", file=status_stream, flush=True)

    stop_deadline: float | None = None
    try:
        while not _shutdown_event.is_set():
            if stop_requested():
                if stop_deadline is None:
                    stop_deadline = time.monotonic() + STOP_CAPTURE_DRAIN_SECONDS
                elif time.monotonic() >= stop_deadline:
                    break
            if process.poll() is not None:
                break
            time.sleep(0.01)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        reader_thread.join(timeout=2)
        process.stdout.close()

    audio_data = b"".join(chunks)
    if not audio_data:
        return None
    audio_data = downmix_pcm16_to_mono(audio_data, capture_channels)
    if capture_sample_rate != sample_rate:
        audio_data = resample_pcm16_mono(audio_data, capture_sample_rate, sample_rate)
    return audio_data or None



def record_audio_interruptible(
    config: DictationConfig,
    pyaudio_module: Any,
    sample_rate: int = 16000,
    *,
    stop_requested: Callable[[], bool] | None = None,
    status_stream=None,
) -> bytes | None:
    status_stream = status_stream or _status_stream(config)
    frames_per_buffer = VAD_FRAME_SAMPLES if config.vad else 1024
    if os.name == "posix" and not config.vad and config.input_device is None and shutil.which("parec") is not None:
        try:
            return _record_audio_with_parec(
                sample_rate,
                stop_requested=(stop_requested or (lambda: _shutdown_event.is_set() or _manual_stop_requested())),
                status_stream=status_stream,
            )
        except Exception as exc:
            if config.debug:
                print(f"parec capture fallback: {exc}", file=status_stream)

    try:
        requested_input_device_id = resolve_input_device_id(config.input_device, pyaudio_module)
    except ValueError as exc:
        error = AudioError(AUDIO_BACKEND_UNREACHABLE, str(exc))
        print(f"❌ Audio error: {error}", file=status_stream)
        return None

    ctx = SilentSTDERR() if not config.debug else nullcontext()
    with ctx:
        pa = pyaudio_module.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio_module.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=requested_input_device_id,
                frames_per_buffer=frames_per_buffer,
            )
        except Exception as exc:
            if requested_input_device_id is None and _looks_like_missing_default_output_device(exc):
                try:
                    fallback_devices = list_input_devices(pyaudio_module)
                    fallback_device_id = fallback_input_device_id(fallback_devices)
                    if fallback_device_id is not None:
                        stream = pa.open(
                            format=pyaudio_module.paInt16,
                            channels=1,
                            rate=sample_rate,
                            input=True,
                            input_device_index=fallback_device_id,
                            frames_per_buffer=frames_per_buffer,
                        )
                    else:
                        raise exc
                except Exception:
                    error = AudioError(AUDIO_BACKEND_UNREACHABLE, str(exc))
                    print(f"❌ Audio error: {error}", file=status_stream)
                    try:
                        pa.terminate()
                    except Exception:
                        pass
                    return None
            else:
                error = AudioError(AUDIO_BACKEND_UNREACHABLE, str(exc))
                print(f"❌ Audio error: {error}", file=status_stream)
                try:
                    pa.terminate()
                except Exception:
                    pass
                return None

    audio_data: bytes | None = None
    print("🎤 Recording...", file=status_stream, flush=True)

    if stop_requested is None:
        stop_requested = lambda: _shutdown_event.is_set() or _manual_stop_requested()

    ctx = SilentSTDERR() if not config.debug else nullcontext()
    with ctx:
        try:
            if config.vad:
                try:
                    vad_backend = build_vad_backend(config.vad_mode)
                except Exception as exc:
                    error = AudioError(AUDIO_BACKEND_UNREACHABLE, f"WebRTC VAD unavailable: {exc}")
                    print(f"❌ Audio error: {error}", file=status_stream)
                    return None

                audio_data = record_until_vad_stop(
                    stream,
                    vad=vad_backend,
                    stop_requested=stop_requested,
                    min_speech_ms=config.min_speech_ms,
                    max_silence_ms=config.max_silence_ms,
                    sample_rate=sample_rate,
                    frame_samples=VAD_FRAME_SAMPLES,
                )
            else:
                frames: list[bytes] = []
                stop_deadline: float | None = None
                while not _shutdown_event.is_set():
                    if stop_requested():
                        if stop_deadline is None:
                            stop_deadline = time.monotonic() + STOP_CAPTURE_DRAIN_SECONDS
                        elif time.monotonic() >= stop_deadline:
                            break
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        frames.append(data)
                    except Exception:
                        continue
                audio_data = b"".join(frames)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    return audio_data or None


def _load_model(
    config: DictationConfig,
    runtime_module: Any,
    torch_module: Any,
    *,
    status_stream=None,
) -> tuple[TranscriptionEngine, bool, float, float]:
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(
        target=spinner_animation,
        args=(stop_spinner, "⏳ Loading model"),
        kwargs={"stream": status_stream or _status_stream(config)},
        daemon=True,
    )
    spinner_thread.start()

    load_ctx = nullcontext() if config.debug else _silence_context()
    start = time.perf_counter()
    try:
        with load_ctx:
            if config.backend == "whisper":
                from local_ai_dictation.whisper import WHISPER_MODEL_ID, WhisperEngine

                use_cuda = torch_module.cuda.is_available() and not config.cpu
                device = "cuda" if use_cuda else "cpu"
                compute_type = "float16" if use_cuda else "int8"
                model = WhisperEngine(
                    runtime_module.WhisperModel(WHISPER_MODEL_ID, device=device, compute_type=compute_type),
                    device=device,
                    compute_type=compute_type,
                    model_id=WHISPER_MODEL_ID,
                )
            elif config.backend == "parakeet":
                model = runtime_module.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
                use_cuda = torch_module.cuda.is_available() and not config.cpu
                device = "cuda" if use_cuda else "cpu"
                model.to(device)
                model.eval()
            elif config.backend == "willow":
                model = runtime_module.WillowEngine.from_config(config)
                use_cuda = False
            else:
                raise ValueError(f"Unsupported backend: {config.backend}")
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc
    finally:
        end = time.perf_counter()
        stop_spinner.set()
        spinner_thread.join()

    return model, use_cuda, start, end


def _transcribe_once(
    config: DictationConfig,
    model: TranscriptionEngine,
    audio_data: bytes,
    sample_rate: int,
    *,
    status_stream=None,
) -> tuple[TranscriptionResult, str, float, float]:
    start = time.perf_counter()
    if not has_probable_speech(audio_data, sample_rate, vad_mode=config.vad_mode):
        return (
            TranscriptionResult(text="", metadata={"backend": config.backend, "silence_filtered": True}),
            "",
            start,
            time.perf_counter(),
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        save_audio(audio_data, tmp.name, sample_rate=sample_rate)
        temp_path = tmp.name

    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(
        target=spinner_animation,
        args=(stop_spinner, "🤖 Generating..."),
        kwargs={"stream": status_stream or _status_stream(config)},
        daemon=True,
    )
    spinner_thread.start()

    infer_ctx = nullcontext() if config.debug else _silence_context()
    try:
        with infer_ctx:
            result = model.transcribe([temp_path], verbose=False)
        if isinstance(result, list) and result:
            first = result[0]
            transcript_text = getattr(first, "text", first if isinstance(first, str) else str(first))
        else:
            transcript_text = str(result)
        transcription = TranscriptionResult(text=transcript_text, metadata={"backend": config.backend})
        return transcription, temp_path, start, time.perf_counter()
    except Exception as exc:
        raise ModelError(MODEL_TRANSCRIBE_FAILED, str(exc)) from exc
    finally:
        stop_spinner.set()
        spinner_thread.join()


def _run_bridge_controlled_dictation(
    config: DictationConfig,
    *,
    pyaudio_module: Any,
    pyperclip_module: Any,
    runtime_module: Any,
    torch_module: Any,
) -> int:
    status_stream = _status_stream(config)
    _bridge_stop_event.clear()

    try:
        model, _use_cuda, _load_start, _load_end = _load_model(config, runtime_module, torch_module)
    except ModelError as exc:
        print(f"❌ Error: {exc}", file=status_stream)
        return int(ExitCode.ERROR)

    if _shutdown_event.is_set():
        return int(ExitCode.OK)

    capture_sample_rate = VAD_SAMPLE_RATE if config.vad else resolve_input_sample_rate(config.input_device, pyaudio_module)
    audio_data = record_audio_interruptible(
        config,
        pyaudio_module,
        sample_rate=capture_sample_rate,
        stop_requested=lambda: _shutdown_event.is_set() or _bridge_stop_event.is_set(),
    )

    if _shutdown_event.is_set():
        return int(ExitCode.OK)

    if not audio_data:
        emit_transcription_result(
            TranscriptionResult(text=""),
            config,
            pyperclip_module=pyperclip_module,
            status_stream=status_stream,
        )
        return int(ExitCode.OK)

    try:
        infer_sample_rate = VAD_SAMPLE_RATE
        prepared_audio_data = resample_pcm16_mono(audio_data, capture_sample_rate, infer_sample_rate)
        transcription, temp_path, _infer_start, _infer_end = _transcribe_once(
            config, model, prepared_audio_data, infer_sample_rate
        )
    except ModelError as exc:
        print(f"❌ Error: {exc}", file=status_stream)
        return int(ExitCode.ERROR)

    emit_transcription_result(
        transcription,
        config,
        pyperclip_module=pyperclip_module,
        status_stream=status_stream,
    )

    try:
        os.unlink(temp_path)
    except Exception:
        pass

    return int(ExitCode.OK)


def run_dictation(args: argparse.Namespace) -> int:
    config = resolve_config(args, env=os.environ)
    status_stream = _status_stream(config)
    configure_logging(config, status_stream=status_stream)
    print("Starting...", file=status_stream, flush=True)

    try:
        runtime_module, pyaudio_module, pyperclip_module, torch_module = _load_runtime_dependencies(
            config.backend,
            config.debug,
        )
    except TypeError:
        runtime_module, pyaudio_module, pyperclip_module, torch_module = _load_runtime_dependencies(config.debug)

    if config.list_devices:
        return list_devices(pyaudio_module)

    if config.debug and config.log_file != "-":
        redirect_library_loggers_to_root_file()

    if bool(getattr(args, "bridge_mode", False)):
        return _run_bridge_controlled_dictation(
            config,
            pyaudio_module=pyaudio_module,
            pyperclip_module=pyperclip_module,
            runtime_module=runtime_module,
            torch_module=torch_module,
        )

    try:
        model, use_cuda, load_start, load_end = _load_model(config, runtime_module, torch_module)
    except ModelError as exc:
        print(f"❌ Error: {exc}", file=status_stream)
        return int(ExitCode.ERROR)

    if _shutdown_event.is_set():
        return int(ExitCode.OK)

    if config.backend == "whisper":
        print("🚀 LOCAL AI DICTATION · WHISPER DISTIL LARGE V3.5", file=status_stream)
    elif config.backend == "parakeet":
        print("🚀 LOCAL AI DICTATION · PARAKEET TDT 0.6B V3", file=status_stream)
    else:
        print("☁️ LOCAL AI DICTATION · WILLOW VOICE FRONTIER (SERVER-SELECTED)", file=status_stream)
    if torch_module.cuda.is_available():
        print(f"✅ GPU: {torch_module.cuda.get_device_name(0)}", file=status_stream)
    print("=" * 60, file=status_stream)
    print("📝 Press ENTER to start → Speak → Press ENTER to stop", file=status_stream)
    print("   Ctrl+C to exit", file=status_stream)
    print("=" * 60 + "\n", file=status_stream)

    if config.debug:
        print(f"Backend: {config.backend}", file=status_stream)
        if config.backend == "willow":
            print("Model device: Willow Voice cloud", file=status_stream)
        else:
            model_parameters = cast(Any, model).parameters()
            print(f"Model device: {next(model_parameters).device}", file=status_stream)
        if config.backend == "whisper":
            print(
                f"Compute type: {getattr(model, '_parakeet_compute_type', 'unknown')}",
                file=status_stream,
            )
        elif use_cuda:
            capability = torch_module.cuda.get_device_capability()
            print(f"CUDA capability: {capability[0]}.{capability[1]}", file=status_stream)
            print(
                f"GPU alloc (MiB) after load: {torch_module.cuda.memory_allocated() / 1024**2:.2f}",
                file=status_stream,
            )
        print(f"Model load time: {load_end - load_start:.3f}s", file=status_stream)

    next_wait_shows_prompt = False

    try:
        while not _shutdown_event.is_set():
            if not wait_for_enter_interruptible(
                show_prompt=next_wait_shows_prompt,
                stream=status_stream,
            ):
                break

            capture_sample_rate = VAD_SAMPLE_RATE if config.vad else resolve_input_sample_rate(config.input_device, pyaudio_module)
            record_start = time.perf_counter()
            audio_data = record_audio_interruptible(config, pyaudio_module, sample_rate=capture_sample_rate)
            record_end = time.perf_counter()

            if not audio_data or _shutdown_event.is_set():
                break

            try:
                infer_sample_rate = VAD_SAMPLE_RATE
                prepared_audio_data = resample_pcm16_mono(audio_data, capture_sample_rate, infer_sample_rate)
                transcription, temp_path, infer_start, infer_end = _transcribe_once(
                    config, model, prepared_audio_data, infer_sample_rate
                )
            except ModelError as exc:
                print(f"❌ Error: {exc}", file=status_stream)
                next_wait_shows_prompt = True
                continue

            if _shutdown_event.is_set():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                break

            emit_transcription_result(
                transcription,
                config,
                pyperclip_module=pyperclip_module,
                status_stream=status_stream,
            )

            if config.debug:
                seconds = len(audio_data) / (2 * capture_sample_rate)
                print(
                    f"Audio length: {seconds:.2f}s | Record: {record_end - record_start:.3f}s | Infer: {infer_end - infer_start:.3f}s",
                    file=status_stream,
                )
                if use_cuda and config.backend == "parakeet":
                    print(
                        f"GPU alloc (MiB) after infer: {torch_module.cuda.memory_allocated() / 1024**2:.2f}",
                        file=status_stream,
                    )

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            next_wait_shows_prompt = True

        return int(ExitCode.OK)
    except KeyboardInterrupt:
        return int(ExitCode.OK)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_dictation(namespace)


if __name__ == "__main__":
    raise SystemExit(main())
