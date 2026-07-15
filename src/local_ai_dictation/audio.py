"""Audio device enumeration helpers and VAD capture utilities for Parakeet."""

from __future__ import annotations

import audioop
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

from local_ai_dictation.types import AudioDevice, VadBackend


PyAudioModule = Any
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2


class WebRtcVadBackend:
    """Small adapter around a WebRTC-compatible VAD implementation."""

    name = "webrtc"

    def __init__(self, mode: int = 2, *, webrtcvad_module: Any | None = None) -> None:
        if webrtcvad_module is None:
            import webrtcvad as webrtcvad_module

        self._vad = webrtcvad_module.Vad(mode)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return bool(self._vad.is_speech(frame, sample_rate))


def build_vad_backend(mode: int = 2) -> VadBackend:
    return WebRtcVadBackend(mode)


def _load_pyaudio_module(pyaudio_module: PyAudioModule | None = None) -> PyAudioModule:
    if pyaudio_module is not None:
        return pyaudio_module

    import pyaudio

    return pyaudio


def _default_input_device_id(pa: Any) -> int | None:
    try:
        info = pa.get_default_input_device_info()
    except Exception:
        return None

    try:
        return int(info.get("index"))
    except Exception:
        return None


def _host_api_name(pa: Any, info: dict[str, Any]) -> str:
    try:
        host_api_index = int(info.get("hostApi", -1))
        host_api_info = pa.get_host_api_info_by_index(host_api_index)
        return str(host_api_info.get("name", "unknown"))
    except Exception:
        return "unknown"


def list_input_devices(pyaudio_module: PyAudioModule | None = None) -> list[AudioDevice]:
    pyaudio_module = _load_pyaudio_module(pyaudio_module)
    pa = pyaudio_module.PyAudio()
    try:
        default_input_id = _default_input_device_id(pa)
        devices: list[AudioDevice] = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            max_input_channels = int(info.get("maxInputChannels", 0))
            if max_input_channels <= 0:
                continue

            device_id = int(info.get("index", index))
            devices.append(
                AudioDevice(
                    id=device_id,
                    name=str(info.get("name", "unknown")),
                    default_sample_rate=int(info.get("defaultSampleRate", 0)),
                    max_input_channels=max_input_channels,
                    host_api=_host_api_name(pa, info),
                    is_default_candidate=(default_input_id is not None and device_id == default_input_id),
                )
            )

        return sorted(devices, key=lambda device: device.id)
    finally:
        pa.terminate()


def _preferred_linux_input_device_id(devices: list[AudioDevice]) -> int | None:
    for device in devices:
        if device.name.strip().lower() == "pipewire":
            return device.id
    return None


def resolve_input_device_id(
    input_device: int | str | None,
    pyaudio_module: PyAudioModule | None = None,
) -> int | None:
    if input_device is None:
        if os.name == "posix":
            try:
                return _preferred_linux_input_device_id(list_input_devices(pyaudio_module))
            except Exception:
                return None
        return None
    if isinstance(input_device, int):
        return input_device

    requested_name = str(input_device).strip()
    if not requested_name:
        return None

    for device in list_input_devices(pyaudio_module):
        if device.name == requested_name:
            return device.id

    raise ValueError(f"No input device named {requested_name!r}")


def pulse_default_source_spec() -> tuple[int, int] | None:
    pactl_path = shutil.which("pactl")
    if pactl_path is None:
        return None

    try:
        default_source = subprocess.run(
            [pactl_path, "get-default-source"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not default_source:
            return None

        sources = subprocess.run(
            [pactl_path, "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except Exception:
        return None

    for line in sources:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] != default_source:
            continue
        spec = parts[3].split()
        if len(spec) < 3:
            continue
        try:
            channels = int(spec[1].removesuffix("ch"))
            sample_rate = int(spec[2].removesuffix("Hz"))
        except ValueError:
            return None
        return sample_rate, channels

    return None


def resolve_input_sample_rate(
    input_device: int | str | None,
    pyaudio_module: PyAudioModule | None = None,
    *,
    fallback: int = VAD_SAMPLE_RATE,
) -> int:
    if input_device is None and os.name == "posix":
        pulse_spec = pulse_default_source_spec()
        if pulse_spec is not None:
            return pulse_spec[0]

    try:
        devices = list_input_devices(pyaudio_module)
    except Exception:
        return fallback

    resolved_device_id = resolve_input_device_id(input_device, pyaudio_module)
    if resolved_device_id is None:
        for device in devices:
            if device.is_default_candidate and device.default_sample_rate > 0:
                return int(device.default_sample_rate)
        return fallback

    for device in devices:
        if device.id == resolved_device_id and device.default_sample_rate > 0:
            return int(device.default_sample_rate)

    return fallback


def fallback_input_device_id(devices: list[AudioDevice]) -> int | None:
    if not devices:
        return None

    for device in devices:
        if device.name.strip().lower() not in {"default", "sysdefault"}:
            return device.id

    return devices[0].id


_CONNECTION_FAILURE_MARKERS = (
    "connection refused",
    "connection failure",
    "failed to connect",
    "timed out",
    "timeout",
)


def _classify_probe_failure(output: str) -> str:
    lowered = output.lower()
    if any(marker in lowered for marker in _CONNECTION_FAILURE_MARKERS):
        return "unreachable"
    return "unknown"


def _normalize_vad_frame(frame: bytes) -> bytes:
    if len(frame) == VAD_FRAME_BYTES:
        return frame
    if len(frame) > VAD_FRAME_BYTES:
        return frame[:VAD_FRAME_BYTES]
    return frame.ljust(VAD_FRAME_BYTES, b"\x00")


def downmix_pcm16_to_mono(audio_data: bytes, channels: int) -> bytes:
    if channels <= 1 or not audio_data:
        return audio_data
    if channels != 2:
        raise ValueError(f"Unsupported channel count for downmix: {channels}")
    return audioop.tomono(audio_data, 2, 0.5, 0.5)


class Pcm16StreamConverter:
    """Incrementally converts native PCM16 capture into mono model-rate PCM."""

    def __init__(self, input_sample_rate: int, input_channels: int, output_sample_rate: int) -> None:
        if input_channels not in {1, 2}:
            raise ValueError(f"Unsupported channel count for downmix: {input_channels}")
        self._input_sample_rate = input_sample_rate
        self._input_channels = input_channels
        self._output_sample_rate = output_sample_rate
        self._pending = bytearray()
        self._rate_state: Any | None = None

    def convert(self, audio_data: bytes) -> bytes:
        if not audio_data:
            return b""
        self._pending.extend(audio_data)
        frame_bytes = 2 * self._input_channels
        complete_bytes = len(self._pending) - (len(self._pending) % frame_bytes)
        if complete_bytes == 0:
            return b""

        complete = bytes(self._pending[:complete_bytes])
        del self._pending[:complete_bytes]
        mono = downmix_pcm16_to_mono(complete, self._input_channels)
        if self._input_sample_rate == self._output_sample_rate:
            return mono
        converted, self._rate_state = audioop.ratecv(
            mono,
            2,
            1,
            self._input_sample_rate,
            self._output_sample_rate,
            self._rate_state,
        )
        return converted

    def finish(self) -> None:
        if self._pending:
            raise ValueError("PCM stream ended with an incomplete sample frame")


def resample_pcm16_mono(
    audio_data: bytes,
    input_sample_rate: int,
    output_sample_rate: int = VAD_SAMPLE_RATE,
) -> bytes:
    if not audio_data or input_sample_rate == output_sample_rate:
        return audio_data
    converted, _state = audioop.ratecv(audio_data, 2, 1, input_sample_rate, output_sample_rate, None)
    return converted


def has_probable_speech(
    audio_data: bytes,
    sample_rate: int,
    *,
    vad_mode: int = 2,
    min_voiced_ms: int = 90,
) -> bool:
    if not audio_data:
        return False

    try:
        vad = build_vad_backend(vad_mode)
    except Exception:
        return True

    prepared_audio = (
        audio_data
        if sample_rate == VAD_SAMPLE_RATE
        else resample_pcm16_mono(audio_data, sample_rate, VAD_SAMPLE_RATE)
    )
    voiced_ms = 0
    required_voiced_ms = max(VAD_FRAME_MS, min_voiced_ms)

    for offset in range(0, len(prepared_audio), VAD_FRAME_BYTES):
        frame = prepared_audio[offset : offset + VAD_FRAME_BYTES]
        if not frame:
            continue
        if not vad.is_speech(_normalize_vad_frame(frame), VAD_SAMPLE_RATE):
            continue
        voiced_ms += VAD_FRAME_MS
        if voiced_ms >= required_voiced_ms:
            return True

    return False


def record_until_vad_stop(
    stream: Any,
    *,
    vad: VadBackend,
    stop_requested: Callable[[], bool],
    min_speech_ms: int,
    max_silence_ms: int,
    sample_rate: int = VAD_SAMPLE_RATE,
    frame_samples: int = VAD_FRAME_SAMPLES,
    on_audio_chunk: Callable[[bytes], None] | None = None,
) -> bytes:
    if sample_rate != VAD_SAMPLE_RATE:
        raise ValueError(f"VAD capture requires {VAD_SAMPLE_RATE} Hz audio")

    frames: list[bytes] = []
    cumulative_voiced_ms = 0
    consecutive_silence_ms = 0
    speech_detected = False

    while True:
        if stop_requested():
            break

        data = stream.read(frame_samples, exception_on_overflow=False)
        if not data:
            continue

        frames.append(data)
        if on_audio_chunk is not None:
            on_audio_chunk(data)
        is_speech = vad.is_speech(_normalize_vad_frame(data), sample_rate)
        if is_speech:
            speech_detected = True
            cumulative_voiced_ms += VAD_FRAME_MS
            consecutive_silence_ms = 0
            continue

        if not speech_detected or cumulative_voiced_ms < min_speech_ms:
            consecutive_silence_ms = 0
            continue

        consecutive_silence_ms += VAD_FRAME_MS
        if consecutive_silence_ms >= max_silence_ms:
            break

    return b"".join(frames)


def probe_audio_backend(
    *,
    env: Mapping[str, str] | None = None,
    pactl_timeout: float = 1.5,
    wslg_socket_path: Path = Path("/mnt/wslg/PulseServer"),
) -> dict[str, str]:
    source = os.environ if env is None else env
    pulse_server = source.get("PULSE_SERVER")
    has_wslg_socket = wslg_socket_path.exists()

    if pulse_server and pulse_server.startswith("tcp:"):
        transport = "tcp"
    elif (pulse_server and pulse_server.startswith("unix:")) or has_wslg_socket:
        transport = "unix"
    elif pulse_server:
        transport = "unknown"
    else:
        transport = "none"

    pactl_path = shutil.which("pactl")
    if pactl_path is None:
        return {
            "status": "binary_missing",
            "transport": transport,
            "detail": "pactl binary is not installed",
        }

    if not pulse_server and not has_wslg_socket:
        return {
            "status": "not_configured",
            "transport": "none",
            "detail": "PULSE_SERVER is unset and the WSLg Pulse socket is unavailable",
        }

    if transport == "unknown":
        return {
            "status": "unknown",
            "transport": "unknown",
            "detail": f"Unsupported PULSE_SERVER transport: {pulse_server}",
        }

    probe_env = dict(source)
    if not pulse_server and has_wslg_socket:
        probe_env["PULSE_SERVER"] = f"unix:{wslg_socket_path}"

    try:
        completed = subprocess.run(
            [pactl_path, "info"],
            capture_output=True,
            env=probe_env,
            text=True,
            timeout=pactl_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "unreachable",
            "transport": transport,
            "detail": f"pactl info timed out after {pactl_timeout:.1f}s",
        }
    except OSError as exc:
        return {
            "status": "unknown",
            "transport": transport,
            "detail": str(exc),
        }

    if completed.returncode == 0:
        return {
            "status": "reachable",
            "transport": transport,
            "detail": "pactl info succeeded",
        }

    combined_output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    status = _classify_probe_failure(combined_output)
    return {
        "status": status,
        "transport": transport,
        "detail": combined_output or f"pactl info exited with status {completed.returncode}",
    }
