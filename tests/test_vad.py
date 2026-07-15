from __future__ import annotations

import importlib
from itertools import repeat
from pathlib import Path
import signal
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


audio_module = importlib.import_module("local_ai_dictation.audio")
dictation_module = importlib.import_module("local_ai_dictation.dictation")
DictationConfig = importlib.import_module("local_ai_dictation.types").DictationConfig

VAD_FRAME_BYTES = audio_module.VAD_FRAME_BYTES
VAD_FRAME_SAMPLES = audio_module.VAD_FRAME_SAMPLES
has_probable_speech = audio_module.has_probable_speech
record_until_vad_stop = audio_module.record_until_vad_stop


class _FakeVadBackend:
    name = "webrtc"

    def __init__(self, speech_frames: set[bytes]):
        self.speech_frames = speech_frames

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        assert sample_rate == 16000
        return frame in self.speech_frames


class _FakeStream:
    def __init__(self, frames: list[bytes]):
        self._frames = frames
        self._index = 0
        self.stop_called = False
        self.close_called = False

    def read(self, frame_samples: int, exception_on_overflow: bool = False) -> bytes:
        assert frame_samples in {dictation_module.PCM_STREAM_CHUNK_SAMPLES, 1024, VAD_FRAME_SAMPLES}
        if self._index >= len(self._frames):
            return self._frames[-1]
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def stop_stream(self) -> None:
        self.stop_called = True

    def close(self) -> None:
        self.close_called = True


class _ManualStopAfter:
    def __init__(self, allowed_reads: int):
        self.allowed_reads = allowed_reads
        self.calls = 0

    def __call__(self) -> bool:
        current = self.calls
        self.calls += 1
        return current >= self.allowed_reads


class _FakePyAudioModule:
    paInt16 = object()

    def __init__(self, stream: _FakeStream):
        self._stream = stream
        self.open_kwargs: dict[str, object] | None = None
        self.open_calls: list[dict[str, object]] = []
        self.terminated = False

    def PyAudio(self):
        return self

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        self.open_calls.append(kwargs)
        return self._stream

    def terminate(self) -> None:
        self.terminated = True


class _FallbackPyAudioModule(_FakePyAudioModule):
    def __init__(self, stream: _FakeStream):
        super().__init__(stream)
        self._devices = [
            {
                "index": 0,
                "name": "pulse",
                "maxInputChannels": 32,
                "defaultSampleRate": 44100,
                "hostApi": 0,
            },
            {
                "index": 1,
                "name": "default",
                "maxInputChannels": 32,
                "defaultSampleRate": 44100,
                "hostApi": 0,
            },
        ]

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        self.open_calls.append(kwargs)
        if len(self.open_calls) == 1 and kwargs.get("input_device_index") is None:
            raise OSError(-9996, "Invalid input device (no default output device)")
        return self._stream

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self._devices[index]

    def get_host_api_info_by_index(self, index: int) -> dict:
        return {"name": "ALSA"}

    def get_default_input_device_info(self) -> dict:
        return {"index": 1}


class _PipeWirePreferredPyAudioModule(_FakePyAudioModule):
    def __init__(self, stream: _FakeStream):
        super().__init__(stream)
        self._devices = [
            {
                "index": 6,
                "name": "pipewire",
                "maxInputChannels": 128,
                "defaultSampleRate": 44100,
                "hostApi": 0,
            },
            {
                "index": 7,
                "name": "default",
                "maxInputChannels": 128,
                "defaultSampleRate": 44100,
                "hostApi": 0,
            },
        ]

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self._devices[index]

    def get_host_api_info_by_index(self, index: int) -> dict:
        return {"name": "ALSA"}

    def get_default_input_device_info(self) -> dict:
        return {"index": 7}


SPEECH_FRAME = b"\x01\x00" * (VAD_FRAME_BYTES // 2)
SILENCE_FRAME = b"\x00\x00" * (VAD_FRAME_BYTES // 2)


def test_has_probable_speech_rejects_silence(monkeypatch):
    monkeypatch.setattr("local_ai_dictation.audio.build_vad_backend", lambda mode: _FakeVadBackend({SPEECH_FRAME}))

    assert has_probable_speech(b"".join([SILENCE_FRAME] * 4), 16000) is False



def test_has_probable_speech_accepts_multiple_voiced_frames(monkeypatch):
    monkeypatch.setattr("local_ai_dictation.audio.build_vad_backend", lambda mode: _FakeVadBackend({SPEECH_FRAME}))

    assert has_probable_speech(b"".join([SPEECH_FRAME] * 3), 16000) is True



def test_vad_auto_stop_only_after_min_speech_and_silence_window():
    stream = _FakeStream(
        [
            SPEECH_FRAME,
            SPEECH_FRAME,
            SPEECH_FRAME,
            SILENCE_FRAME,
            SILENCE_FRAME,
            SPEECH_FRAME,
        ]
    )

    audio_data = record_until_vad_stop(
        stream,
        vad=_FakeVadBackend({SPEECH_FRAME}),
        stop_requested=lambda: False,
        min_speech_ms=90,
        max_silence_ms=60,
    )

    assert audio_data == b"".join(
        [
            SPEECH_FRAME,
            SPEECH_FRAME,
            SPEECH_FRAME,
            SILENCE_FRAME,
            SILENCE_FRAME,
        ]
    )


def test_vad_no_speech_session_stays_manual_stop_only():
    stream = _FakeStream(list(repeat(SILENCE_FRAME, 6)))

    audio_data = record_until_vad_stop(
        stream,
        vad=_FakeVadBackend({SPEECH_FRAME}),
        stop_requested=_ManualStopAfter(4),
        min_speech_ms=90,
        max_silence_ms=60,
    )

    assert audio_data == b"".join([SILENCE_FRAME] * 4)


def test_vad_manual_stop_remains_available_even_with_voiced_audio():
    stream = _FakeStream(list(repeat(SPEECH_FRAME, 6)))

    audio_data = record_until_vad_stop(
        stream,
        vad=_FakeVadBackend({SPEECH_FRAME}),
        stop_requested=_ManualStopAfter(2),
        min_speech_ms=90,
        max_silence_ms=60,
    )

    assert audio_data == b"".join([SPEECH_FRAME] * 2)


def test_dictation_recording_uses_vad_backend_when_enabled(monkeypatch):
    dictation_module._shutdown_event.clear()
    stream = _FakeStream([SILENCE_FRAME])
    pyaudio_module = _FakePyAudioModule(stream)
    config = DictationConfig(vad=True, clipboard=False)
    backend = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr("local_ai_dictation.dictation.build_vad_backend", lambda mode: backend)

    def _fake_record_until_vad_stop(stream_arg, **kwargs):
        calls["stream"] = stream_arg
        calls.update(kwargs)
        return b"vad-audio"

    monkeypatch.setattr("local_ai_dictation.dictation.record_until_vad_stop", _fake_record_until_vad_stop)

    audio_data = dictation_module.record_audio_interruptible(config, pyaudio_module)

    assert audio_data == b"vad-audio"
    assert pyaudio_module.open_kwargs is not None
    assert pyaudio_module.open_kwargs["frames_per_buffer"] == VAD_FRAME_SAMPLES
    assert calls["stream"] is stream
    assert calls["vad"] is backend
    assert calls["min_speech_ms"] == config.min_speech_ms
    assert calls["max_silence_ms"] == config.max_silence_ms
    assert stream.stop_called is True
    assert stream.close_called is True
    assert pyaudio_module.terminated is True



def test_dictation_recording_falls_back_to_explicit_pulse_device_when_default_device_is_broken(monkeypatch):
    dictation_module._shutdown_event.clear()
    stream = _FakeStream([b"\x00\x00" * 1024])
    pyaudio_module = _FallbackPyAudioModule(stream)
    config = DictationConfig(clipboard=False)

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: None)

    audio_data = dictation_module.record_audio_interruptible(
        config,
        pyaudio_module,
        stop_requested=_ManualStopAfter(1),
    )

    assert audio_data is not None
    assert audio_data.startswith(b"\x00\x00" * 1024)
    assert len(audio_data) >= len(b"\x00\x00" * 1024)
    assert pyaudio_module.open_calls[0]["input_device_index"] is None
    assert pyaudio_module.open_calls[1]["input_device_index"] == 0
    assert stream.stop_called is True
    assert stream.close_called is True
    assert pyaudio_module.terminated is True



def test_dictation_recording_resolves_named_input_device():
    dictation_module._shutdown_event.clear()
    stream = _FakeStream([b"\x00\x00" * 1024])
    pyaudio_module = _FallbackPyAudioModule(stream)
    config = DictationConfig(input_device="pulse", clipboard=False)

    audio_data = dictation_module.record_audio_interruptible(
        config,
        pyaudio_module,
        stop_requested=_ManualStopAfter(1),
    )

    assert audio_data is not None
    assert audio_data.startswith(b"\x00\x00" * 1024)
    assert len(audio_data) >= len(b"\x00\x00" * 1024)
    assert len(pyaudio_module.open_calls) == 1
    assert pyaudio_module.open_calls[0]["input_device_index"] == 0



def test_dictation_recording_prefers_pipewire_device_on_linux_when_input_device_is_unset(monkeypatch):
    dictation_module._shutdown_event.clear()
    stream = _FakeStream([b"\x00\x00" * 1024])
    pyaudio_module = _PipeWirePreferredPyAudioModule(stream)
    config = DictationConfig(clipboard=False)

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: None)

    audio_data = dictation_module.record_audio_interruptible(
        config,
        pyaudio_module,
        stop_requested=_ManualStopAfter(1),
    )

    assert audio_data is not None
    assert audio_data.startswith(b"\x00\x00" * 1024)
    assert len(audio_data) >= len(b"\x00\x00" * 1024)
    assert len(pyaudio_module.open_calls) == 1
    assert pyaudio_module.open_calls[0]["input_device_index"] == 6



def test_resolve_input_sample_rate_prefers_pipewire_default_rate_on_linux(monkeypatch):
    stream = _FakeStream([b"\x00\x00" * 1024])
    pyaudio_module = _PipeWirePreferredPyAudioModule(stream)

    monkeypatch.setattr("local_ai_dictation.audio.pulse_default_source_spec", lambda: None)
    sample_rate = audio_module.resolve_input_sample_rate(None, pyaudio_module)

    assert sample_rate == 44100



def test_parec_manual_stop_only_drains_one_capture_buffer(monkeypatch):
    dictation_module._shutdown_event.clear()
    clock = {"now": 0.0}

    class _FakeStdout:
        def __init__(self):
            self.reads = 0

        def read(self, size: int) -> bytes:
            assert size == 4096
            self.reads += 1
            return b"\x01\x00" * 16 if self.reads == 1 else b""

        def close(self) -> None:
            pass

    class _FakeParecProcess:
        def __init__(self):
            self.stdout = _FakeStdout()
            self.running = True
            self.signal_received = None

        def poll(self):
            return None if self.running else 0

        def send_signal(self, sent_signal) -> None:
            self.signal_received = sent_signal
            self.running = False

        def wait(self, timeout: int):
            return 0

    process = _FakeParecProcess()

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: "/usr/bin/parec")
    monkeypatch.setattr("local_ai_dictation.dictation.pulse_default_source_spec", lambda: (16000, 1))
    monkeypatch.setattr("local_ai_dictation.dictation.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("local_ai_dictation.dictation.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("local_ai_dictation.dictation.time.sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))

    audio_data = dictation_module._record_audio_with_parec(16000, stop_requested=lambda: True)

    assert audio_data == b"\x01\x00" * 16
    assert clock["now"] < 0.2
    assert process.signal_received == signal.SIGINT


def test_parec_stops_immediately_when_streaming_sink_fails(monkeypatch):
    dictation_module._shutdown_event.clear()
    pcm = b"\x01\x00" * 512
    callback_calls = 0

    class _FakeStdout:
        def __init__(self):
            self.reads = 0

        def read(self, size: int) -> bytes:
            self.reads += 1
            return pcm if self.reads <= 2 else b""

        def close(self) -> None:
            pass

    class _FakeParecProcess:
        def __init__(self):
            self.stdout = _FakeStdout()
            self.running = True
            self.signal_received = None

        def poll(self):
            return None if self.running else 0

        def send_signal(self, sent_signal) -> None:
            self.signal_received = sent_signal
            self.running = False

        def wait(self, timeout: int):
            return 0

    process = _FakeParecProcess()

    def sink(data: bytes) -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 2:
            raise RuntimeError("Willow stream failed")

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: "/usr/bin/parec")
    monkeypatch.setattr("local_ai_dictation.dictation.pulse_default_source_spec", lambda: (16000, 1))
    monkeypatch.setattr("local_ai_dictation.dictation.subprocess.Popen", lambda *args, **kwargs: process)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Willow stream failed"):
        dictation_module._record_audio_with_parec(
            16000,
            stop_requested=lambda: False,
            on_audio_chunk=sink,
        )

    assert time.monotonic() - started < 0.5
    assert callback_calls == 2
    assert process.signal_received == signal.SIGINT


def test_parec_streaming_validates_channel_count_before_starting_process(monkeypatch):
    process_started = False

    def popen(*args, **kwargs):
        nonlocal process_started
        process_started = True
        raise AssertionError("parec must not start for an unsupported stream format")

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: "/usr/bin/parec")
    monkeypatch.setattr("local_ai_dictation.dictation.pulse_default_source_spec", lambda: (48000, 6))
    monkeypatch.setattr("local_ai_dictation.dictation.subprocess.Popen", popen)

    with pytest.raises(ValueError, match="Unsupported channel count"):
        dictation_module._record_audio_with_parec(
            16000,
            stop_requested=lambda: True,
            on_audio_chunk=lambda data: None,
        )

    assert process_started is False


def test_parec_streaming_callback_converts_native_capture_incrementally(monkeypatch):
    dictation_module._shutdown_event.clear()
    clock = {"now": 0.0}
    command: list[str] = []
    streamed: list[bytes] = []
    native_pcm = b"\x01\x00\x03\x00" * 1024
    expected_pcm = audio_module.resample_pcm16_mono(
        audio_module.downmix_pcm16_to_mono(native_pcm, 2),
        48000,
        16000,
    )

    class _FakeStdout:
        def __init__(self):
            self.reads = 0

        def read(self, size: int) -> bytes:
            assert size == 4096
            self.reads += 1
            return native_pcm if self.reads == 1 else b""

        def close(self) -> None:
            pass

    class _FakeParecProcess:
        def __init__(self):
            self.stdout = _FakeStdout()
            self.running = True

        def poll(self):
            return None if self.running else 0

        def send_signal(self, sent_signal) -> None:
            assert sent_signal == signal.SIGINT
            self.running = False

        def wait(self, timeout: int):
            return 0

    process = _FakeParecProcess()

    def popen(args, **kwargs):
        command.extend(args)
        return process

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: "/usr/bin/parec")
    monkeypatch.setattr("local_ai_dictation.dictation.pulse_default_source_spec", lambda: (48000, 2))
    monkeypatch.setattr("local_ai_dictation.dictation.subprocess.Popen", popen)
    monkeypatch.setattr("local_ai_dictation.dictation.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("local_ai_dictation.dictation.time.sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))

    audio_data = dictation_module._record_audio_with_parec(
        16000,
        stop_requested=lambda: True,
        on_audio_chunk=streamed.append,
    )

    assert audio_data == expected_pcm
    assert b"".join(streamed) == expected_pcm
    assert command[command.index("--rate") + 1] == "48000"
    assert command[command.index("--channels") + 1] == "2"


def test_streaming_recording_falls_back_to_pyaudio_before_parec_sends_audio(monkeypatch):
    dictation_module._shutdown_event.clear()
    stream = _FakeStream([b"\x01\x00" * 512])
    pyaudio_module = _PipeWirePreferredPyAudioModule(stream)
    config = DictationConfig(clipboard=False)
    streamed: list[bytes] = []

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: "/usr/bin/parec")
    def fail_before_audio(*args, **kwargs):
        raise RuntimeError("parec setup failed")

    monkeypatch.setattr("local_ai_dictation.dictation._record_audio_with_parec", fail_before_audio)
    monkeypatch.setattr("local_ai_dictation.dictation.STOP_CAPTURE_DRAIN_SECONDS", 0.0)

    audio_data = dictation_module.record_audio_interruptible(
        config,
        pyaudio_module,
        sample_rate=16000,
        stop_requested=_ManualStopAfter(0),
        on_audio_chunk=streamed.append,
    )

    assert audio_data == b"\x01\x00" * 512
    assert streamed == [audio_data]
    assert len(pyaudio_module.open_calls) == 1


def test_dictation_recording_prefers_parec_capture_when_available(monkeypatch):
    dictation_module._shutdown_event.clear()
    stream = _FakeStream([b"\x00\x00" * 1024])
    pyaudio_module = _PipeWirePreferredPyAudioModule(stream)
    config = DictationConfig(clipboard=False)

    monkeypatch.setattr("local_ai_dictation.dictation.shutil.which", lambda name: "/usr/bin/parec" if name == "parec" else None)
    monkeypatch.setattr("local_ai_dictation.dictation._record_audio_with_parec", lambda sample_rate, *, stop_requested, status_stream=None: b"parec-audio")

    audio_data = dictation_module.record_audio_interruptible(
        config,
        pyaudio_module,
        stop_requested=_ManualStopAfter(1),
    )

    assert audio_data == b"parec-audio"
    assert pyaudio_module.open_calls == []



def test_streaming_pcm_converter_matches_whole_buffer_conversion():
    native_pcm = b"\x01\x00\x03\x00" * 4096
    converter = audio_module.Pcm16StreamConverter(48000, 2, 16000)

    converted_chunks = [
        converter.convert(native_pcm[:1001]),
        converter.convert(native_pcm[1001:5099]),
        converter.convert(native_pcm[5099:]),
    ]
    converter.finish()

    expected = audio_module.resample_pcm16_mono(
        audio_module.downmix_pcm16_to_mono(native_pcm, 2),
        48000,
        16000,
    )
    assert b"".join(converted_chunks) == expected


def test_resample_pcm16_mono_downsamples_to_model_rate():
    source_audio = b"\x00\x00" * 44100

    resampled = audio_module.resample_pcm16_mono(source_audio, 44100, 16000)

    assert len(resampled) == 16000 * 2
