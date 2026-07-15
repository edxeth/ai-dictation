from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


whisper_module = importlib.import_module("local_ai_dictation.whisper")
WhisperEngine = whisper_module.WhisperEngine
warmup = whisper_module.warmup


class _FakeWord:
    def __init__(self, start: float, end: float, word: str) -> None:
        self.start = start
        self.end = end
        self.word = word


class _FakeSegment:
    def __init__(self, text: str, words: list[_FakeWord] | None = None) -> None:
        self.text = text
        self.words = words


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_path: str, **kwargs):
        self.calls.append({"audio_path": audio_path, **kwargs})
        return [_FakeSegment(" hello")], object()



class _StreamingFakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._condition = threading.Condition()

    def transcribe(self, audio, **kwargs):
        with self._condition:
            call_index = len(self.calls)
            self.calls.append({"audio": audio, **kwargs})
            self._condition.notify_all()

        if not kwargs.get("word_timestamps"):
            return [_FakeSegment(" short transcript")], object()

        words_by_call = [
            [
                _FakeWord(0.5, 0.5, " one"),
                _FakeWord(1.5, 1.5, " two"),
                _FakeWord(2.5, 2.5, " three"),
                _FakeWord(3.5, 3.5, " four"),
            ],
            [
                _FakeWord(0.5, 0.5, " three"),
                _FakeWord(1.5, 1.5, " four"),
                _FakeWord(2.8, 2.8, " five"),
                _FakeWord(3.6, 3.6, " six"),
            ],
            [
                _FakeWord(0.5, 0.5, " five"),
                _FakeWord(1.5, 1.5, " six"),
                _FakeWord(2.8, 2.8, " seven"),
                _FakeWord(3.6, 3.6, " eight"),
            ],
        ]
        if call_index >= 3 and getattr(audio, "shape", (0,))[0] <= 2 * 16000:
            words = [
                _FakeWord(0.5, 0.5, " seven"),
                _FakeWord(1.5, 1.5, " eight"),
            ]
        else:
            words = words_by_call[min(call_index, len(words_by_call) - 1)]
        return [_FakeSegment("".join(word.word for word in words), words)], object()

    def wait_for_calls(self, count: int) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: len(self.calls) >= count, timeout=1)


def test_whisper_streaming_short_audio_preserves_batch_transcription():
    model = _StreamingFakeModel()
    engine = WhisperEngine(model, device="cpu", compute_type="int8", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()

    session.send_audio(b"\x01\x00" * 16000)
    text = session.finish()

    assert text == "short transcript"
    assert len(model.calls) == 1
    assert model.calls[0].get("word_timestamps", False) is False


def test_whisper_streaming_commits_timed_windows_without_duplicate_words():
    model = _StreamingFakeModel()
    engine = WhisperEngine(model, device="cuda", compute_type="float16", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()

    session.send_audio(b"\x01\x00" * (4 * 16000))
    assert model.wait_for_calls(1)
    session.send_audio(b"\x01\x00" * (2 * 16000))
    assert model.wait_for_calls(2)
    session.send_audio(b"\x01\x00" * (2 * 16000))
    text = session.finish()

    assert text == "one two three four five six seven eight"
    assert len(model.calls) == 3
    assert all(call["word_timestamps"] is True for call in model.calls)
    assert all(isinstance(call["audio"], np.ndarray) for call in model.calls)


def test_whisper_streaming_keeps_new_word_just_after_window_boundary():
    class BoundaryModel(_StreamingFakeModel):
        def transcribe(self, audio, **kwargs):
            with self._condition:
                call_index = len(self.calls)
                self.calls.append({"audio": audio, **kwargs})
                self._condition.notify_all()
            if not kwargs.get("word_timestamps"):
                return [_FakeSegment(" one two three")], object()
            words = (
                [_FakeWord(1.0, 1.0, " one"), _FakeWord(3.9, 3.9, " two")]
                if call_index == 0
                else [_FakeWord(1.9, 1.9, " two"), _FakeWord(2.15, 2.15, " three")]
            )
            return [_FakeSegment("".join(word.word for word in words), words)], object()

    model = BoundaryModel()
    engine = WhisperEngine(model, device="cuda", compute_type="float16", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()

    session.send_audio(b"\x01\x00" * (4 * 16000))
    assert model.wait_for_calls(1)
    session.send_audio(b"\x01\x00" * int(0.2 * 16000))
    text = session.finish()

    assert text == "one two three"


def test_whisper_streaming_falls_back_when_overlap_revises_a_word():
    class RevisionModel(_StreamingFakeModel):
        def transcribe(self, audio, **kwargs):
            with self._condition:
                call_index = len(self.calls)
                self.calls.append({"audio": audio, **kwargs})
                self._condition.notify_all()
            if not kwargs.get("word_timestamps"):
                return [_FakeSegment(" one cap now")], object()
            words = (
                [_FakeWord(1.0, 1.0, " one"), _FakeWord(3.9, 3.9, " cat")]
                if call_index == 0
                else [_FakeWord(1.9, 1.9, " cap"), _FakeWord(2.15, 2.15, " now")]
            )
            return [_FakeSegment("".join(word.word for word in words), words)], object()

    model = RevisionModel()
    engine = WhisperEngine(model, device="cuda", compute_type="float16", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()

    session.send_audio(b"\x01\x00" * (4 * 16000))
    assert model.wait_for_calls(1)
    session.send_audio(b"\x01\x00" * int(0.2 * 16000))
    text = session.finish()

    assert text == "one cap now"
    assert model.calls[-1].get("word_timestamps", False) is False


def test_whisper_streaming_falls_back_when_overlap_inserts_a_word():
    class InsertionModel(_StreamingFakeModel):
        def transcribe(self, audio, **kwargs):
            with self._condition:
                call_index = len(self.calls)
                self.calls.append({"audio": audio, **kwargs})
                self._condition.notify_all()
            if not kwargs.get("word_timestamps"):
                return [_FakeSegment(" one cat um now")], object()
            words = (
                [_FakeWord(1.0, 1.0, " one"), _FakeWord(3.5, 3.5, " cat")]
                if call_index == 0
                else [
                    _FakeWord(1.5, 1.5, " cat"),
                    _FakeWord(1.8, 1.8, " um"),
                    _FakeWord(2.15, 2.15, " now"),
                ]
            )
            return [_FakeSegment("".join(word.word for word in words), words)], object()

    model = InsertionModel()
    engine = WhisperEngine(model, device="cuda", compute_type="float16", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()

    session.send_audio(b"\x01\x00" * (4 * 16000))
    assert model.wait_for_calls(1)
    session.send_audio(b"\x01\x00" * int(0.2 * 16000))
    text = session.finish()

    assert text == "one cat um now"
    assert model.calls[-1].get("word_timestamps", False) is False


def test_whisper_cpu_capture_does_not_backpressure_on_rolling_inference():
    model = _StreamingFakeModel()
    engine = WhisperEngine(model, device="cpu", compute_type="int8", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()
    pcm = b"\x01\x00" * (30 * 16000)

    started = time.perf_counter()
    session.send_audio(pcm)
    send_elapsed = time.perf_counter() - started
    text = session.finish()

    assert send_elapsed < 0.1
    assert text == "short transcript"
    assert len(model.calls) == 1
    assert model.calls[0].get("word_timestamps", False) is False


def test_whisper_cancel_waits_until_model_inference_releases_ownership():
    inference_started = threading.Event()
    release_inference = threading.Event()

    class BlockingModel(_StreamingFakeModel):
        def transcribe(self, audio, **kwargs):
            if kwargs.get("word_timestamps"):
                inference_started.set()
                assert release_inference.wait(timeout=1)
            return super().transcribe(audio, **kwargs)

    model = BlockingModel()
    engine = WhisperEngine(model, device="cuda", compute_type="float16", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()
    session.send_audio(b"\x01\x00" * (4 * 16000))
    assert inference_started.wait(timeout=1)

    cancel_started = time.perf_counter()
    session.cancel()
    assert time.perf_counter() - cancel_started < 0.1

    finished = threading.Event()
    wait_thread = threading.Thread(
        target=lambda: (session.wait_finished(), finished.set()),
        daemon=True,
    )
    wait_thread.start()
    time.sleep(0.05)
    assert finished.is_set() is False

    release_inference.set()
    wait_thread.join(timeout=1)
    assert finished.is_set()


def test_whisper_streaming_falls_back_to_full_transcription_when_rolling_fails():
    class FailingRollingModel(_StreamingFakeModel):
        def transcribe(self, audio, **kwargs):
            if kwargs.get("word_timestamps"):
                with self._condition:
                    self.calls.append({"audio": audio, **kwargs})
                    self._condition.notify_all()
                raise RuntimeError("rolling inference failed")
            return super().transcribe(audio, **kwargs)

    model = FailingRollingModel()
    engine = WhisperEngine(model, device="cuda", compute_type="float16", model_id="fake")
    session = engine.start_streaming()
    session.wait_ready()

    session.send_audio(b"\x01\x00" * (4 * 16000))
    assert model.wait_for_calls(1)
    text = session.finish()

    assert text == "short transcript"
    assert len(model.calls) == 2
    assert model.calls[0]["word_timestamps"] is True
    assert model.calls[1].get("word_timestamps", False) is False


def test_whisper_warmup_forces_model_inference():
    model = _FakeModel()
    engine = WhisperEngine(model, device="cpu", compute_type="int8", model_id="fake")

    warmup(engine)

    call = model.calls[0]
    assert call["language"] == "en"
    assert call["condition_on_previous_text"] is False
    assert call["vad_filter"] is False
    assert call["beam_size"] == 1
    assert getattr(call["audio_path"], "shape", None) == (16000,)


def test_whisper_engine_enables_vad_filter_for_transcription():
    model = _FakeModel()
    engine = WhisperEngine(model, device="cpu", compute_type="int8", model_id="fake")

    result = engine.transcribe(["/tmp/sample.wav"])

    assert len(result) == 1
    assert result[0].text == "hello"
    assert model.calls == [
        {
            "audio_path": "/tmp/sample.wav",
            "language": "en",
            "condition_on_previous_text": False,
            "vad_filter": True,
            "beam_size": 5,
        }
    ]
