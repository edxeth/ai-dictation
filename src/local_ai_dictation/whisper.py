"""Whisper backend helpers for local dictation."""

from __future__ import annotations

import importlib
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from local_ai_dictation.errors import MODEL_IMPORT_FAILED, MODEL_TRANSCRIBE_FAILED, ModelError
from local_ai_dictation.gpu import nvidia_driver_loaded
from local_ai_dictation.types import DictationConfig, TranscriptionResult


WHISPER_MODEL_ID = "deepdml/faster-distil-whisper-large-v3.5"
WHISPER_SAMPLE_RATE = 16000
WHISPER_STREAM_WINDOW_SECONDS = 4
WHISPER_STREAM_STRIDE_SECONDS = 2
WHISPER_STREAM_RECONCILE_TOLERANCE_SECONDS = 0.65


class _DummyParameter:
    def __init__(self, device: str) -> None:
        self.device = device


class _TranscriptItem:
    def __init__(self, text: str) -> None:
        self.text = text


def _pcm_to_waveform(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        raise ValueError("Whisper PCM stream ended with an incomplete sample")
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _word_key(text: str) -> str:
    return re.sub(r"[^\w]+", "", text).lower()


class WhisperStreamingSession:
    """Runs stable Whisper prefix windows while capture appends independently."""

    def __init__(self, engine: "WhisperEngine") -> None:
        self._engine = engine
        self._condition = threading.Condition()
        self._audio = bytearray()
        self._cancelled = False
        self._finish_requested = False
        self._finished = threading.Event()
        self._transcript: str | None = None
        self._error: ModelError | None = None
        self._committed_words: list[str] = []
        self._committed_until = 0.0
        self._pending_words: list[tuple[float, float, str]] = []
        self._pending_window_end = 0.0
        self._previous_window_words: list[tuple[float, float, str]] = []
        self._rolling_disabled = engine._parakeet_device != "cuda"
        self._next_window_end = WHISPER_STREAM_WINDOW_SECONDS * WHISPER_SAMPLE_RATE
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_ready(self) -> None:
        self._raise_if_failed()

    def send_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._condition:
            if self._finish_requested or self._cancelled:
                raise RuntimeError("Cannot send Whisper audio after finish")
            self._audio.extend(pcm)
            self._condition.notify_all()

    def finish(self) -> str:
        with self._condition:
            self._finish_requested = True
            self._condition.notify_all()
        self._finished.wait()
        self._raise_if_failed()
        with self._condition:
            if self._cancelled:
                raise ModelError(MODEL_TRANSCRIBE_FAILED, "Whisper streaming session was cancelled")
        return self._transcript or ""

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._finish_requested = True
            self._condition.notify_all()

    def wait_finished(self, timeout: float | None = None) -> bool:
        return self._finished.wait(timeout=timeout)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._cancelled
                        or self._finish_requested
                        or (
                            not self._rolling_disabled
                            and len(self._audio) // 2 >= self._next_window_end
                        )
                    )
                    if self._cancelled:
                        return
                    if self._finish_requested:
                        pcm = bytes(self._audio)
                        break
                    window_samples = WHISPER_STREAM_WINDOW_SECONDS * WHISPER_SAMPLE_RATE
                    start_sample = self._next_window_end - window_samples
                    window_pcm = bytes(
                        self._audio[start_sample * 2 : self._next_window_end * 2]
                    )

                inference_started = time.monotonic()
                self._process_rolling_window(window_pcm, start_sample=start_sample)
                inference_seconds = time.monotonic() - inference_started
                with self._condition:
                    if (
                        inference_seconds >= WHISPER_STREAM_STRIDE_SECONDS
                        or len(self._audio) // 2 >= self._next_window_end
                    ):
                        self._disable_rolling()

            transcript = self._finish_transcription(pcm)
            with self._condition:
                if not self._cancelled:
                    self._transcript = transcript
        except ModelError as exc:
            with self._condition:
                if not self._cancelled:
                    self._error = exc
        except Exception as exc:
            with self._condition:
                if not self._cancelled:
                    self._error = ModelError(MODEL_TRANSCRIBE_FAILED, str(exc))
        finally:
            self._finished.set()

    def _process_rolling_window(self, pcm: bytes, *, start_sample: int) -> None:
        try:
            words = self._engine._transcribe_timed_words(
                pcm,
                offset_seconds=start_sample / WHISPER_SAMPLE_RATE,
            )
        except ModelError:
            self._disable_rolling()
            return

        if not words:
            self._disable_rolling()
            return

        matched_indices: set[int] = set()
        if self._previous_window_words:
            matched_indices, ambiguous = self._reconcile_overlap(
                words,
                current_window_start=start_sample / WHISPER_SAMPLE_RATE,
            )
            if ambiguous:
                self._disable_rolling()
                return
            if self._pending_words:
                self._commit_pending_words()

        self._pending_words = [
            word
            for index, word in enumerate(words)
            if index not in matched_indices and word[1] > self._committed_until
        ]
        self._pending_window_end = self._next_window_end / WHISPER_SAMPLE_RATE
        self._previous_window_words = words
        self._next_window_end += WHISPER_STREAM_STRIDE_SECONDS * WHISPER_SAMPLE_RATE

    def _reconcile_overlap(
        self,
        current_words: list[tuple[float, float, str]],
        *,
        current_window_start: float,
    ) -> tuple[set[int], bool]:
        expected_overlap = [
            word for word in self._previous_window_words if word[1] >= current_window_start
        ]
        crossing_words = [word for word in expected_overlap if word[0] < current_window_start]
        stable_words = expected_overlap[len(crossing_words) :]
        agreements: list[
            tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]
        ] = []
        for crossing_count in range(len(crossing_words) + 1):
            expected = crossing_words[len(crossing_words) - crossing_count :] + stable_words
            current = current_words[: len(expected)]
            if len(current) != len(expected):
                continue
            if [_word_key(word[2]) for word in expected] != [
                _word_key(word[2]) for word in current
            ]:
                continue
            if any(
                abs(previous[1] - observed[1])
                > WHISPER_STREAM_RECONCILE_TOLERANCE_SECONDS
                for previous, observed in zip(expected, current, strict=True)
            ):
                continue
            agreements.append((expected, current))
        if len(agreements) != 1:
            return set(), True
        _expected_overlap, current_overlap = agreements[0]
        if any(
            word[1] <= self._pending_window_end
            for word in current_words[len(current_overlap) :]
        ):
            return set(), True

        reconciled = list(self._pending_words)
        used_current: set[int] = set()
        for pending_index, pending in enumerate(reconciled):
            for current_index, current in enumerate(current_overlap):
                if current_index in used_current:
                    continue
                if _word_key(current[2]) != _word_key(pending[2]):
                    continue
                if abs(current[1] - pending[1]) > WHISPER_STREAM_RECONCILE_TOLERANCE_SECONDS:
                    continue
                reconciled[pending_index] = current
                used_current.add(current_index)
                break
        self._pending_words = reconciled
        return set(range(len(current_overlap))), False

    def _commit_pending_words(self) -> None:
        self._committed_words.extend(text for _start, _end, text in self._pending_words)
        self._committed_until = self._pending_window_end

    def _disable_rolling(self) -> None:
        self._rolling_disabled = True
        self._committed_words.clear()
        self._committed_until = 0.0
        self._pending_words.clear()
        self._pending_window_end = 0.0
        self._previous_window_words.clear()

    def _finish_transcription(self, pcm: bytes) -> str:
        if self._rolling_disabled or (not self._committed_words and not self._pending_words):
            return self._engine._transcribe_pcm_text(pcm)

        overlap_seconds = WHISPER_STREAM_WINDOW_SECONDS - WHISPER_STREAM_STRIDE_SECONDS
        pending_boundary = self._pending_window_end if self._pending_words else self._committed_until
        final_start_seconds = max(0.0, pending_boundary - overlap_seconds)
        final_start_sample = int(final_start_seconds * WHISPER_SAMPLE_RATE)
        if final_start_sample * 2 >= len(pcm):
            self._commit_pending_words()
            return "".join(self._committed_words).strip()

        final_words = self._engine._transcribe_timed_words(
            pcm[final_start_sample * 2 :],
            offset_seconds=final_start_seconds,
        )
        matched_indices: set[int] = set()
        if self._previous_window_words:
            matched_indices, ambiguous = self._reconcile_overlap(
                final_words,
                current_window_start=final_start_seconds,
            )
            if ambiguous:
                return self._engine._transcribe_pcm_text(pcm)
            if self._pending_words:
                self._commit_pending_words()

        new_words = [
            text
            for index, (_start, end, text) in enumerate(final_words)
            if index not in matched_indices and end > self._committed_until
        ]
        total_duration = len(pcm) / 2 / WHISPER_SAMPLE_RATE
        if total_duration > self._committed_until and not new_words:
            return self._engine._transcribe_pcm_text(pcm)
        self._committed_words.extend(new_words)
        return "".join(self._committed_words).strip()


class WhisperEngine:
    def __init__(self, model: Any, *, device: str, compute_type: str, model_id: str) -> None:
        self._model = model
        self._parakeet_device = device
        self._parakeet_compute_type = compute_type
        self._parakeet_model_id = model_id
        self._inference_lock = threading.Lock()

    def to(self, device: str) -> Any:
        self._parakeet_device = device
        return self

    def eval(self) -> Any:
        return self

    def parameters(self):
        yield _DummyParameter(self._parakeet_device)

    def transcribe(self, audio: Sequence[str], *, verbose: bool = False) -> list[Any]:
        if not audio:
            return [_TranscriptItem("")]
        return [_TranscriptItem(self._transcribe_text(str(audio[0])))]

    def start_streaming(self) -> WhisperStreamingSession:
        return WhisperStreamingSession(self)

    def _transcribe_segments(self, audio: Any, *, word_timestamps: bool = False) -> list[Any]:
        kwargs: dict[str, Any] = {
            "language": "en",
            "condition_on_previous_text": False,
            "vad_filter": True,
            "beam_size": 5,
        }
        if word_timestamps:
            kwargs["word_timestamps"] = True
        try:
            with self._inference_lock:
                segments, _info = self._model.transcribe(audio, **kwargs)
                return list(segments)
        except Exception as exc:
            raise ModelError(MODEL_TRANSCRIBE_FAILED, str(exc)) from exc

    def _transcribe_text(self, audio: Any) -> str:
        return "".join(segment.text for segment in self._transcribe_segments(audio)).strip()

    def _transcribe_pcm_text(self, pcm: bytes) -> str:
        return self._transcribe_text(_pcm_to_waveform(pcm))

    def _transcribe_timed_words(
        self,
        pcm: bytes,
        *,
        offset_seconds: float,
    ) -> list[tuple[float, float, str]]:
        words: list[tuple[float, float, str]] = []
        for segment in self._transcribe_segments(_pcm_to_waveform(pcm), word_timestamps=True):
            for word in getattr(segment, "words", None) or []:
                words.append(
                    (
                        offset_seconds + float(word.start),
                        offset_seconds + float(word.end),
                        str(word.word),
                    )
                )
        return words


def _load_runtime_dependencies() -> tuple[Any, Any]:
    try:
        faster_whisper = importlib.import_module("faster_whisper")
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc
    return faster_whisper, torch_module


def _compute_type(config: DictationConfig, torch_module: Any) -> tuple[str, str]:
    use_cuda = (
        nvidia_driver_loaded()
        and bool(getattr(torch_module.cuda, "is_available", lambda: False)())
        and not config.cpu
    )
    if use_cuda:
        return "cuda", "float16"
    return "cpu", "int8"


def load_engine(config: DictationConfig) -> WhisperEngine:
    faster_whisper, torch_module = _load_runtime_dependencies()
    device, compute_type = _compute_type(config, torch_module)

    try:
        model = faster_whisper.WhisperModel(WHISPER_MODEL_ID, device=device, compute_type=compute_type)
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc

    return WhisperEngine(model, device=device, compute_type=compute_type, model_id=WHISPER_MODEL_ID)


def warmup(engine: WhisperEngine) -> None:
    engine.eval()
    with engine._inference_lock:
        segments, _info = engine._model.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,
            beam_size=1,
        )
        list(segments)


def transcribe_wav(engine: WhisperEngine, path: str | Path) -> TranscriptionResult:
    result = engine.transcribe([str(path)], verbose=False)
    first = result[0] if result else _TranscriptItem("")
    transcript_text = getattr(first, "text", first if isinstance(first, str) else str(first))
    return TranscriptionResult(
        text=transcript_text,
        device=getattr(engine, "_parakeet_device", None),
        metadata={
            "backend": "whisper",
            "compute_type": getattr(engine, "_parakeet_compute_type", None),
            "model_id": getattr(engine, "_parakeet_model_id", WHISPER_MODEL_ID),
        },
    )


def check_model_cache(_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        _load_runtime_dependencies()
        import_ready = True
        import_error = None
    except ModelError as exc:
        import_ready = False
        import_error = str(exc)

    detail = "Whisper runtime imports look ready" if import_ready else "Whisper runtime imports failed"
    return {
        "checked": True,
        "cache_present": None,
        "cache_path": None,
        "model_id": WHISPER_MODEL_ID,
        "import_ready": import_ready,
        "import_error": import_error,
        "detail": detail,
    }
