"""Fixture-based benchmark helpers and CLI support for AI Dictation."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import wave

from local_ai_dictation.errors import AppError, ExitCode, MODEL_TRANSCRIBE_FAILED
from local_ai_dictation.model import load_engine, transcribe_wav
from local_ai_dictation.types import BenchmarkReport, DictationConfig, ModelBackend
import unicodedata


class _RedirectStdoutToStderr:
    def __enter__(self):
        self._stdout_fd = None
        self._saved_stdout_fd = None
        self._python_redirect = None

        try:
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
        except (AttributeError, io.UnsupportedOperation, ValueError):
            self._python_redirect = redirect_stdout(sys.stderr)
            self._python_redirect.__enter__()
            return self

        if stdout_fd == stderr_fd:
            return self

        sys.stdout.flush()
        sys.stderr.flush()
        self._stdout_fd = stdout_fd
        self._saved_stdout_fd = os.dup(stdout_fd)
        os.dup2(stderr_fd, stdout_fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._python_redirect is not None:
            return self._python_redirect.__exit__(exc_type, exc, tb)

        if self._stdout_fd is not None and self._saved_stdout_fd is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(self._saved_stdout_fd, self._stdout_fd)
            os.close(self._saved_stdout_fd)
            self._saved_stdout_fd = None
        return False


def normalize_transcript(text: str) -> str:
    """Normalize transcript text using the milestone-1 exact-match contract."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    alnum_or_space = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(alnum_or_space.split())


def expected_sidecar_path(fixture_path: str | Path) -> Path:
    """Return the expected-transcript sidecar path for a WAV fixture."""
    return Path(fixture_path).with_suffix(".expected.txt")


def load_expected_transcript(
    fixture_path: str | Path,
    *,
    required: bool = False,
) -> str | None:
    """Load the raw expected transcript sidecar for a fixture if present."""
    sidecar_path = expected_sidecar_path(fixture_path)
    if not sidecar_path.is_file():
        if required:
            raise FileNotFoundError(f"Expected transcript sidecar not found: {sidecar_path}")
        return None
    return sidecar_path.read_text(encoding="utf-8")


def load_normalized_expected_transcript(
    fixture_path: str | Path,
    *,
    required: bool = False,
) -> str | None:
    """Load and normalize the expected transcript sidecar for a fixture if present."""
    transcript = load_expected_transcript(fixture_path, required=required)
    if transcript is None:
        return None
    return normalize_transcript(transcript)


def normalized_exact_match(actual: str, expected: str) -> bool:
    """Compare two transcripts using the deterministic normalization contract."""
    return normalize_transcript(actual) == normalize_transcript(expected)


def _validate_fixture_path(fixture_path: str | Path) -> Path:
    candidate = Path(fixture_path)
    candidate_text = str(fixture_path)

    if "://" in candidate_text:
        raise ValueError(f"Fixture path must be local: {fixture_path}")
    if candidate.suffix.lower() != ".wav":
        raise ValueError(f"Fixture must be a WAV file: {fixture_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    try:
        with wave.open(str(candidate), "rb") as wav_file:
            wav_file.getnchannels()
            wav_file.getframerate()
            wav_file.getnframes()
    except wave.Error as exc:
        raise ValueError(f"Fixture is not a readable WAV file: {fixture_path}") from exc

    return candidate


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def benchmark_fixture(
    fixture_path: str | Path,
    *,
    runs: int,
    backend: ModelBackend = "parakeet",
    cpu: bool = False,
    check_expected: bool = False,
    perf_counter=time.perf_counter,
    load_engine_fn=None,
    transcribe_wav_fn=None,
) -> BenchmarkReport:
    if runs <= 0:
        raise ValueError("Benchmark runs must be a positive integer")

    fixture = _validate_fixture_path(fixture_path)
    expected_text = load_expected_transcript(fixture, required=check_expected)

    if load_engine_fn is None or transcribe_wav_fn is None:
        if backend == "whisper":
            from local_ai_dictation.whisper import load_engine as selected_load_engine
            from local_ai_dictation.whisper import transcribe_wav as selected_transcribe_wav
        elif backend == "willow":
            from local_ai_dictation.willow import load_engine as selected_load_engine
            from local_ai_dictation.willow import transcribe_wav as selected_transcribe_wav
        else:
            selected_load_engine = load_engine
            selected_transcribe_wav = transcribe_wav
        load_engine_fn = load_engine_fn or selected_load_engine
        transcribe_wav_fn = transcribe_wav_fn or selected_transcribe_wav

    load_start = perf_counter()
    engine = load_engine_fn(DictationConfig(backend=backend, cpu=cpu))
    load_ms = (perf_counter() - load_start) * 1000.0

    run_ms: list[float] = []
    final_result = None
    for _ in range(runs):
        run_start = perf_counter()
        final_result = transcribe_wav_fn(engine, fixture)
        run_ms.append((perf_counter() - run_start) * 1000.0)

    if final_result is None:
        raise AppError(MODEL_TRANSCRIBE_FAILED, "Benchmark did not produce a transcription result")

    transcript = final_result.text
    normalized_transcript = normalize_transcript(transcript)
    normalized_match = (
        normalized_exact_match(transcript, expected_text) if expected_text is not None else None
    )
    total_ms = load_ms + sum(run_ms)

    device = final_result.device or getattr(engine, "_parakeet_device", None) or ("cpu" if cpu else "cuda")

    return BenchmarkReport(
        fixture=str(fixture_path),
        runs=runs,
        device=device,
        load_ms=load_ms,
        run_ms=run_ms,
        mean_transcribe_ms=statistics.mean(run_ms),
        median_transcribe_ms=statistics.median(run_ms),
        p95_transcribe_ms=_p95(run_ms),
        total_ms=total_ms,
        transcript=transcript,
        normalized_transcript=normalized_transcript,
        expected_text=expected_text,
        normalized_match=normalized_match,
    )


def run_benchmark_command(
    fixture_path: str,
    *,
    runs: int,
    backend: ModelBackend = "parakeet",
    cpu: bool = False,
    json_output: bool = False,
    check_expected: bool = False,
) -> int:
    try:
        with _RedirectStdoutToStderr():
            report = benchmark_fixture(
                fixture_path,
                runs=runs,
                backend=backend,
                cpu=cpu,
                check_expected=check_expected,
            )
    except (AppError, FileNotFoundError, ValueError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return int(ExitCode.ERROR)

    if json_output:
        print(json.dumps(asdict(report)))
    else:
        print(f"fixture: {report.fixture}")
        print(f"runs: {report.runs}")
        print(f"device: {report.device}")
        print(f"load_ms: {report.load_ms:.3f}")
        print(f"mean_transcribe_ms: {report.mean_transcribe_ms:.3f}")
        print(f"median_transcribe_ms: {report.median_transcribe_ms:.3f}")
        print(f"p95_transcribe_ms: {report.p95_transcribe_ms:.3f}")
        print(f"total_ms: {report.total_ms:.3f}")
        print(f"transcript: {report.transcript}")
        if report.expected_text is not None:
            print(f"normalized_match: {report.normalized_match}")

    return int(ExitCode.OK)
