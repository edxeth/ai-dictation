"""Stable internal data models and protocol boundaries for AI Dictation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

SchemaVersion = Literal[1]
TranscriptFormat = Literal["text", "json"]
ModelBackend = Literal["parakeet", "whisper", "willow"]
IssueSeverity = Literal["warn", "fail"]


@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str
    default_sample_rate: int
    max_input_channels: int
    host_api: str = "unknown"
    is_default_candidate: bool = False


@dataclass(frozen=True)
class DoctorIssue:
    code: str
    severity: IssueSeverity
    message: str
    remediation: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    schema_version: SchemaVersion = 1
    platform: dict[str, Any] = field(default_factory=dict)
    wsl: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    pulse: dict[str, Any] = field(default_factory=dict)
    audio_devices: list[AudioDevice] = field(default_factory=list)
    clipboard: dict[str, Any] = field(default_factory=dict)
    cuda: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: SchemaVersion = 1
    fixture: str = ""
    runs: int = 0
    device: str = "cpu"
    load_ms: float = 0.0
    run_ms: list[float] = field(default_factory=list)
    mean_transcribe_ms: float = 0.0
    median_transcribe_ms: float = 0.0
    p95_transcribe_ms: float = 0.0
    total_ms: float = 0.0
    transcript: str = ""
    normalized_transcript: str | None = None
    expected_text: str | None = None
    normalized_match: bool | None = None


@dataclass(frozen=True)
class DictationConfig:
    backend: ModelBackend = "whisper"
    cpu: bool = False
    input_device: int | str | None = None
    vad: bool = False
    max_silence_ms: int = 1200
    min_speech_ms: int = 300
    vad_mode: int = 2
    format: TranscriptFormat = "text"
    output_file: str | None = None
    clipboard: bool = True
    debug: bool = False
    log_file: str = "transcriber.debug.log"
    list_devices: bool = False


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    normalized_text: str | None = None
    device: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VadBackend(Protocol):
    name: str

    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


class TranscriptionEngine(Protocol):
    def transcribe(self, audio: Sequence[str]) -> Any: ...
