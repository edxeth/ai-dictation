"""Application error codes and exceptions for AI Dictation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal


IssueCode = Literal[
    "AUDIO_BACKEND_UNREACHABLE",
    "AUDIO_NO_INPUT_DEVICE",
    "AUDIO_DEVICE_NOT_FOUND",
    "AUDIO_DEVICE_AMBIGUOUS",
    "AUDIO_SAMPLE_RATE_UNSUPPORTED",
    "CLIPBOARD_UNAVAILABLE",
    "CUDA_UNAVAILABLE",
    "MODEL_CACHE_MISSING",
    "MODEL_IMPORT_FAILED",
    "MODEL_TRANSCRIBE_FAILED",
]

AUDIO_BACKEND_UNREACHABLE: IssueCode = "AUDIO_BACKEND_UNREACHABLE"
AUDIO_NO_INPUT_DEVICE: IssueCode = "AUDIO_NO_INPUT_DEVICE"
AUDIO_DEVICE_NOT_FOUND: IssueCode = "AUDIO_DEVICE_NOT_FOUND"
AUDIO_DEVICE_AMBIGUOUS: IssueCode = "AUDIO_DEVICE_AMBIGUOUS"
AUDIO_SAMPLE_RATE_UNSUPPORTED: IssueCode = "AUDIO_SAMPLE_RATE_UNSUPPORTED"
CLIPBOARD_UNAVAILABLE: IssueCode = "CLIPBOARD_UNAVAILABLE"
CUDA_UNAVAILABLE: IssueCode = "CUDA_UNAVAILABLE"
MODEL_CACHE_MISSING: IssueCode = "MODEL_CACHE_MISSING"
MODEL_IMPORT_FAILED: IssueCode = "MODEL_IMPORT_FAILED"
MODEL_TRANSCRIBE_FAILED: IssueCode = "MODEL_TRANSCRIBE_FAILED"


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1
    RECORDING_BLOCKED = 2
    DEGRADED = 3


@dataclass(slots=True)
class AppError(Exception):
    code: IssueCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class AudioError(AppError):
    pass


class ModelError(AppError):
    pass
