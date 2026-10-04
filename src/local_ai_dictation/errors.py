"""Application error codes and exceptions for AI Dictation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

IssueCode = Literal[
    "AUDIO_BACKEND_UNREACHABLE",
    "AUDIO_NO_INPUT_DEVICE",
    "CLIPBOARD_UNAVAILABLE",
    "CUDA_UNAVAILABLE",
    "MODEL_CACHE_MISSING",
    "MODEL_IMPORT_FAILED",
    "MODEL_TRANSCRIBE_FAILED",
    "WILLOW_AUTH_REQUIRED",
]

AUDIO_BACKEND_UNREACHABLE: IssueCode = "AUDIO_BACKEND_UNREACHABLE"
AUDIO_NO_INPUT_DEVICE: IssueCode = "AUDIO_NO_INPUT_DEVICE"
CLIPBOARD_UNAVAILABLE: IssueCode = "CLIPBOARD_UNAVAILABLE"
CUDA_UNAVAILABLE: IssueCode = "CUDA_UNAVAILABLE"
MODEL_CACHE_MISSING: IssueCode = "MODEL_CACHE_MISSING"
MODEL_IMPORT_FAILED: IssueCode = "MODEL_IMPORT_FAILED"
MODEL_TRANSCRIBE_FAILED: IssueCode = "MODEL_TRANSCRIBE_FAILED"
WILLOW_AUTH_REQUIRED: IssueCode = "WILLOW_AUTH_REQUIRED"


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1
    RECORDING_BLOCKED = 2
    DEGRADED = 3


@dataclass(slots=True)
class AppError(Exception):
    code: IssueCode
    message: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class AudioError(AppError):
    pass


class ModelError(AppError):
    pass
