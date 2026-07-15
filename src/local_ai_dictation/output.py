"""Transcript output rendering and delivery helpers for AI Dictation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, TextIO

from local_ai_dictation.errors import AppError, CLIPBOARD_UNAVAILABLE
from local_ai_dictation.types import DictationConfig, TranscriptionResult


_DICTATION_JSON_SCHEMA_VERSION = 1


def render_transcription(transcription: TranscriptionResult, output_format: str) -> str:
    """Render a transcription result in text or JSON form."""
    if output_format == "json":
        payload: dict[str, object] = {
            "schema_version": _DICTATION_JSON_SCHEMA_VERSION,
            "transcript": transcription.text,
        }
        if transcription.normalized_text is not None:
            payload["normalized_transcript"] = transcription.normalized_text
        if transcription.device is not None:
            payload["device"] = transcription.device
        if transcription.metadata:
            payload["metadata"] = transcription.metadata
        return json.dumps(payload)

    return transcription.text


def write_output_file(path: str, rendered_output: str) -> None:
    Path(path).write_text(f"{rendered_output}\n", encoding="utf-8")


def copy_transcript_to_clipboard(
    transcription: TranscriptionResult,
    pyperclip_module: Any,
) -> AppError | None:
    try:
        pyperclip_module.copy(transcription.text)
    except Exception as exc:
        return AppError(CLIPBOARD_UNAVAILABLE, str(exc))
    return None


def emit_transcription_result(
    transcription: TranscriptionResult,
    config: DictationConfig,
    *,
    pyperclip_module: Any | None = None,
    stdout: TextIO | None = None,
    status_stream: TextIO | None = None,
) -> AppError | None:
    rendered_output = render_transcription(transcription, config.format)

    print(rendered_output, file=stdout or sys.stdout)

    if config.output_file:
        write_output_file(config.output_file, rendered_output)

    clipboard_warning: AppError | None = None
    if config.clipboard and pyperclip_module is not None:
        clipboard_warning = copy_transcript_to_clipboard(transcription, pyperclip_module)
        if clipboard_warning is not None and config.debug:
            print(f"Clipboard warning: {clipboard_warning}", file=status_stream or sys.stderr)

    return clipboard_warning
