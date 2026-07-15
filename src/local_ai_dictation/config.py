"""Configuration parsing and precedence resolution for AI Dictation."""

from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from local_ai_dictation.backend_state import get_backend
from local_ai_dictation.types import DictationConfig, TranscriptFormat

try:  # pragma: no cover - exercised indirectly on Python 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


CONFIG_PATH = Path.home() / ".config" / "local-ai-dictation" / "config.toml"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_optional_str(value: Any) -> str | None:
    text = str(value).strip()
    return text or None


def _parse_optional_device(value: Any) -> int | str | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _parse_format(value: Any) -> TranscriptFormat:
    text = str(value).strip().lower()
    if text not in {"text", "json"}:
        raise ValueError(f"Invalid transcript format: {value!r}")
    return text  # type: ignore[return-value]


def _parse_backend(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in {"parakeet", "whisper", "willow"}:
        raise ValueError(f"Invalid backend: {value!r}")
    return text


FIELD_PARSERS: dict[str, Callable[[Any], Any]] = {
    "backend": _parse_backend,
    "cpu": _parse_bool,
    "input_device": _parse_optional_device,
    "vad": _parse_bool,
    "max_silence_ms": int,
    "min_speech_ms": int,
    "vad_mode": int,
    "format": _parse_format,
    "output_file": _parse_optional_str,
    "clipboard": _parse_bool,
    "debug": _parse_bool,
}

ENV_FIELD_MAP = {
    "LOCAL_AI_DICTATION_BACKEND": "backend",
    "LOCAL_AI_DICTATION_CPU": "cpu",
    "LOCAL_AI_DICTATION_INPUT_DEVICE": "input_device",
    "LOCAL_AI_DICTATION_VAD": "vad",
    "LOCAL_AI_DICTATION_MAX_SILENCE_MS": "max_silence_ms",
    "LOCAL_AI_DICTATION_MIN_SPEECH_MS": "min_speech_ms",
    "LOCAL_AI_DICTATION_VAD_MODE": "vad_mode",
    "LOCAL_AI_DICTATION_FORMAT": "format",
    "LOCAL_AI_DICTATION_OUTPUT_FILE": "output_file",
    "LOCAL_AI_DICTATION_CLIPBOARD": "clipboard",
    "LOCAL_AI_DICTATION_DEBUG": "debug",
}

DEFAULTS: dict[str, Any] = {
    "backend": "whisper",
    "cpu": False,
    "input_device": None,
    "vad": False,
    "max_silence_ms": 1200,
    "min_speech_ms": 300,
    "vad_mode": 2,
    "format": "text",
    "output_file": None,
    "clipboard": True,
    "debug": False,
}


def _normalize_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, parser in FIELD_PARSERS.items():
        if key not in values:
            continue
        raw_value = values[key]
        if raw_value is None:
            continue
        normalized[key] = parser(raw_value)
    return normalized


def load_config_file(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return {}

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level table: {config_path}")

    return _normalize_mapping(data)


def load_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = os.environ if env is None else env
    values: dict[str, Any] = {}
    for env_name, field_name in ENV_FIELD_MAP.items():
        if env_name not in source:
            continue
        values[field_name] = source[env_name]
    return _normalize_mapping(values)


def load_cli_overrides(namespace: Namespace) -> dict[str, Any]:
    values = {
        "backend": getattr(namespace, "backend", None),
        "cpu": getattr(namespace, "cpu", None),
        "input_device": getattr(namespace, "input_device", None),
        "vad": getattr(namespace, "vad", None),
        "max_silence_ms": getattr(namespace, "max_silence_ms", None),
        "min_speech_ms": getattr(namespace, "min_speech_ms", None),
        "vad_mode": getattr(namespace, "vad_mode", None),
        "format": getattr(namespace, "format", None),
        "output_file": getattr(namespace, "output_file", None),
        "clipboard": getattr(namespace, "clipboard", None),
        "debug": getattr(namespace, "debug", None),
    }
    return _normalize_mapping(values)


def resolve_config(
    namespace: Namespace,
    *,
    env: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> DictationConfig:
    resolved = dict(DEFAULTS)
    resolved["backend"] = get_backend(env)
    resolved.update(load_config_file(config_path))
    resolved.update(load_env(env))
    resolved.update(load_cli_overrides(namespace))

    log_file = str(getattr(namespace, "log_file", "transcriber.debug.log"))
    list_devices = bool(getattr(namespace, "list_devices", False))

    return DictationConfig(
        backend=resolved["backend"],
        cpu=bool(resolved["cpu"]),
        input_device=resolved["input_device"],
        vad=bool(resolved["vad"]),
        max_silence_ms=int(resolved["max_silence_ms"]),
        min_speech_ms=int(resolved["min_speech_ms"]),
        vad_mode=int(resolved["vad_mode"]),
        format=resolved["format"],
        output_file=resolved["output_file"],
        clipboard=bool(resolved["clipboard"]),
        debug=bool(resolved["debug"]),
        log_file=log_file,
        list_devices=list_devices,
    )
