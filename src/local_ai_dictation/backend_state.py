"""Persistent backend selection for local UI integrations such as Waybar."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_BACKEND = "whisper"
BACKEND_ORDER = ("whisper", "parakeet", "willow")
_VALID_BACKENDS = set(BACKEND_ORDER)


def normalize_backend(value: str | None) -> str:
    text = (value or DEFAULT_BACKEND).strip().lower()
    if text not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend: {value!r}")
    return text


def state_path(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    xdg_state_home = source.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "local-ai-dictation" / "backend.json"
    home = source.get("HOME")
    if home:
        return Path(home) / ".local" / "state" / "local-ai-dictation" / "backend.json"
    return Path("/tmp/local-ai-dictation-backend.json")


def get_backend(env: Mapping[str, str] | None = None) -> str:
    path = state_path(env)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_BACKEND
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_BACKEND
    return normalize_backend(payload.get("backend")) if isinstance(payload, dict) else DEFAULT_BACKEND


def set_backend(backend: str, env: Mapping[str, str] | None = None) -> str:
    normalized = normalize_backend(backend)
    path = state_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"backend": normalized}) + "\n", encoding="utf-8")
    return normalized


def toggle_backend(env: Mapping[str, str] | None = None) -> str:
    current = get_backend(env)
    current_index = BACKEND_ORDER.index(current)
    target = BACKEND_ORDER[(current_index + 1) % len(BACKEND_ORDER)]
    return set_backend(target, env)


def backend_payload(backend: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {
        "backend": normalize_backend(backend),
        "state_path": str(state_path(env)),
    }
