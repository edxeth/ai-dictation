#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BRIDGE_URL = os.environ.get("LOCAL_AI_DICTATION_BRIDGE_URL", "http://127.0.0.1:8765")
STATE_HOME = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
STATE_DIR = STATE_HOME / "local-ai-dictation-waybar"
STATE_DIR.mkdir(parents=True, exist_ok=True)
LAST_STATE_PATH = STATE_DIR / "last-state"
LAST_COMPLETED_PATH = STATE_DIR / "last-completed-at"
BACKEND_STATE_PATH = STATE_HOME / "local-ai-dictation" / "backend.json"
ASSET_DIR = Path(os.environ.get("LOCAL_AI_DICTATION_ASSET_DIR", "/nonexistent"))
START_CUE = ASSET_DIR / "session-start.wav"
COMPLETE_CUE = ASSET_DIR / "session-complete.wav"


def preferred_backend() -> str:
    try:
        payload = json.loads(BACKEND_STATE_PATH.read_text(encoding="utf-8"))
        backend = str(payload.get("backend") or "whisper").strip().lower()
    except Exception:
        backend = "whisper"
    return backend if backend in {"whisper", "parakeet", "willow"} else "whisper"


def backend_label(value: str) -> str:
    return {
        "whisper": "Whisper",
        "parakeet": "Parakeet",
        "willow": "Willow",
    }.get(value, "Whisper")


def read_last_state() -> str | None:
    try:
        value = LAST_STATE_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def write_last_state(state: str) -> None:
    LAST_STATE_PATH.write_text(state, encoding="utf-8")


def read_last_completed_at() -> str | None:
    try:
        value = LAST_COMPLETED_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def write_last_completed_at(completed_at: str) -> None:
    LAST_COMPLETED_PATH.write_text(completed_at, encoding="utf-8")


def play_cue(path: Path) -> None:
    if not path.exists():
        return
    candidates = [
        [shutil.which("paplay"), str(path)],
        [shutil.which("pw-play"), str(path)],
        [shutil.which("ffplay"), "-v", "quiet", "-nodisp", "-autoexit", str(path)],
    ]
    for command in candidates:
        executable = command[0]
        if not executable:
            continue
        subprocess.Popen([executable, *command[1:]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return


def active_window_class() -> str:
    try:
        completed = subprocess.run(["hyprctl", "-j", "activewindow"], capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not completed.stdout.strip():
            return ""
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("class") or "")


def send_paste_shortcut() -> None:
    shortcut = "CTRL,V,activewindow"
    if active_window_class() == "com.mitchellh.ghostty":
        shortcut = "CTRL SHIFT,V,activewindow"
    subprocess.run(["hyprctl", "dispatch", "sendshortcut", shortcut], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def classify(payload: dict[str, object] | None) -> tuple[str, str, str, str]:
    preferred = preferred_backend()
    preferred_text = backend_label(preferred)
    module_text = f"󰍬 {preferred_text}"
    if not payload:
        return ("offline", module_text, f"{preferred_text} offline", f"Bridge offline\nPreferred model: {preferred_text}")

    bridge = payload.get("bridge")
    session = payload.get("session")
    if not isinstance(bridge, dict) or not isinstance(session, dict):
        return ("error", module_text, "Local AI Dictation error", "Unexpected bridge payload")

    running_backend = str((session.get("config") or {}).get("backend") if isinstance(session.get("config"), dict) else preferred)
    running_backend = running_backend if running_backend in {"whisper", "parakeet", "willow"} else preferred
    running_text = backend_label(running_backend)
    module_text = f"󰍬 {running_text}"
    state = str(session.get("state") or "idle")
    model_loaded = bool(bridge.get("model_loaded"))
    last_error = session.get("last_error")
    transcript = None
    last_transcript = session.get("last_transcript")
    if isinstance(last_transcript, dict):
        transcript = str(last_transcript.get("transcript") or "").strip() or None

    if state == "recording":
        return (state, module_text, f"{running_text} recording", f"Recording with {running_text}…\nLeft click to stop\nRight click to switch model")
    if state == "starting" or bool(session.get("model_loading")):
        message = "Loading model and preparing to record…" if not model_loaded else "Preparing to record…"
        return ("starting", module_text, f"{running_text} starting", f"{message}\nLeft click: start/stop\nRight click: switch model")
    if state == "transcribing":
        return (state, module_text, f"{running_text} transcribing", f"Transcribing with {running_text}…")
    if state == "error":
        return (state, module_text, f"{running_text} error", str(last_error or "Bridge error"))
    if model_loaded:
        tooltip = f"{running_text} ready\nLeft click: record\nRight click: switch model"
        if transcript:
            tooltip = f"{running_text} ready\nLast transcript:\n{transcript}"
        return ("idle", module_text, f"{running_text} ready", tooltip)
    if preferred == "willow":
        return (
            "idle",
            module_text,
            f"{preferred_text} ready",
            "Ready. First recording will connect to Willow Voice cloud.\nLeft click: record\nRight click: switch model",
        )
    return ("idle", module_text, f"{preferred_text} ready", f"Ready. First recording will load {preferred_text}.\nLeft click: record\nRight click: switch model")


try:
    with urlopen(f"{BRIDGE_URL}/health", timeout=1.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
except (URLError, TimeoutError, OSError, json.JSONDecodeError):
    payload = None

state, icon, text, tooltip = classify(payload)
previous_state = read_last_state()
previous_completed_at = read_last_completed_at()
completed_at = None
paste_dispatched_at = None
transcript_text = ""
if isinstance(payload, dict):
    session = payload.get("session")
    if isinstance(session, dict):
        raw_completed_at = session.get("last_completed_at")
        if raw_completed_at is not None:
            completed_at = str(raw_completed_at)
        paste_dispatched_at = session.get("paste_dispatched_at")
        last_transcript = session.get("last_transcript")
        if isinstance(last_transcript, dict):
            transcript_text = str(last_transcript.get("transcript") or "").strip()

if previous_state != state:
    if state in {"starting", "recording"} and previous_state not in {"starting", "recording"}:
        play_cue(START_CUE)
    elif state == "idle" and previous_state in {"starting", "recording", "transcribing"}:
        play_cue(COMPLETE_CUE)
    write_last_state(state)

if completed_at is not None and completed_at != previous_completed_at:
    write_last_completed_at(completed_at)
    if previous_completed_at is not None and transcript_text and paste_dispatched_at is None:
        send_paste_shortcut()

print(json.dumps({"text": icon, "alt": text, "tooltip": tooltip, "class": state}))
