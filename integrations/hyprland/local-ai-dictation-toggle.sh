#!/usr/bin/env bash
set -euo pipefail

bridge_url="${LOCAL_AI_DICTATION_BRIDGE_URL:-http://127.0.0.1:8765}"
bridge_launcher="${LOCAL_AI_DICTATION_BRIDGE_LAUNCHER:-$HOME/.local/bin/local-ai-dictation-bridge}"
log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/local-ai-dictation"
log_file="$log_dir/hotkey.log"

mkdir -p "$log_dir"

bridge_available=0
if ! curl -fsS "$bridge_url/health" >/dev/null 2>&1; then
  if [ -x "$bridge_launcher" ]; then
    "$bridge_launcher" >/dev/null 2>&1 &
  fi
  for _ in $(seq 1 40); do
    if curl -fsS "$bridge_url/health" >/dev/null 2>&1; then
      bridge_available=1
      break
    fi
    sleep 0.25
  done
else
  bridge_available=1
fi

if [ "$bridge_available" -eq 1 ]; then
  for _ in $(seq 1 480); do
    health="$(curl -fsS "$bridge_url/health" 2>/dev/null || true)"
    if [ -n "$health" ] && ! printf '%s' "$health" | grep -Eq '"model_loading"[[:space:]]*:[[:space:]]*true'; then
      break
    fi
    sleep 0.25
  done
fi

if ! response="$(curl --fail-with-body -sS --request POST "$bridge_url/session/toggle" 2>&1)"; then
  compact_response="${response//$'\n'/ }"
  printf '%s toggle failed: %s\n' "$(date --iso-8601=seconds)" "$compact_response" >> "$log_file"
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "AI Dictation" "Recording failed: ${compact_response:0:240}" >/dev/null 2>&1 || true
  fi
  printf '%s\n' "$response" >&2
  exit 1
fi
