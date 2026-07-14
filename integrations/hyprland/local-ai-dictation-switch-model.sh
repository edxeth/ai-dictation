#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/local-ai-dictation"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/waybar-switch-model.log"
CLI="${LOCAL_AI_DICTATION_CLI:-$HOME/.local/bin/local-ai-dictation}"

before="$("$CLI" backend get)"
"$CLI" backend toggle --restart-bridge >/dev/null
after="$("$CLI" backend get)"
printf '%s before=%s after=%s\n' "$(date --iso-8601=seconds)" "$before" "$after" >> "$LOG_FILE"

case "$after" in
  whisper) label="Whisper" ;;
  parakeet) label="Parakeet" ;;
  willow) label="Willow Voice" ;;
  *) label="$after" ;;
esac

if command -v notify-send >/dev/null 2>&1; then
  notify-send "Local AI Dictation" "Model switched to $label" >/dev/null 2>&1 || true
fi

pkill -RTMIN+8 waybar 2>/dev/null || true
