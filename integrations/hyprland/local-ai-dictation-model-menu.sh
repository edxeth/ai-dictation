#!/usr/bin/env bash
# Pick the dictation model from a Walker menu and switch to it in one step.
# Cycling with backend toggle restarts the bridge, and loads a model, for every
# model passed on the way; this restarts once and loads only the chosen model.
set -euo pipefail

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/local-ai-dictation"
mkdir -p "$STATE_DIR"
LOG_FILE="$STATE_DIR/waybar-switch-model.log"
BACKEND_FILE="$STATE_DIR/backend.json"
CLI="${LOCAL_AI_DICTATION_CLI:-$HOME/.local/bin/local-ai-dictation}"

# Keys, order and labels match BACKEND_LABELS in
# src/local_ai_dictation/backend_state.py.
keys=(whisper parakeet parakeet-redux bark bark-pro willow)
labels=("Whisper" "Parakeet Ultra" "Parakeet Redux" "Bark 3.0" "Bark 3.0 Pro" "Willow")
hints=(
  "Distil large-v3.5"
  "25 languages"
  "25 languages · less accurate"
  "English · less accurate"
  "English · most accurate"
  "Cloud"
)

notify() {
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "AI Dictation" "$1" >/dev/null 2>&1 || true
  fi
}

# Read the saved choice directly: starting the Python CLI first would delay the
# menu. A missing file means the default backend, as in backend_state.py.
current="whisper"
if [ -f "$BACKEND_FILE" ]; then
  current="$(sed -n 's/.*"backend"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$BACKEND_FILE" | head -n 1)"
fi

lines=()
for i in "${!keys[@]}"; do
  if [ "${keys[$i]}" = "$current" ]; then
    lines+=("● ${labels[$i]} · current")
  else
    lines+=("󰍬 ${labels[$i]} · ${hints[$i]}")
  fi
done

walker_args=(-d -e -p "Dictation model")
if [ -n "${LOCAL_AI_DICTATION_WALKER_THEME:-}" ]; then
  walker_args+=(-t "$LOCAL_AI_DICTATION_WALKER_THEME")
fi
# Walker exits non-zero when the menu is dismissed; that is not an error.
choice="$(printf '%s\n' "${lines[@]}" | walker "${walker_args[@]}" 2>/dev/null || true)"

target=""
label=""
for i in "${!keys[@]}"; do
  if [ "$choice" = "${lines[$i]}" ]; then
    target="${keys[$i]}"
    label="${labels[$i]}"
  fi
done
if [ -z "$target" ] || [ "$target" = "$current" ]; then
  exit 0
fi

status=0
"$CLI" backend set "$target" --restart-bridge >/dev/null 2>>"$LOG_FILE" || status=$?
printf '%s before=%s after=%s status=%s\n' "$(date --iso-8601=seconds)" "$current" "$target" "$status" >> "$LOG_FILE"
if [ "$status" -eq 0 ]; then
  notify "Model switched to $label"
else
  notify "Could not switch to $label. See $LOG_FILE"
fi

pkill -RTMIN+8 waybar 2>/dev/null || true
exit "$status"
