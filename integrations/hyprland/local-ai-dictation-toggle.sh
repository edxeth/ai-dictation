#!/usr/bin/env bash
set -euo pipefail

bridge_url="${LOCAL_AI_DICTATION_BRIDGE_URL:-http://127.0.0.1:8765}"
bridge_launcher="${LOCAL_AI_DICTATION_BRIDGE_LAUNCHER:-$HOME/.local/bin/local-ai-dictation-bridge}"
log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/local-ai-dictation"
log_file="$log_dir/hotkey.log"
login_command='local-ai-dictation willow-session login'

mkdir -p "$log_dir"

copy_login_command() {
  if command -v wl-copy >/dev/null 2>&1; then
    printf '%s' "$login_command" | wl-copy
  elif command -v xclip >/dev/null 2>&1; then
    printf '%s' "$login_command" | xclip -selection clipboard
  elif command -v xsel >/dev/null 2>&1; then
    printf '%s' "$login_command" | xsel --clipboard --input
  else
    return 1
  fi
}

notify_willow_session_expired() {
  # libnotify >= 0.8 prints the clicked action key on stdout; clicking the
  # action copies the sign-in command straight to the clipboard.
  if ! action="$(notify-send \
    --app-name "AI Dictation" \
    --urgency critical \
    --expire-time 60000 \
    --action "copy=Copy sign-in command" \
    "Willow session expired" \
    "Sign in again. Click to copy: $login_command" 2>/dev/null)"; then
    notify-send "AI Dictation" "Willow session expired. Run: $login_command" >/dev/null 2>&1 || true
    return 0
  fi
  if [ "$action" = "copy" ]; then
    copy_login_command || true
  fi
}

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
    if printf '%s' "$response" | grep -Eq '"auth_required"[[:space:]]*:[[:space:]]*true'; then
      notify_willow_session_expired >/dev/null 2>&1 &
    else
      notify-send "AI Dictation" "Recording failed: ${compact_response:0:240}" >/dev/null 2>&1 || true
    fi
  fi
  printf '%s\n' "$response" >&2
  exit 1
fi
