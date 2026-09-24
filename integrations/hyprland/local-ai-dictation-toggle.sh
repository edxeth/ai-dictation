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

health_curl_max_time="${LOCAL_AI_DICTATION_HEALTH_CURL_MAX_TIME:-2}"
bridge_available=0
if ! curl --max-time "$health_curl_max_time" -fsS "$bridge_url/health" >/dev/null 2>&1; then
  if [ -x "$bridge_launcher" ]; then
    "$bridge_launcher" >/dev/null 2>&1 &
  fi
  start_attempts="${LOCAL_AI_DICTATION_BRIDGE_START_ATTEMPTS:-40}"
  for _ in $(seq 1 "$start_attempts"); do
    if curl --max-time "$health_curl_max_time" -fsS "$bridge_url/health" >/dev/null 2>&1; then
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
    health="$(curl --max-time "$health_curl_max_time" -fsS "$bridge_url/health" 2>/dev/null || true)"
    if [ -n "$health" ] && ! printf '%s' "$health" | grep -Eq '"model_loading"[[:space:]]*:[[:space:]]*true'; then
      break
    fi
    sleep 0.25
  done
fi

toggle_curl_max_time="${LOCAL_AI_DICTATION_TOGGLE_CURL_MAX_TIME:-25}"
curl_status=0
response="$(curl --max-time "$toggle_curl_max_time" --fail-with-body -sS --request POST "$bridge_url/session/toggle" 2>&1)" || curl_status=$?
if [ "$curl_status" -ne 0 ]; then
  compact_response="${response//$'\n'/ }"
  printf '%s toggle failed: %s\n' "$(date --iso-8601=seconds)" "$compact_response" >> "$log_file"
  if printf '%s' "$response" | grep -Eq 'still transcribing|still winding down'; then
    printf '%s\n' "$response" >&2
    exit 1
  fi
  if command -v notify-send >/dev/null 2>&1; then
    if printf '%s' "$response" | grep -Eq '"auth_required"[[:space:]]*:[[:space:]]*true'; then
      notify_willow_session_expired >/dev/null 2>&1 &
    elif [ "$curl_status" -eq 28 ]; then
      notify-send "AI Dictation" "The recording took too long and was stopped. Press Super+R to try again, or switch the model." >/dev/null 2>&1 || true
    else
      notify-send "AI Dictation" "The connection to Willow failed, so the text was not pasted. Press Super+R to try again, or switch the model." >/dev/null 2>&1 || true
    fi
  fi
  printf '%s\n' "$response" >&2
  exit 1
fi
