#!/usr/bin/env bash
set -euo pipefail

bridge_url="${LOCAL_AI_DICTATION_BRIDGE_URL:-http://127.0.0.1:8765}"
bridge_launcher="${LOCAL_AI_DICTATION_BRIDGE_LAUNCHER:-$HOME/.local/bin/local-ai-dictation-bridge}"

if ! curl -fsS "$bridge_url/health" >/dev/null 2>&1; then
  if [ -x "$bridge_launcher" ]; then
    "$bridge_launcher" >/dev/null 2>&1 &
  fi
  for _ in $(seq 1 40); do
    if curl -fsS "$bridge_url/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

curl -fsS --request POST "$bridge_url/session/toggle" >/dev/null
