#!/usr/bin/env bash
set -euo pipefail

bridge_url="http://127.0.0.1:8765/health"
bridge_launcher="${LOCAL_AI_DICTATION_BRIDGE_LAUNCHER:-$HOME/.local/bin/local-ai-dictation-bridge}"
cli="${LOCAL_AI_DICTATION_CLI:-$HOME/.local/bin/local-ai-dictation}"

if ! curl -fsS "$bridge_url" >/dev/null 2>&1; then
  if [ -x "$bridge_launcher" ]; then
    "$bridge_launcher" >/dev/null 2>&1 &
  fi
  for _ in $(seq 1 40); do
    if curl -fsS "$bridge_url" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

"$cli" bridge-toggle --host 127.0.0.1 --port 8765 >/dev/null
pkill -RTMIN+8 waybar 2>/dev/null || true
