# AI Dictation

AI voice dictation with local and cloud transcription for Linux, plus a desktop GUI and system integrations:

- local **Whisper** and **Parakeet** speech-to-text backends
- the **Willow Voice** cloud backend
- a persistent localhost bridge
- a native Electrobun control GUI
- Hyprland hotkeys and a push-updated Waybar module

The factory default is Whisper. After you select a backend, the choice is
persisted in `~/.local/state/local-ai-dictation/backend.json`. That persisted
choice becomes the effective default for the bridge, GUI, Hyprland shortcut,
and Waybar. If Waybar currently displays `Willow`, Willow is what the next
recording uses.

## Main commands

```bash
local-ai-dictation dictation
local-ai-dictation bridge
local-ai-dictation bridge-toggle
local-ai-dictation gui

local-ai-dictation backend get
local-ai-dictation backend set whisper
local-ai-dictation backend set parakeet
local-ai-dictation backend set willow
local-ai-dictation backend toggle --restart-bridge

local-ai-dictation willow-session status --json
local-ai-dictation devices --json
local-ai-dictation doctor --json
local-ai-dictation doctor --backend willow --check-model-cache --json

local-ai-dictation benchmark --backend whisper --fixture recording.wav --runs 1 --json
local-ai-dictation benchmark --backend parakeet --fixture recording.wav --runs 1 --json
local-ai-dictation benchmark --backend willow --fixture recording.wav --runs 1 --json
```

## Backends

### Whisper

- Factory default
- Lower local VRAM pressure than Parakeet
- Model: `deepdml/faster-distil-whisper-large-v3.5`
- The persistent bridge performs a real inference warmup, then overlaps long
  recordings with timestamped rolling windows so stop only finalizes the tail

### Parakeet

- Local NVIDIA NeMo inference
- Higher model-load VRAM spike
- Model: `nvidia/parakeet-tdt-0.6b-v3`
- The persistent bridge performs a real inference warmup; this offline
  checkpoint stays on batch inference because cache-aware streaming needs
  multi-second context

### Willow Voice cloud

- Sends audio only during an active transcription request
- Uses Willow's production binary MessagePack WebSocket protocol
- Converts input to 16 kHz, mono, signed 16-bit PCM
- Sends 512-sample audio packets, followed by a flush packet
- Lets Willow select Frontier Mini or Frontier Pro according to the account
  entitlement; this client does not spoof or select a paid model

Willow is not a downloadable model. It is a cloud service and requires a valid
Willow account plus a refreshable Supabase session.

## Willow authentication

The backend does **not** provide an interactive Willow login screen. It accepts
one of these credentials:

1. An app-owned imported Willow session — recommended.
2. `WILLOW_ACCESS_TOKEN` plus optional `WILLOW_USER_ID` — useful for isolated
   or ephemeral environments.

### Import an official Willow session

Authenticate once in an official Willow application, find its
`supabase-session.json`, then import it:

```bash
local-ai-dictation willow-session import /path/to/supabase-session.json
local-ai-dictation willow-session status --json
local-ai-dictation backend set willow --restart-bridge
```

The importer supports both email/password and OAuth-created Willow accounts.
It deliberately retains only:

- access token
- refresh token
- expiry timestamp
- Willow user ID

OAuth provider tokens, profile fields, email addresses, and unrelated session
metadata are discarded. The canonical session is written to:

```text
~/.local/state/local-ai-dictation/willow-session.json
```

The state directory is mode `0700`; the session is mode `0600`. Access tokens
are refreshed automatically before expiry, rotated refresh credentials are
persisted atomically, and tokens never enter bridge responses, GUI RPC data, or
Waybar output.

Typical official Electron session locations include:

```text
# Windows
%APPDATA%\willow-voice\supabase-session.json

# macOS
~/Library/Application Support/willow-voice/supabase-session.json

# Official Windows app running through Wine
$WINEPREFIX/drive_c/users/$USER/AppData/Roaming/willow-voice/supabase-session.json

# The same Windows file from WSL
/mnt/c/Users/<windows-user>/AppData/Roaming/willow-voice/supabase-session.json
```

A legacy `~/.local/state/willow-linux/supabase-session.json` file is migrated
into the canonical minimal format automatically on first use.

If Willow revokes the refresh token or changes its authentication system,
reauthenticate in an official application and run `willow-session import`
again. The official app and unofficial clients are not needed during normal
operation after a successful import.

## Install

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install the appropriate PyTorch build for your hardware.
uv pip install torch torchaudio torchvision \
  --index-url https://download.pytorch.org/whl/cu130

uv pip install -e '.[test]'
```

Common Linux system dependencies:

- `uv`
- `bun`
- PortAudio / `pyaudio` build dependencies
- PipeWire or PulseAudio capture utilities
- `wl-clipboard` or `xclip`
- WebKitGTK and Electrobun desktop dependencies

Create stable user commands if the repository is installed from a working
checkout:

```bash
install -Dm755 /dev/stdin ~/.local/bin/local-ai-dictation <<EOF
#!/bin/sh
exec "$PWD/.venv/bin/local-ai-dictation" "\$@"
EOF

install -Dm755 integrations/bin/local-ai-dictation-bridge \
  ~/.local/bin/local-ai-dictation-bridge
```

## Configuration

Configuration file:

```text
~/.config/local-ai-dictation/config.toml
```

Example:

```toml
backend = "whisper"
cpu = false
input_device = ""
vad = false
max_silence_ms = 1200
min_speech_ms = 300
vad_mode = 2
format = "text"
output_file = ""
clipboard = true
debug = false
```

Precedence is CLI arguments over environment variables over the TOML file over
the persisted backend/defaults.

Supported environment variables:

- `LOCAL_AI_DICTATION_BACKEND`
- `LOCAL_AI_DICTATION_CPU`
- `LOCAL_AI_DICTATION_INPUT_DEVICE`
- `LOCAL_AI_DICTATION_VAD`
- `LOCAL_AI_DICTATION_MAX_SILENCE_MS`
- `LOCAL_AI_DICTATION_MIN_SPEECH_MS`
- `LOCAL_AI_DICTATION_VAD_MODE`
- `LOCAL_AI_DICTATION_FORMAT`
- `LOCAL_AI_DICTATION_OUTPUT_FILE`
- `LOCAL_AI_DICTATION_CLIPBOARD`
- `LOCAL_AI_DICTATION_DEBUG`
- `LOCAL_AI_DICTATION_BRIDGE_URL`
- `LOCAL_AI_DICTATION_WAYBAR_SIGNAL`
- `LOCAL_AI_DICTATION_HYPRLAND_PASTE` — dispatch paste directly from the bridge after clipboard copy
- `LOCAL_AI_DICTATION_RETAIN_AUDIO` — opt-in private diagnostic WAV retention
- `WILLOW_ACCESS_TOKEN`
- `WILLOW_USER_ID`
- `WILLOW_TRANSCRIBE_URL` — test/development override

## GUI and application launcher

Run the GUI directly:

```bash
local-ai-dictation gui
```

The GUI displays the running backend, bridge command, session state,
transcription history, errors, and a model switch control that cycles:

```text
Whisper → Parakeet → Willow → Whisper
```

Switching backend restarts the bridge so local model VRAM is released. The GUI
subscribes to `/events` using Server-Sent Events rather than polling bridge
state.

Install the desktop entry for Walker, Elephant, or another XDG launcher:

```bash
install -Dm644 integrations/applications/local-ai-dictation.desktop \
  ~/.local/share/applications/local-ai-dictation.desktop
update-desktop-database ~/.local/share/applications
```

If `local-ai-dictation` is not available in the graphical session's `PATH`,
replace the desktop entry's `Exec` and `TryExec` values with the absolute path
to `~/.local/bin/local-ai-dictation`.

The current application name is **AI Dictation**. Old launcher entries
named **Parakeet Dictation** or `parakeet-desktop` are obsolete build artifacts,
not a separate application. Remove stale desktop entries and restart the
launcher provider, for example:

```bash
systemctl --user restart elephant.service
```

If the obsolete name survives only in Elephant's launch history, reset that
single provider cache with a backup first:

```bash
history="$HOME/.cache/elephant/desktopapplications_history.gob"
backup_dir="$HOME/.local/state/local-ai-dictation"
backup="$backup_dir/desktopapplications_history.gob.backup"
install -d -m 0700 "$backup_dir"
systemctl --user stop elephant.service
if [ -f "$history" ]; then
  cp -a "$history" "$backup" && rm -f "$history"
fi
systemctl --user start elephant.service
```

This resets application ranking/history in Walker but does not remove desktop
entries or application data. Restore the backup if needed.

## Arch Linux + Hyprland + Waybar

The `integrations/` directory contains the setup used by this repository's
primary Arch/Hyprland environment:

```text
integrations/bin/local-ai-dictation-bridge
integrations/hyprland/local-ai-dictation-toggle.sh
integrations/hyprland/local-ai-dictation-switch-model.sh
integrations/waybar/local-ai-dictation-status.py
integrations/applications/local-ai-dictation.desktop
```

Install the scripts:

```bash
install -Dm755 integrations/bin/local-ai-dictation-bridge \
  ~/.local/bin/local-ai-dictation-bridge
install -Dm755 integrations/hyprland/local-ai-dictation-toggle.sh \
  ~/.config/hypr/scripts/local-ai-dictation-toggle.sh
install -Dm755 integrations/hyprland/local-ai-dictation-switch-model.sh \
  ~/.config/hypr/scripts/local-ai-dictation-switch-model.sh
install -Dm755 integrations/waybar/local-ai-dictation-status.py \
  ~/.config/waybar/scripts/local-ai-dictation-status.py
```

Optional cue sounds can be enabled for the Waybar script by exporting the
asset directory before Waybar starts:

```bash
export LOCAL_AI_DICTATION_ASSET_DIR="$PWD/desktop/electrobun/src/mainview/assets"
```

### Hyprland configuration

```ini
# Start one persistent bridge when the session starts.
exec-once = ~/.local/bin/local-ai-dictation-bridge &

# Global start/stop recording shortcut.
bind = SUPER, R, exec, ~/.config/hypr/scripts/local-ai-dictation-toggle.sh
```

The compositor shortcut is the reliable global activation path on Wayland.
The Electrobun window's `Ctrl+Alt+R` fallback works only while the window is
focused unless a desktop portal/global-shortcut implementation is available.

### Waybar configuration

```jsonc
"custom/local_ai_dictation": {
  "format": "{}",
  "return-type": "json",
  "signal": 8,
  "exec": "~/.config/waybar/scripts/local-ai-dictation-status.py",
  "on-click": "~/.config/hypr/scripts/local-ai-dictation-toggle.sh",
  "on-click-right": "~/.config/hypr/scripts/local-ai-dictation-switch-model.sh",
  "tooltip": true
}
```

Add `custom/local_ai_dictation` to a Waybar module list. Example styling:

```css
#custom-local_ai_dictation {
  padding-left: 7px;
  padding-right: 7px;
}

#custom-local_ai_dictation.idle {
  color: @text;
}

#custom-local_ai_dictation.recording,
#custom-local_ai_dictation.transcribing {
  color: @accent;
}

#custom-local_ai_dictation.offline {
  color: @subtext;
}

#custom-local_ai_dictation.error {
  color: @red;
}
```

Behavior:

- Left click or `Super+R`: start/stop recording.
- Right click: persist the next backend and restart the bridge.
- Display: `󰍬 Whisper`, `󰍬 Parakeet`, or `󰍬 Willow`.
- Successful completion: copy the transcript, then ask Hyprland to paste into
  the focused application directly from the bridge; Waybar remains the fallback.
- Ghostty uses `Ctrl+Shift+V`; other applications use `Ctrl+V`.
- State changes signal Waybar with `RTMIN+8`; no timer polling is required.
- `/health` remains available for one-shot status and recovery checks.

The switch order is deterministic, and the selected backend remains active
after logout or reboot:

```text
Whisper → Parakeet → Willow → Whisper
```

## Omarchy and other Hyprland distributions

Omarchy is an Arch/Hyprland environment, so the same bridge, Hyprland, Waybar,
and XDG desktop-entry examples apply. Adjust only:

- absolute repository/virtual-environment paths
- Waybar module placement and theme color names
- the preferred compositor key binding
- audio input routing
- clipboard/paste tools

Coding agents can treat `integrations/` as a working reference rather than
inventing a new state model. Keep the backend state file, bridge HTTP contract,
and `RTMIN+8` notification mechanism intact while adapting paths and visuals.

## WSL

A useful WSL arrangement is:

- run the Python bridge and model backend inside WSL
- expose only the localhost bridge port
- use WSLg/PulseAudio or a configured capture device
- run the Electrobun Windows package on Windows if a native Windows control
  surface is desired
- import a Windows Willow session through `/mnt/c/...` when using Willow

Examples:

```bash
local-ai-dictation doctor --json
local-ai-dictation devices --json
local-ai-dictation bridge --host 127.0.0.1 --port 8765
```

Hyprland and Waybar files are Linux-desktop examples, not requirements for WSL.
Use Windows shortcuts, tray controls, or another client that calls the same
bridge endpoints.

## Bridge API

The bridge binds to localhost by default and exposes:

- `GET /health`
- `GET /session`
- `GET /events` — SSE state stream
- `GET /devices`
- `GET /doctor`
- `POST /session/start`
- `POST /session/stop`
- `POST /session/toggle`
- `POST /session/clear-history`

Do not expose the bridge to an untrusted network without adding authentication
and transport security.

## Operational notes

- Whisper is generally safer than Parakeet on a 6 GB GPU.
- Parakeet has a larger transient model-load VRAM spike.
- Willow uses cloud processing and does not consume local model VRAM.
- Switching backend restarts the bridge and releases the previous local model.
- The Willow session refreshes without the official Willow application being
  installed.
- Removing the canonical Willow session signs this application out locally but
  does not delete the Willow account.
- The bridge does not retain recordings by default. Set
  `LOCAL_AI_DICTATION_RETAIN_AUDIO=1` only for explicit diagnostics; retained
  WAV files are stored under the app's private state directory with mode
  `0600`.
- Willow requests send active transcription audio to Willow's service.

## Verification

Run the complete local checks:

```bash
python -m compileall src tests
.venv/bin/python -m pytest -q
cd desktop/electrobun && bun run check
```

Useful runtime checks:

```bash
local-ai-dictation backend get
local-ai-dictation willow-session status --json
local-ai-dictation doctor --backend willow --check-model-cache --json
curl -fsS http://127.0.0.1:8765/health | python -m json.tool
```
