# AI Dictation

AI voice dictation with local and cloud transcription for Linux, plus a desktop GUI and system integrations:

- local **Whisper**, **Parakeet Ultra**, **Parakeet Redux**, **Bark 3.0**, and
  **Bark 3.0 Pro** speech-to-text backends
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
local-ai-dictation backend set parakeet-redux
local-ai-dictation backend set bark
local-ai-dictation backend set bark-pro
local-ai-dictation backend set willow
local-ai-dictation backend toggle --restart-bridge

local-ai-dictation willow-session login
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
- Local GPU or CPU inference
- Model: `deepdml/faster-distil-whisper-large-v3.5`
- The persistent bridge performs a real inference warmup, then overlaps long
  recordings with timestamped rolling windows so stop only finalizes the tail

### Parakeet Ultra

- Model: `moondream/parakeet-ultra`, with CUDA acceleration and CPU fallback
- Uses the direct Parakeet Ultra runtime from `kestrel==0.9.1`, not the Photon client
  or inference-engine wrapper. Those wrappers send machine and usage telemetry;
  this integration does not create them.
- Downloads weights from Hugging Face on first use. A complete cached model
  loads without network access. No Moondream account or API key is required.
- The persistent bridge performs a real inference warmup. Transcription remains
  batch-based and starts after recording stops; live streaming is not enabled.
- The `parakeet` backend key, desktop controls, and hotkeys are unchanged.

### Parakeet Redux, Bark 3.0, and Bark 3.0 Pro

These three backends use the same Kestrel runtime as Parakeet Ultra. They
use the same direct path, without telemetry. They use CUDA when it is
available and the CPU when it is not. The bridge warms up the model when it
starts. Transcription starts after the recording stops.

| Backend key | Model | Languages | Notes |
| --- | --- | --- | --- |
| `parakeet-redux` | `moondream/parakeet-redux` | 25 | The weights are ternary (each weight is -1, 0, or +1). The download is 178 MB. On the CPU, it was about 4 times faster than Parakeet Ultra in testing. On CUDA, Kestrel expands the weights to full size, so it uses about the same GPU memory as Parakeet Ultra. |
| `bark` | `ConwayResearch/Underdog-Bark-3.0` | English | The download is a packed 163 MB file, `model.bark`. On the first load, the app unpacks it one time into a 1.3 GB checkpoint (a folder of model files). The unpack takes one to two minutes. |
| `bark-pro` | `ConwayResearch/Underdog-Bark-3.0-Pro` | English | The download is 1.25 GB of full-precision bf16 weights. |

The app keeps the unpacked Bark 3.0 checkpoint in
`$XDG_CACHE_HOME/local-ai-dictation/models/`. If `XDG_CACHE_HOME` is not set,
the folder is `~/.cache/local-ai-dictation/models/`.

Other facts about these backends:

- The app pins both Bark repositories to the commits that it was tested with.
  New uploads to these repositories do not change the weights. The Moondream
  repositories follow `main`, as Kestrel does.
- Bark is a fine-tune of Parakeet TDT 0.6B v3, so Kestrel loads it as that
  model. Bark 3.0 does not include `tokenizer.json`. The app uses the NVIDIA
  tokenizer from the revision that Kestrel pins. Bark uses this tokenizer
  without changes.
- The app unpacks `model.bark` with its own decoder in
  `src/local_ai_dictation/bark_format.py`. It does not run the Python files in
  the model repository. The unpacked weights are identical, bit for bit, to
  the output of the `bark_loader.py` script from Underdog.
- If two processes load Bark 3.0 for the first time together, only one of
  them unpacks it. The other process waits and then uses the result.
- Bark has no voice activity head (a part of the model that finds pauses).
  For long recordings, Kestrel splits the audio at quiet points that it finds
  from the sound level. Parakeet Ultra and Parakeet Redux find the pauses with
  the model.

The models were tested on 2,308 clips from the English test sets of the Open
ASR Leaderboard, on an RTX 3060. The scores use the text normalizer of the
leaderboard. The average word error rate was 4.95% for Bark 3.0 Pro, 5.26% for
Parakeet Ultra, 5.50% for Bark 3.0, and 6.01% for Parakeet Redux. Bark 3.0 Pro
was better on meeting speech (AMI) and parliament speech (VoxPopuli). On the
other sets, it had the same result as Parakeet Ultra. On noisy speech, Bark
3.0 Pro and Parakeet Ultra had the same result, and both were better than
Bark 3.0 and Parakeet Redux. For languages other than English, use a
Parakeet model.

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

The backend supports Google OAuth through the system browser and also accepts
imported or explicit credentials:

1. Browser-based Google OAuth — recommended for Google-created accounts.
2. An app-owned imported Willow session.
3. `WILLOW_ACCESS_TOKEN` plus optional `WILLOW_USER_ID` — useful for isolated
   or ephemeral environments.

### Sign in with Google OAuth

```bash
local-ai-dictation willow-session login
```

The command opens Willow's normal Google login in the system browser using
Supabase PKCE. After login, copy the complete
`https://willowvoice.com/success-open-app?code=...` URL from the browser address
bar and paste it into the terminal. The callback code is exchanged using the
in-memory PKCE verifier, and only the minimal refreshable session is stored.
If the local bridge is already running, it is restarted automatically so it
loads the new credentials. The official Willow app and Wine are not required.

Treat the callback URL as a short-lived credential: paste it only into the
waiting command, not into chat, logs, or shell history. Each independently
refreshing device must perform its own login; never copy one session file to
multiple active devices.

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
uv pip install torch \
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
Whisper → Parakeet Ultra → Parakeet Redux → Bark 3.0 → Bark 3.0 Pro → Willow Voice → Whisper
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

## Arch Linux + Hyprland + Waybar

The `integrations/` directory contains the setup used by this repository's
primary Arch/Hyprland environment:

```text
integrations/bin/local-ai-dictation-bridge
integrations/hyprland/local-ai-dictation-toggle.sh
integrations/hyprland/local-ai-dictation-switch-model.sh
integrations/hyprland/local-ai-dictation-model-menu.sh
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
install -Dm755 integrations/hyprland/local-ai-dictation-model-menu.sh \
  ~/.config/hypr/scripts/local-ai-dictation-model-menu.sh
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

When the Willow session expires, the hotkey script sends a critical
"Willow session expired" notification instead of a raw bridge error. Clicking
the notification's action copies `local-ai-dictation willow-session login`
to the clipboard so the sign-in command can be pasted straight into a terminal
(requires libnotify >= 0.8, a notification daemon with action support such as
dunst, and wl-clipboard or xclip).

### Waybar configuration

```jsonc
"custom/local_ai_dictation": {
  "format": "{}",
  "return-type": "json",
  "signal": 8,
  "exec": "~/.config/waybar/scripts/local-ai-dictation-status.py",
  "on-click": "~/.config/hypr/scripts/local-ai-dictation-toggle.sh",
  "on-click-right": "~/.config/hypr/scripts/local-ai-dictation-model-menu.sh",
  "tooltip": true
}
```

The model menu uses [Walker](https://github.com/abenz1267/walker) in dmenu
mode (Walker reads the menu lines and returns the selected line). To use a
Walker theme, set `LOCAL_AI_DICTATION_WALKER_THEME` to the theme name. If this
variable is not set, Walker uses the theme from its configuration. To use the
older cycle, which moves to the next backend on each click, set
`on-click-right` to `~/.config/hypr/scripts/local-ai-dictation-switch-model.sh`.

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
- Right click: Walker shows a menu of all backends and marks the current
  backend. If you select a different backend, the app saves it and restarts
  the bridge one time. Only the selected model loads. If you press Escape or
  select the current backend, nothing changes.
- Display: `󰍬 Whisper`, `󰍬 Parakeet Ultra`, `󰍬 Parakeet Redux`,
  `󰍬 Bark 3.0`, `󰍬 Bark 3.0 Pro`, or `󰍬 Willow`.
- Successful completion: copy the transcript, then ask Hyprland to paste into
  the focused application directly from the bridge; Waybar remains the fallback.
- Ghostty uses `Ctrl+Shift+V`; other applications use `Ctrl+V`.
- State changes signal Waybar with `RTMIN+8`; no timer polling is required.
- `/health` remains available for one-shot status and recovery checks.

The selected backend stays active after logout or reboot.
`local-ai-dictation backend toggle` saves the next backend in a fixed order.
`local-ai-dictation backend toggle --restart-bridge` and
`local-ai-dictation-switch-model.sh` also restart the bridge with that backend,
so each step loads a model:

```text
Whisper → Parakeet Ultra → Parakeet Redux → Bark 3.0 → Bark 3.0 Pro → Willow → Whisper
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

- The Kestrel models (Parakeet Ultra, Parakeet Redux, Bark 3.0, and Bark 3.0
  Pro) use about 1.7 GB of GPU memory each for weights and inference buffers.
  Keep free GPU memory for other applications, especially on a 6 GB GPU.
- Kestrel is pinned because this integration uses its exported lower-level
  Parakeet TDT runtime. Before an upgrade, repeat GPU/CPU inference and a cached,
  network-traced run to verify that the direct path still avoids telemetry.
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

Install the development tools into the existing Python environment and install
the locked desktop dependencies:

```bash
uv pip install -e '.[test]' --group dev
(cd desktop/electrobun && bun install --frozen-lockfile)
make install-hooks
make check
```

`make check` runs Ruff lint and format checks, Vulture, deptry, the complete
Python test suite, and the desktop Oxlint, Oxfmt, Knip, test, and build checks.
Run `make format` to format the Python and desktop sources.

The versioned `.githooks/pre-commit` hook blocks a commit if `make check` fails.
It checks the exact staged tree in a temporary directory, not unstaged or
untracked source files. It reuses the installed `.venv` and desktop
`node_modules`, with Python imports directed to the staged source. It does not
stash, auto-format, or stage your working files. Missing dependencies block the
commit rather than skipping checks.

Run `make install-hooks` once per clone. It sets the repository-local
`core.hooksPath` and refuses to replace an existing hook setup. Include the
Makefile, hook files, and check configuration in the first commit of this setup.
Git can still bypass local hooks with `--no-verify`; server-side required checks
are needed to prevent deliberate bypasses.

Vulture's whitelist records HTTP callbacks and serialized fields that are used
indirectly. deptry records runtime packages loaded through `importlib`. Knip
keeps `rcedit` because the Python Windows installer executes its bundled binary.
Review new findings against real callers before adding an exception.

Useful runtime checks:

```bash
local-ai-dictation backend get
local-ai-dictation willow-session status --json
local-ai-dictation doctor --backend willow --check-model-cache --json
curl -fsS http://127.0.0.1:8765/health | python -m json.tool
```
