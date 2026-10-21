# Changelog

All notable changes to AI Dictation are recorded here. Versions follow [Semantic Versioning](https://semver.org/). Each entry names the commits it covers.

The command, package, configuration, and state names stay `local-ai-dictation` in every version, so existing installs keep working.

## [Unreleased]

## [1.2.1] - 2026-10-21

### 🐛 Bug Fixes

#### Willow recordings stop when Willow ends the session
Refs: `27ff3e3`

Fixed Willow recordings that stayed active after Willow closed the streaming session (`1000 (OK) Session complete`). The bridge kept recording and discarded all later audio. A recording left running looked active in Waybar for hours. When it was stopped, the bridge reported `MODEL_TRANSCRIBE_FAILED` and pasted nothing.

The bridge now stops capture as soon as a connected Willow session ends without a stop request, whether Willow closed it or the network dropped. It logs `Willow ended the session; recording stopped`. Silence is discarded, and the bridge returns to idle. Audio that may contain speech goes through the normal stop path. After a network failure, the bridge retries with the captured audio. After a `Session complete` close, it reports an error and does not retry, so background noise from a forgotten recording is never pasted.

A failed connection at the start of a recording still does not stop capture. The bridge keeps recording and replays the audio into a new session, as before.

## [1.2.0] - 2026-10-04

### ✨ Features

#### Parakeet Ultra replaces the original local Parakeet model
Refs: `ee54386`

The local Parakeet backend now uses `moondream/parakeet-ultra` through Kestrel's direct runtime. It uses CUDA when available and falls back to CPU. The runtime does not construct Photon's telemetry-reporting wrappers. No Moondream account is required, and a complete cached model works offline.

The GUI, Waybar widget, and model-switch notifications now say **Parakeet Ultra**. The `parakeet` backend key and command are unchanged. First use downloads the new weights; the original NVIDIA Parakeet cache is not reused.

#### Git checks the staged changes before a commit
Refs: `ee54386`

Run `make install-hooks` once per clone to enable the versioned pre-commit hook. It runs `make check` against the staged snapshot and leaves unstaged work untouched. Failed checks or missing development dependencies block the commit.

### 🐛 Bug Fixes

#### Willow sign-in guidance and bounded connection recovery
Refs: `d5cf394`, `272f6f1`

Expired or rejected Willow sessions now return `WILLOW_AUTH_REQUIRED`. The session and health payloads report `auth_required` and `last_error_code`. The Hyprland hotkey shows a "Willow session expired" notification, and its action copies `local-ai-dictation willow-session login` to the clipboard. The login prompt now shows the pasted callback URL, so you can check it before you submit it.

Transport failures now get up to two retries, after 1 second and then 2 seconds. Each retry replays the captured audio. This includes a missing close frame and a DNS name-lookup failure. Willow finalization stops 20 seconds after you end the recording. The worker is then released, so the next recording can start. This deadline does not apply to Whisper. An HTTP 401 or 403 socket rejection is treated as a sign-in failure, not a transport retry. Every hotkey request has a time limit, and failures show a plain sentence instead of raw JSON.

#### The desktop GUI respects the saved model selection
Refs: `ee54386`

The GUI no longer falls back to Whisper without notice when Parakeet Ultra or Willow is the saved backend.

### ♻️ Refactoring

#### Obsolete code removed; cleanup checks are repeatable
Refs: `ee54386`

Removed unused model adapters, inactive desktop UI and audio paths, obsolete NeMo documentation, and unused dependencies. Python and desktop sources are formatted. Ruff, Vulture, deptry, Oxlint, Oxfmt, and Knip are configured with narrow exceptions. `make format` formats sources, and `make check` runs the complete quality gate.

### ⚠️ Compatibility

Python support is limited to **3.10–3.12**; the documented setup uses 3.12. Refresh dependencies after you upgrade, because the Parakeet backend now uses Kestrel instead of NeMo.

## [1.1.2] - 2026-09-03

### 🐛 Bug Fixes

#### NVIDIA GPUs in integrated mode stay powered down
Refs: `514b33f`

AI Dictation reads `/proc/modules` to check whether the NVIDIA kernel driver is loaded before it calls `torch.cuda`. When the driver is intentionally unloaded, as on laptops in integrated-GPU-only mode, the bridge, `doctor`, status output, and model loaders use CPU. They no longer let `nvidia-modprobe` wake the discrete GPU. Systems with the driver loaded use CUDA as before, and `--cpu` still forces CPU.

## [1.1.1] - 2026-09-01

### 🐛 Bug Fixes

#### Hyprland hotkey waits for bridge warmup and reports real failures
Refs: `76c9ff3`

Pressing Super+R right after the bridge starts no longer fails with a misleading HTTP 409 (curl exit 22). The toggle script waits until `/health` no longer reports `model_loading`.

`/session/start`, `/session/stop`, and `/session/toggle` return a non-2xx status when the session ends in an error state, and the response body keeps the diagnostics. Hotkey failures are logged to `$XDG_STATE_HOME/local-ai-dictation/hotkey.log` and shown with `notify-send` when it is available.

#### Willow recordings survive a transient streaming failure
Refs: `139a860`

When a Willow streaming session fails with a handshake, timeout, connection, or network error, the bridge starts a new session and resends the full captured audio. Version 1.2.0 extended this to two retries.

## [1.1.0] - 2026-07-21

### ✨ Features

#### Sign in to Willow with Google from the terminal
Refs: `67ea06f`

`local-ai-dictation willow-session login` signs in Google-created Willow accounts through the system browser with Supabase PKCE. Wine and the official Willow desktop app are not required. After you sign in, paste the full `https://willowvoice.com/success-open-app?code=…` URL from the browser into the waiting command. Treat that URL as a short-lived credential. Only the minimal refreshable session is saved, with mode `0600`.

A running bridge restarts so it uses the new credentials; `--host` and `--port` select which bridge to restart. Token expiry is stored as an absolute time for both login and refresh, and failure messages include bounded Supabase error detail.

### 🐛 Bug Fixes

#### Willow no longer drops the first seconds of speech
Refs: `1f805f4`

The microphone used to open only after Willow's streaming handshake finished, which often took 1–2 seconds, so the start of each recording was lost. Capture now starts immediately. Audio captured during the handshake is buffered and sent when the session is ready. If the session ends on its own, `send_audio` drops audio instead of raising out of the recorder; the error is reported when recording finalizes. Whisper is unchanged.

## [1.0.0] - 2026-07-15

The first versioned release. It combines local and cloud speech-to-text backends with a persistent bridge, a desktop GUI, and Linux and Windows integrations. The product is now named **AI Dictation** (`d5fec2d`).

### ✨ Features

#### Local dictation with Whisper and Parakeet
Refs: `88d428c`…`b365d90`, `58d3e3a`

Record from a selected microphone and transcribe with local Whisper (the default) or NVIDIA Parakeet. The CLI supports configuration precedence, WebRTC voice activity detection with auto-stop, clipboard delivery, text or JSON output, device discovery (`devices`), diagnostics including model-cache checks (`doctor`), and repeatable benchmarks (`benchmark`).

#### Willow Voice cloud transcription
Refs: `704c3ee`

Willow Voice is a first-class cloud backend. AI Dictation implements Willow's MessagePack WebSocket protocol. It imports and refreshes Willow sessions and stores credentials privately. It streams the recording audio to Willow in real time. The selected backend persists across bridge and desktop restarts. The README documents the privacy boundary between the cloud backend and the local backends.

#### Persistent bridge and desktop GUI
Refs: `e526056`, `c5b1f64`, `1fcb0d0`, `a906d48`, `d17f81e`, `f9dc56d`, `61d37fa`

The localhost bridge keeps the model warm and manages recording sessions, transcript history, backend switching, cancellation, and recovery. It pushes state changes to Waybar with RTMIN signals and to the GUI through an `/events` Server-Sent Events stream; `/health` remains for one-shot checks. A `flock` lock prevents duplicate bridges.

The Electrobun desktop GUI has recording controls, paginated transcript history with a clear-history action, model selection, tray controls, and a global shortcut. It starts a bridge when none is running and reuses an existing one.

#### Linux desktop integrations
Refs: `1a9ddbe`, `c2dd993`, `704c3ee`

A native Linux GUI (`gui`), an XDG launcher, Hyprland shortcuts, a Waybar status module with model switching, and a `bridge-toggle` command for compositor keybinds on Wayland, where Electrobun cannot deliver global shortcuts reliably.

#### Windows GUI with a WSL bridge
Refs: `cdf3a19`…`86c8e55`, `61b7905`, `0bbf89d`, `19012f6`

The desktop GUI can be packaged for Windows 11 x64 (`gui-package`) while the Python bridge and models run in WSL. `gui-package-verify` runs unattended checks of the packaged window, tray, hotkey, startup, and bridge-recovery flows against a deterministic E2E bridge mode.

### ⚡ Performance

#### Transcription overlaps recording
Refs: `459f12b`

Willow receives audio while you speak. On CUDA, Whisper transcribes rolling windows during capture and finalizes only the remaining overlap when you stop. Local models run a warmup inference before they accept recordings. Hyprland paste is sent as soon as the transcript reaches the clipboard.

### 🐛 Bug Fixes

- Silence-only recordings no longer produce hallucinated text such as "Thank you." (`a6da3a5`).
- If Willow closes the session normally (`Session complete`) while the bridge waits for the result, the bridge returns an empty transcript instead of a model failure (`b58ae83`).
- The GUI stays in sync with a bridge that Waybar or a hotkey controls (`b58ae83`).
- Under WSLg, a misleading PortAudio "no default output device" error falls back to a concrete input device (`5bcb3cf`).
- Linux capture keeps the last words before stop, and recording cues no longer play twice (`b452ff0`).
- Bridge shutdown and stopping during model load end cleanly (`3407c52`, `9654fed`).

### History before 1.0.0

- 2025-10-06: Parakeet Dictation began as a GPU/CPU dictation CLI (`f6595a3`).
- 2026-03-14: Packaged CLI, localhost bridge, and Electrobun desktop app (`88d428c`…`5e05528`).
- 2026-03-23 to 2026-03-29: Windows packaged GUI with a WSL bridge (`cdf3a19`…`19012f6`).
- 2026-03-31: Native Linux GUI and Wayland hotkey fallback (`1a9ddbe`, `c2dd993`).
- 2026-04-01: Renamed to Local AI Dictation, with Whisper as the default backend and live backend switching (`58d3e3a`).
- 2026-07-14: Willow Voice cloud backend (`704c3ee`).

[Unreleased]: https://github.com/edxeth/ai-dictation/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/edxeth/ai-dictation/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/edxeth/ai-dictation/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/edxeth/ai-dictation/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/edxeth/ai-dictation/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/edxeth/ai-dictation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/edxeth/ai-dictation/releases/tag/v1.0.0
