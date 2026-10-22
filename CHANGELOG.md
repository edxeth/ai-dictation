# Changelog

All notable changes to AI Dictation are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

#### Parakeet Redux, Bark 3.0, and Bark 3.0 Pro backends

Added three local backends that run on the same Kestrel runtime as Parakeet Ultra: `parakeet-redux` (`moondream/parakeet-redux`, ternary weights, 25 languages), `bark` (`ConwayResearch/Underdog-Bark-3.0`, English), and `bark-pro` (`ConwayResearch/Underdog-Bark-3.0-Pro`, English). Select them with `local-ai-dictation backend set <key>`, `--backend <key>`, the config file, or `LOCAL_AI_DICTATION_BACKEND`. The GUI, Waybar widget, and model-switch notifications show their names.

Both Bark repositories are pinned to the tested commits. Bark 3.0 downloads a 163 MB packed `model.bark` file. On first load the app decodes it once, with a vendored decoder, into a checkpoint under `$XDG_CACHE_HOME/local-ai-dictation/models/`. This adds the `constriction` and `safetensors` dependencies.

The backend cycle (`backend toggle` and the GUI switch control) now visits the new backends between Parakeet Ultra and Willow.

#### Waybar model menu

Added `integrations/hyprland/local-ai-dictation-model-menu.sh`. Bound to the Waybar module's right click, it opens a Walker menu of every backend with the current one marked. Picking a backend switches to it directly, with one bridge restart, so only the chosen model loads. Before this, reaching a backend meant cycling through, and loading, every backend in between.

## [1.2.1] - 2026-10-21

### Fixed

#### Willow recordings stop when Willow ends the session
Refs: `27ff3e3`

Fixed Willow recordings that stayed active after Willow closed the streaming session (`1000 (OK) Session complete`). The bridge kept recording and discarded all later audio. A recording left running looked active in Waybar for hours. When it was stopped, the bridge reported `MODEL_TRANSCRIBE_FAILED` and pasted nothing.

The bridge now stops capture as soon as a connected Willow session ends without a stop request, whether Willow closed it or the network dropped. It logs `Willow ended the session; recording stopped`. Silence is discarded, and the bridge returns to idle. Audio that may contain speech goes through the normal stop path. After a network failure, the bridge retries with the captured audio. After a `Session complete` close, it reports an error and does not retry, so background noise from a forgotten recording is never pasted.

A failed connection at the start of a recording still does not stop capture. The bridge keeps recording and replays the audio into a new session, as before.

## [1.2.0] - 2026-10-04

### Breaking Changes

- Limited Python support to 3.10–3.12; before this release, any version from 3.10 up was accepted. Install with Python 3.12, as the README documents. Reinstall dependencies after you upgrade, because the Parakeet backend now uses Kestrel instead of NeMo (`ee54386`).

### Added

- Added a versioned pre-commit hook. Run `make install-hooks` once per clone. The hook runs `make check` against the staged snapshot and blocks the commit when checks fail or development dependencies are missing (`ee54386`).
- Added `make check`, which runs Ruff, Vulture, deptry, pytest, and the desktop checks, and `make format`, which formats Python and desktop sources (`ee54386`).

### Changed

#### Parakeet Ultra replaces the original local Parakeet model
Refs: `ee54386`

Changed the local Parakeet backend to `moondream/parakeet-ultra`, which runs through Kestrel's direct runtime. It uses CUDA when available and falls back to CPU. It does not construct Photon's telemetry-reporting wrappers. It needs no Moondream account and works offline once the model is cached.

The GUI, Waybar widget, and model-switch notifications now say **Parakeet Ultra**. The `parakeet` backend key and command are unchanged. First use downloads the new weights; the original NVIDIA Parakeet cache is not reused.

### Fixed

- Fixed the desktop GUI falling back to Whisper without notice when Parakeet Ultra or Willow was the saved backend. The GUI now uses the saved selection (`ee54386`).

#### Willow sign-in guidance and bounded connection recovery
Refs: `d5cf394`, `272f6f1`

Fixed expired or rejected Willow sessions showing a raw curl error 22 in the hotkey notification. The bridge now reports `WILLOW_AUTH_REQUIRED`, with `auth_required` and `last_error_code` in the session and health payloads. The hotkey shows a "Willow session expired" notification, and its action copies `local-ai-dictation willow-session login` to the clipboard. The login prompt now shows the pasted callback URL, so you can check it before you submit it.

Fixed a dropped Willow connection at stop losing the recording and showing a truncated JSON popup. Transport failures, including a missing close frame and a DNS name-lookup failure, now get up to two retries, after 1 second and then 2 seconds. Each retry replays the captured audio. Willow finalization stops 20 seconds after you end the recording. The worker is then released, so the next recording can start; this deadline does not apply to Whisper.

An HTTP 401 or 403 socket rejection is treated as a sign-in failure, not a transport retry. Every hotkey request has a time limit, and failures show a plain sentence instead of raw JSON.

### Removed

- Removed unused model adapters, inactive desktop UI and audio paths, obsolete NeMo documentation, and unused dependencies (`ee54386`).

## [1.1.2] - 2026-09-03

### Fixed

- Fixed AI Dictation waking a powered-down NVIDIA GPU through `nvidia-modprobe` when the driver was intentionally unloaded, as on laptops in integrated-GPU-only mode. The bridge, `doctor`, status output, and model loaders now read `/proc/modules` first and use CPU when the driver is not loaded. Systems with the driver loaded use CUDA as before, and `--cpu` still forces CPU (`514b33f`).

## [1.1.1] - 2026-09-01

### Fixed

- Fixed Willow recordings being lost after a transient handshake, timeout, connection, or network failure. The bridge now starts a new session and resends the full captured audio. Version 1.2.0 extended this to two retries (`139a860`).

#### Hyprland hotkey waits for bridge warmup and reports real failures
Refs: `76c9ff3`

Fixed Super+R failing with a misleading HTTP 409 (curl exit 22) when it was pressed right after the bridge started. The toggle script now waits until `/health` no longer reports `model_loading`.

`/session/start`, `/session/stop`, and `/session/toggle` now return a non-2xx status when the session ends in an error state, and the response body keeps the diagnostics. Hotkey failures are logged to `$XDG_STATE_HOME/local-ai-dictation/hotkey.log` and shown with `notify-send` when it is available.

## [1.1.0] - 2026-07-21

### Added

#### Sign in to Willow with Google from the terminal
Refs: `67ea06f`

Added `local-ai-dictation willow-session login`. It signs in Google-created Willow accounts through the system browser with Supabase PKCE, without Wine or the official Willow desktop app. After you sign in, paste the full `https://willowvoice.com/success-open-app?code=…` URL from the browser into the waiting command. Treat that URL as a short-lived credential. Only the minimal refreshable session is saved, with mode `0600`.

A running bridge restarts so it uses the new credentials; `--host` and `--port` select which bridge to restart. Token expiry is stored as an absolute time for both login and refresh, and failure messages include bounded Supabase error detail.

### Fixed

#### Willow no longer drops the first seconds of speech
Refs: `1f805f4`

Fixed the opening 1–2 seconds of every Willow recording being lost. The microphone opened only after Willow's streaming handshake finished.

Capture now starts immediately. Audio captured during the handshake is buffered and sent when the session is ready. If the session ends on its own, `send_audio` drops audio instead of raising out of the recorder; the error is reported when recording finalizes. Whisper is unchanged.

## [1.0.0] - 2026-07-15

### Added

#### Local dictation with Whisper and Parakeet
Refs: `88d428c`…`b365d90`, `58d3e3a`

Added local transcription with Whisper (the default) or NVIDIA Parakeet from a selected microphone. The CLI supports configuration precedence, WebRTC voice activity detection with auto-stop, clipboard delivery, and text or JSON output. It also adds device discovery (`devices`), diagnostics including model-cache checks (`doctor`), and repeatable benchmarks (`benchmark`).

#### Willow Voice cloud transcription
Refs: `704c3ee`

Added Willow Voice as a cloud backend. AI Dictation implements Willow's MessagePack WebSocket protocol. It imports and refreshes Willow sessions and stores credentials privately. The selected backend persists across bridge and desktop restarts. The README documents the privacy boundary between the cloud backend and the local backends.

#### Persistent bridge and desktop GUI
Refs: `e526056`, `c5b1f64`, `1fcb0d0`, `a906d48`, `d17f81e`, `f9dc56d`, `61d37fa`

Added a localhost bridge that keeps the model warm and manages recording sessions, transcript history, backend switching, cancellation, and recovery. It pushes state changes to Waybar with RTMIN signals and to the GUI through an `/events` Server-Sent Events stream; `/health` remains for one-shot checks. A `flock` lock prevents duplicate bridges.

Added the Electrobun desktop GUI with recording controls, paginated transcript history with a clear-history action, model selection, tray controls, and a global shortcut. It starts a bridge when none is running and reuses an existing one.

#### Linux desktop integrations
Refs: `1a9ddbe`, `c2dd993`, `704c3ee`

Added a native Linux GUI (`gui`), an XDG launcher, Hyprland shortcuts, and a Waybar status module with model switching. Added a `bridge-toggle` command for compositor keybinds on Wayland, where Electrobun cannot deliver global shortcuts reliably.

#### Windows GUI with a WSL bridge
Refs: `cdf3a19`…`86c8e55`, `61b7905`, `0bbf89d`, `19012f6`

Added packaging of the desktop GUI for Windows 11 x64 (`gui-package`) while the Python bridge and models run in WSL. `gui-package-verify` runs unattended checks of the packaged window, tray, hotkey, startup, and bridge-recovery flows against a deterministic E2E bridge mode.

### Changed

- Changed the product name to **AI Dictation**. Command, package, configuration, and state names stay `local-ai-dictation`, so existing installs keep working (`d5fec2d`).

#### Transcription overlaps recording
Refs: `459f12b`

Changed transcription to run during recording, which shortens the wait after you stop. Willow receives audio while you speak. On CUDA, Whisper transcribes rolling windows during capture and finalizes only the remaining overlap when you stop.

Local models run a warmup inference before they accept recordings. Hyprland paste is sent as soon as the transcript reaches the clipboard.

### Fixed

- Fixed silence-only recordings producing hallucinated text such as "Thank you." They now return an empty transcript (`a6da3a5`).
- Fixed a normal Willow close (`Session complete`) while the bridge waited for the result being reported as a model failure. The bridge now returns an empty transcript (`b58ae83`).
- Fixed the GUI staying behind a "preparing" overlay while a bridge controlled by Waybar or a hotkey was already recording. The GUI now refreshes bridge state through native RPC when Server-Sent Events are unavailable (`b58ae83`).
- Fixed sessions under WSLg failing with a misleading PortAudio "no default output device" error. The bridge now retries with a concrete input device (`5bcb3cf`).
- Fixed Linux capture losing the last words before stop, and recording cues playing twice (`b452ff0`).

[1.2.1]: https://github.com/edxeth/ai-dictation/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/edxeth/ai-dictation/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/edxeth/ai-dictation/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/edxeth/ai-dictation/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/edxeth/ai-dictation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/edxeth/ai-dictation/releases/tag/v1.0.0
