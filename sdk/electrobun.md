# Electrobun notes for this repo

## What it is

Electrobun is the desktop app framework used by `desktop/electrobun/` in this repo.
It provides:
- a Bun-side desktop process
- a webview-based renderer
- RPC between the desktop process and renderer
- desktop APIs like window, tray, and global shortcuts

In this repo it powers the `AI Dictation` and talks to the Python bridge over HTTP.

## How this repo uses it

Key files:
- `desktop/electrobun/src/bun/index.ts` — creates the window, tray, global hotkey, startup diagnostics, and E2E automation surface
- `desktop/electrobun/src/mainview/index.ts` — renderer UI and renderer snapshot/test hooks
- `desktop/electrobun/package.json` — Electrobun package plus `bun run check`
- `desktop/electrobun/electrobun.config.ts` — cross-platform Electrobun build config
- `src/local_ai_dictation/desktop.py` — native GUI launch plus WSL-side staging, Windows packaging, packaged-app smoke, and unattended verify entrypoints
- `src/local_ai_dictation/cli.py` — exposes `gui` and `gui-package*` commands

Current bridge assumptions:
- GUI reads `LOCAL_AI_DICTATION_BRIDGE_URL`
- GUI reads `LOCAL_AI_DICTATION_BRIDGE_COMMAND`
- GUI expects the backend to expose `/health`, `/session/start`, `/session/stop`, `/session/toggle`, and `/session/clear-history`

## Supported topology and boundaries

Supported in this repo:
- native Linux Electrobun GUI over localhost
- packaged Windows 11 x64 Electrobun GUI
- `local-ai-dictation bridge` running separately inside WSL over `127.0.0.1` for the Windows workflow
- WebView2 renderer on Windows

Not supported in this release:
- native Windows backend or microphone/model execution
- ARM64-native Windows packaging
- automatic bridge startup in the packaged Windows workflow

Lifecycle rules:
- bridge startup remains user-controlled
- no microphone access before explicit user action
- the packaged app may show bridge guidance, but it does not own WSL lifecycle

## Windows support summary

Electrobun currently documents:
- Windows 11+ support
- Windows x64 as stable
- Windows uses WebView2 by default
- CEF can be bundled as a fallback for advanced rendering needs

Important caveats for future tasks:
- WebView2 runtime is required on Windows
- some Windows rough edges are still reported upstream, including HiDPI and some Unicode/menu issues
- Windows behavior should be validated for tray, hotkeys, focus, and windowing
- ARM64 should not be assumed as a first-class target for this repo without extra validation

## WSL to Windows packaging rule

The repo usually lives at a `\\wsl.localhost\...` path when opened from WSL. Windows Bun/Electrobun tooling must not build from that UNC working directory.

Use the repo-supported handoff instead:
- `local-ai-dictation gui-stage` copies `desktop/electrobun/` into `%LOCALAPPDATA%\LocalAIDictation\staging\...`
- `local-ai-dictation gui-package` runs Windows `bun install` and `bunx electrobun build --env=stable` from that drive-backed staging path
- packaging artifacts are emitted from the staged desktop folder under `build/stable-win-x64/` and `artifacts/`

## Canonical Windows commands

From WSL with the project virtualenv active:

```bash
.venv/bin/local-ai-dictation gui-stage --json
.venv/bin/local-ai-dictation gui-package --json
.venv/bin/local-ai-dictation gui-package-verify --json --timeout-seconds 240
```

Meaning:
- `gui-stage` proves the drive-backed Windows staging handoff
- `gui-package` produces the unsigned Windows x64 installer
- `gui-package-verify` is the single unattended local Windows + WSL packaging/E2E entrypoint

`gui-package-verify` currently runs these packaged checks:
- smoke startup diagnostics
- localhost automation surface
- bridge offline → online recovery
- main-window start/stop
- tray open/toggle/quit
- global hotkey registration and callback path

## Development vs release workflow

Development inside Linux/WSL can use:
- `local-ai-dictation gui`
- `cd desktop/electrobun && bun install && bun run start`

Those commands are for local dev ergonomics. The supported packaged Windows workflow is still driven by `gui-package` and verified by `gui-package-verify`.

## Useful upstream docs

- Docs home: https://blackboard.sh/electrobun/docs/
- Compatibility: https://blackboard.sh/electrobun/docs/guides/compatability
- Cross-platform development: https://blackboard.sh/electrobun/docs/guides/cross-platform-development
- Build configuration: https://blackboard.sh/electrobun/docs/apis/cli/build-configuration
- Bundling CEF: https://blackboard.sh/electrobun/docs/apis/bundling-cef
- Tray API: https://blackboard.sh/electrobun/docs/apis/tray
- BrowserWindow API: https://blackboard.sh/electrobun/docs/apis/browser-window
- BrowserView API: https://blackboard.sh/electrobun/docs/apis/browser-view
- Bun-side API entrypoint: https://blackboard.sh/electrobun/docs/apis/bun

## Source notes

Research for these notes was based on the Electrobun docs and repository, plus Microsoft WSL interop/networking docs for the Windows GUI + WSL bridge topology.
