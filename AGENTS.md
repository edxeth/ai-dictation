# AGENTS.md

Local Python dictation bridge with Whisper, Kestrel (Parakeet Ultra and Redux, Bark 3.0 and Pro), and Willow backends.

## Commands

| Command | What it does |
| --- | --- |
| `.venv/bin/pytest -q` | Python test suite. |
| `make check` | Python and desktop format, lint, dead-code, test, and build gates. Run before handoff. |

- When adding or renaming a backend: `BACKEND_LABELS` in `src/local_ai_dictation/backend_state.py` is the source; update its standalone copies in `integrations/waybar/local-ai-dictation-status.py`, `integrations/hyprland/local-ai-dictation-model-menu.sh`, `integrations/hyprland/local-ai-dictation-switch-model.sh`, and `desktop/electrobun/src/shared/backends.ts`.
- When adding a `hyprctl dispatch` call: Hyprland's Lua config manager rejects legacy dispatcher syntax. Probe with `hyprctl eval 'return 1'` (`ok` means Lua) and send the matching form, as `_hyprland_paste_command` in `src/local_ai_dictation/bridge.py` does.
- When touching CUDA selection or diagnostics: gate every `torch.cuda.*` access with `nvidia_driver_loaded()` from `src/local_ai_dictation/gpu.py`; an unguarded `torch.cuda.is_available()` can invoke `nvidia-modprobe` and wake a disabled dGPU.
