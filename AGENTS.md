# AGENTS.md

Local Python dictation bridge with Whisper, Parakeet, and Willow backends.

## Commands

| Command | What it does |
| --- | --- |
| `.venv/bin/pytest -q` | Full test suite. Run before handoff. |

- When touching CUDA selection or diagnostics: gate every `torch.cuda.*` access with `nvidia_driver_loaded()` from `src/local_ai_dictation/gpu.py`; an unguarded `torch.cuda.is_available()` can invoke `nvidia-modprobe` and wake a disabled dGPU.
