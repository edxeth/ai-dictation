"""Model readiness and transcription helpers for AI Dictation."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import wave

from local_ai_dictation.errors import MODEL_IMPORT_FAILED, MODEL_TRANSCRIBE_FAILED, ModelError
from local_ai_dictation.types import DictationConfig, TranscriptionEngine, TranscriptionResult


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
_MODEL_CACHE_DIRNAME = "models--nvidia--parakeet-tdt-0.6b-v3"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_cache_roots(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> list[Path]:
    source = {} if env is None and home is not None else (os.environ if env is None else env)
    home_dir = Path.home() if home is None else home
    roots: list[Path] = []

    hf_hub_cache = source.get("HF_HUB_CACHE") or source.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        roots.append(Path(hf_hub_cache).expanduser())

    hf_home = source.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    xdg_cache_home = source.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        roots.append(Path(xdg_cache_home).expanduser() / "huggingface" / "hub")

    roots.append(home_dir.expanduser() / ".cache" / "huggingface" / "hub")
    return _dedupe_paths(roots)


def _find_model_cache_dir(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> Path | None:
    for root in _candidate_cache_roots(env, home=home):
        candidate = root / _MODEL_CACHE_DIRNAME
        if candidate.exists():
            return candidate
    return None


def _has_snapshot(cache_dir: Path) -> bool:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    try:
        return any(child.is_dir() for child in snapshots_dir.iterdir())
    except OSError:
        return False


def _check_import_readiness() -> tuple[bool, str | None]:
    try:
        importlib.import_module("torch")
        importlib.import_module("nemo.collections.asr")
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        return False, str(exc)
    return True, None


def _load_runtime_dependencies() -> tuple[Any, Any]:
    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
        torch_module = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - exercised via ModelError in tests/calls
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc
    return nemo_asr, torch_module


def check_model_cache(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> dict[str, Any]:
    cache_dir = _find_model_cache_dir(env, home=home)
    cache_present = bool(cache_dir and _has_snapshot(cache_dir))
    import_ready, import_error = _check_import_readiness()

    if cache_present and import_ready:
        detail = "Local cache and model imports look ready"
    elif not cache_present and import_ready:
        detail = "Local model cache snapshot is missing"
    elif cache_present and not import_ready:
        detail = "Local cache exists, but model imports failed"
    else:
        detail = "Local cache is missing and model imports failed"

    return {
        "checked": True,
        "cache_present": cache_present,
        "cache_path": str(cache_dir) if cache_dir is not None else None,
        "model_id": MODEL_ID,
        "import_ready": import_ready,
        "import_error": import_error,
        "detail": detail,
    }


def load_engine(config: DictationConfig) -> TranscriptionEngine:
    nemo_asr, torch_module = _load_runtime_dependencies()

    try:
        engine = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc

    device = "cpu" if config.cpu or not torch_module.cuda.is_available() else "cuda"
    engine.to(device)
    engine.eval()
    setattr(engine, "_parakeet_device", device)
    return engine


def warmup(engine: TranscriptionEngine) -> None:
    engine.eval()
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temporary_path = handle.name
        with wave.open(temporary_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 8000)
        engine.transcribe([temporary_path], verbose=False)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def transcribe_wav(engine: TranscriptionEngine, path: str | Path) -> TranscriptionResult:
    _, torch_module = _load_runtime_dependencies()

    try:
        with torch_module.inference_mode():
            result = engine.transcribe([str(path)], verbose=False)
    except Exception as exc:
        raise ModelError(MODEL_TRANSCRIBE_FAILED, str(exc)) from exc

    if isinstance(result, list) and result:
        first = result[0]
        transcript_text = getattr(first, "text", first if isinstance(first, str) else str(first))
    else:
        transcript_text = str(result)

    return TranscriptionResult(
        text=transcript_text,
        device=getattr(engine, "_parakeet_device", None),
    )
