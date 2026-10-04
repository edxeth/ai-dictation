"""Model readiness and transcription helpers for AI Dictation."""

from __future__ import annotations

import importlib
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_ai_dictation.errors import (
    MODEL_IMPORT_FAILED,
    MODEL_TRANSCRIBE_FAILED,
    ModelError,
)
from local_ai_dictation.gpu import nvidia_driver_loaded
from local_ai_dictation.types import (
    DictationConfig,
    TranscriptionEngine,
    TranscriptionResult,
)

MODEL_ID = "moondream/parakeet-ultra"
_MODEL_CACHE_DIRNAME = "models--moondream--parakeet-ultra"
_MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")


class ParakeetUltraEngine:
    def __init__(self, runtime: Any, *, device: str) -> None:
        self._runtime = runtime
        self._device = device
        self._inference_lock = threading.Lock()

    def transcribe(self, audio: Sequence[str]) -> list[TranscriptionResult]:
        # Warmup and recording run on separate threads; the direct runtime owns
        # mutable CUDA graphs and must not process both calls at once.
        with self._inference_lock:
            results = self._runtime.forward(
                "transcribe", [{"audio": str(path)} for path in audio]
            )
        transcripts = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            transcripts.append(
                TranscriptionResult(text=result["text"], device=self._device)
            )
        return transcripts

    def close(self) -> None:
        with self._inference_lock:
            self._runtime.shutdown()


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
    source = (
        {} if env is None and home is not None else (os.environ if env is None else env)
    )
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
        return any(
            all((child / name).is_file() for name in _MODEL_FILES)
            for child in snapshots_dir.iterdir()
        )
    except OSError:
        return False


def _check_import_readiness() -> tuple[bool, str | None]:
    try:
        _load_runtime_dependencies()
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        return False, str(exc)
    return True, None


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        runtime_class = importlib.import_module(
            "kestrel.models.parakeet_tdt"
        ).ParakeetTdtRuntime
        config_class = importlib.import_module("kestrel.config").RuntimeConfig
        torch_module = importlib.import_module("torch")
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised via ModelError in tests/calls
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc
    return runtime_class, config_class, torch_module


def _resolve_model_path() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        cached = snapshot_download(
            MODEL_ID, allow_patterns=list(_MODEL_FILES), local_files_only=True
        )
    except LocalEntryNotFoundError:
        pass
    else:
        if all((Path(cached) / name).is_file() for name in _MODEL_FILES):
            return cached
    return snapshot_download(MODEL_ID, allow_patterns=list(_MODEL_FILES))


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


def load_engine(config: DictationConfig) -> ParakeetUltraEngine:
    runtime_class, config_class, torch_module = _load_runtime_dependencies()
    device = (
        "cpu"
        if config.cpu
        or not nvidia_driver_loaded()
        or not torch_module.cuda.is_available()
        else "cuda"
    )
    try:
        # Use Kestrel's exported runtime, not PhotonClient/InferenceEngine:
        # those wrappers send machine and usage telemetry without an opt-out.
        runtime = runtime_class(
            config_class(
                model=MODEL_ID,
                model_path=_resolve_model_path(),
                device=device,
            )
        )
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc
    return ParakeetUltraEngine(runtime, device=device)


def warmup(engine: TranscriptionEngine) -> None:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temporary_path = handle.name
        with wave.open(temporary_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 8000)
        engine.transcribe([temporary_path])
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def transcribe_wav(
    engine: TranscriptionEngine, path: str | Path
) -> TranscriptionResult:
    try:
        result = engine.transcribe([str(path)])
    except Exception as exc:
        raise ModelError(MODEL_TRANSCRIBE_FAILED, str(exc)) from exc

    if isinstance(result, list) and result:
        first = result[0]
        transcript_text = getattr(
            first, "text", first if isinstance(first, str) else str(first)
        )
    else:
        transcript_text = str(result)

    return TranscriptionResult(
        text=transcript_text,
        device=getattr(engine, "_device", None),
    )
