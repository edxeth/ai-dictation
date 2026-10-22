"""Model readiness and transcription helpers for AI Dictation."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import shutil
import tempfile
import threading
import wave
from dataclasses import dataclass
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
_MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")
# Kestrel registers the Parakeet TDT 0.6B v3 architecture under NVIDIA's id; a
# local checkpoint of that architecture (Bark) loads through it via model_path.
_PARAKEET_V3_ID = "nvidia/parakeet-tdt-0.6b-v3"
# The Bark weights reuse NVIDIA's tokenizer unchanged; Bark 3.0 does not ship
# tokenizer.json, so it comes from the revision Kestrel itself pins.
_PARAKEET_V3_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
_DECODED_MODELS_DIRNAME = "local-ai-dictation/models"


@dataclass(frozen=True)
class KestrelModelSpec:
    """A checkpoint that the Kestrel Parakeet TDT runtime can load."""

    runtime_model_id: str
    repo_id: str
    # Third-party repositories are pinned so new commits cannot change the
    # weights under a release; Moondream's own repositories track main, as
    # Kestrel does for them.
    revision: str | None = None
    files: tuple[str, ...] = _MODEL_FILES
    # A packed weight file decoded once into a local checkpoint directory.
    packed_file: str | None = None


KESTREL_MODELS: dict[str, KestrelModelSpec] = {
    "parakeet": KestrelModelSpec(MODEL_ID, MODEL_ID),
    "parakeet-redux": KestrelModelSpec(
        "moondream/parakeet-redux",
        "moondream/parakeet-redux",
        # Kestrel treats a checkpoint as ternary only when this manifest exists.
        files=_MODEL_FILES + ("ternary.json",),
    ),
    "bark": KestrelModelSpec(
        _PARAKEET_V3_ID,
        "ConwayResearch/Underdog-Bark-3.0",
        revision="85db2ad18ab41e534a222ee253c29191d77ed919",
        files=("model.bark",),
        packed_file="model.bark",
    ),
    "bark-pro": KestrelModelSpec(
        _PARAKEET_V3_ID,
        "ConwayResearch/Underdog-Bark-3.0-Pro",
        revision="4b5c51e04aa8d16dda9ecdff71bc29f7360149e4",
    ),
}


def kestrel_model_spec(backend: str) -> KestrelModelSpec:
    try:
        return KESTREL_MODELS[backend]
    except KeyError:
        raise ValueError(f"Backend does not run on Kestrel: {backend!r}") from None


class ParakeetTdtEngine:
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
    spec: KestrelModelSpec,
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> Path | None:
    dirname = "models--" + spec.repo_id.replace("/", "--")
    for root in _candidate_cache_roots(env, home=home):
        candidate = root / dirname
        if candidate.exists():
            return candidate
    return None


def _has_files(directory: Path, names: Sequence[str]) -> bool:
    return all((directory / name).is_file() for name in names)


def _has_snapshot(cache_dir: Path, spec: KestrelModelSpec) -> bool:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    if spec.revision is not None:
        return _has_files(snapshots_dir / spec.revision, spec.files)
    try:
        return any(_has_files(child, spec.files) for child in snapshots_dir.iterdir())
    except OSError:
        return False


def _decoded_model_dir(
    spec: KestrelModelSpec,
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> Path:
    source = (
        {} if env is None and home is not None else (os.environ if env is None else env)
    )
    xdg_cache_home = source.get("XDG_CACHE_HOME")
    cache_root = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else (Path.home() if home is None else home).expanduser() / ".cache"
    )
    return (
        cache_root
        / _DECODED_MODELS_DIRNAME
        / spec.repo_id.replace("/", "--")
        / str(spec.revision)
    )


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


def _resolve_snapshot(spec: KestrelModelSpec) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        cached = snapshot_download(
            spec.repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.files),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass
    else:
        if _has_files(Path(cached), spec.files):
            return cached
    return snapshot_download(
        spec.repo_id, revision=spec.revision, allow_patterns=list(spec.files)
    )


def _write_decoded_checkpoint(spec: KestrelModelSpec, staging: Path) -> None:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import save_file

    from local_ai_dictation.bark_format import decode

    packed = Path(_resolve_snapshot(spec)) / str(spec.packed_file)
    tokenizer = hf_hub_download(
        _PARAKEET_V3_ID, "tokenizer.json", revision=_PARAKEET_V3_REVISION
    )
    state, config = decode(packed.read_bytes())
    save_file(state, str(staging / "model.safetensors"))
    (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shutil.copyfile(tokenizer, staging / "tokenizer.json")


def _resolve_decoded_checkpoint(spec: KestrelModelSpec) -> str:
    # A complete checkpoint is never modified or removed, so it can be read
    # without the lock. Builders hold a per-checkpoint lock across processes
    # (bridge, benchmark, doctor) and decode at most once.
    target = _decoded_model_dir(spec)
    if _has_files(target, _MODEL_FILES):
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target.parent / f".{target.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _has_files(target, _MODEL_FILES):
            return str(target)
        staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=target.parent))
        try:
            _write_decoded_checkpoint(spec, staging)
            if target.exists():
                # Left incomplete outside this protocol; nothing can use it.
                shutil.rmtree(target)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return str(target)


def _resolve_model_path(spec: KestrelModelSpec) -> str:
    if spec.packed_file is not None:
        return _resolve_decoded_checkpoint(spec)
    return _resolve_snapshot(spec)


def check_model_cache(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
    backend: str = "parakeet",
) -> dict[str, Any]:
    spec = kestrel_model_spec(backend)
    if spec.packed_file is not None:
        decoded_dir = _decoded_model_dir(spec, env, home=home)
        cache_present = _has_files(decoded_dir, _MODEL_FILES)
        cache_dir = decoded_dir if decoded_dir.exists() else None
    else:
        cache_dir = _find_model_cache_dir(spec, env, home=home)
        cache_present = bool(cache_dir and _has_snapshot(cache_dir, spec))
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
        "model_id": spec.repo_id,
        "import_ready": import_ready,
        "import_error": import_error,
        "detail": detail,
    }


def load_engine(config: DictationConfig) -> ParakeetTdtEngine:
    spec = kestrel_model_spec(config.backend)
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
                model=spec.runtime_model_id,
                model_path=_resolve_model_path(spec),
                device=device,
            )
        )
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, str(exc)) from exc
    return ParakeetTdtEngine(runtime, device=device)


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
