"""Parakeet Ultra on transcribe.cpp, used where Kestrel would run on the CPU.

On a Ryzen 7 5800H laptop, Kestrel's CPU path took 1.0 s median and 5.6 s p95
on short dictation clips. transcribe.cpp took 0.26 s / 0.37 s on the Radeon
iGPU over Vulkan (Q8_0) and 0.34 s / 0.71 s on eight CPU threads (F16), and
never opened an NVIDIA device. F16 matched Kestrel's word error rate; Q8_0
differed by one word in each of two long clips. The measurements are on
branch prototype/transcribe-cpp-bench. Kestrel stays faster on CUDA, so this
runtime only replaces the CPU fallback.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from local_ai_dictation.audio import downmix_pcm16_to_mono, resample_pcm16_mono
from local_ai_dictation.errors import MODEL_IMPORT_FAILED, ModelError
from local_ai_dictation.types import TranscriptionResult

GGUF_REPO_ID = "handy-computer/parakeet-ultra-gguf"
# Third-party weights are pinned, as for the Bark checkpoints. This is the
# revision Hex ships and the one the benchmark measured.
GGUF_REVISION = "a1539be1b849ea7463b88343e5277b373d0c4ea9"
IGPU_MODEL_FILE = "parakeet-ultra-Q8_0.gguf"
CPU_MODEL_FILE = "parakeet-ultra-F16.gguf"
# Mesa's Vulkan driver for AMD GPUs (Arch package vulkan-radeon).
RADV_DRIVER_FILE = Path("/usr/share/vulkan/icd.d/radeon_icd.json")
# Eight threads were three times faster than sixteen on the 8-core 5800H.
CPU_THREADS = 8
MODEL_SAMPLE_RATE = 16000
_PCM16_FULL_SCALE = 32768.0


def _import_transcribe_cpp() -> Any:
    return importlib.import_module("transcribe_cpp")


def _hf_hub_download(
    repo_id: str, filename: str, *, revision: str, local_files_only: bool = False
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id, filename, revision=revision, local_files_only=local_files_only
    )


def backend_environment(radv_driver_file: Path) -> dict[str, str]:
    """Return the settings that keep transcribe.cpp off the NVIDIA GPU.

    transcribe.cpp registers its backends, and the Vulkan loader reads its
    driver list, once per process when the library loads. Restricting Vulkan
    to RADV means the NVIDIA driver is never loaded, so a suspended dGPU is
    never enumerated. Without RADV only the CPU backend may register.
    """
    if radv_driver_file.is_file():
        return {
            "TRANSCRIBE_BACKENDS": "vulkan",
            "VK_DRIVER_FILES": str(radv_driver_file),
        }
    return {"TRANSCRIBE_BACKENDS": "cpu"}


@contextmanager
def _scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    # The bridge starts other programs; they must not inherit the restriction.
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_model_file(filename: str) -> str:
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return _hf_hub_download(
            GGUF_REPO_ID, filename, revision=GGUF_REVISION, local_files_only=True
        )
    except LocalEntryNotFoundError:
        return _hf_hub_download(GGUF_REPO_ID, filename, revision=GGUF_REVISION)


def _library_installed() -> bool:
    return importlib.util.find_spec("transcribe_cpp") is not None


def check_model_cache() -> dict[str, Any]:
    """Report the planned device and its cached file without loading the library.

    Loading transcribe.cpp enumerates GPUs, so this reads only the RADV driver
    file, the Hugging Face cache, and the installed package metadata.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    igpu = "VK_DRIVER_FILES" in backend_environment(RADV_DRIVER_FILE)
    filename = IGPU_MODEL_FILE if igpu else CPU_MODEL_FILE
    try:
        cache_path: str | None = _hf_hub_download(
            GGUF_REPO_ID, filename, revision=GGUF_REVISION, local_files_only=True
        )
    except LocalEntryNotFoundError:
        cache_path = None
    return {
        "runtime": "transcribe.cpp",
        "device": "igpu" if igpu else "cpu",
        "model_id": GGUF_REPO_ID,
        "revision": GGUF_REVISION,
        "file": filename,
        "cache_present": cache_path is not None,
        "cache_path": cache_path,
        "import_ready": _library_installed(),
    }


def _read_model_pcm(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(
            f"Parakeet GGUF input must be 16-bit PCM WAV: {path} has "
            f"{8 * sample_width}-bit samples"
        )
    mono = downmix_pcm16_to_mono(frames, channels)
    pcm16 = resample_pcm16_mono(mono, sample_rate, MODEL_SAMPLE_RATE)
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / _PCM16_FULL_SCALE


class GgufParakeetEngine:
    def __init__(self, model: Any, session: Any, *, device: str) -> None:
        self._model = model
        self._session = session
        self._device = device
        # A transcribe.cpp session is not thread-safe; warmup and recording
        # run on separate threads.
        self._inference_lock = threading.Lock()

    def transcribe(self, audio: Sequence[str]) -> list[TranscriptionResult]:
        transcripts = []
        for path in audio:
            pcm = _read_model_pcm(path)
            with self._inference_lock:
                result = self._session.run(pcm)
            transcripts.append(
                TranscriptionResult(text=result.text, device=self._device)
            )
        return transcripts

    def close(self) -> None:
        with self._inference_lock:
            self._session.close()
            self._model.close()


def load_engine() -> GgufParakeetEngine:
    try:
        with _scoped_environment(backend_environment(RADV_DRIVER_FILE)):
            transcribe_cpp = _import_transcribe_cpp()
            devices = transcribe_cpp.backends()
            vulkan = next((d for d in devices if d.kind == "vulkan"), None)
            if vulkan is not None:
                model = transcribe_cpp.Model(
                    _resolve_model_file(IGPU_MODEL_FILE), device=vulkan
                )
                return GgufParakeetEngine(model, model.session(), device="igpu")
            cpu = next((d for d in devices if d.kind == "cpu"), None)
            if cpu is None:
                raise RuntimeError(
                    f"transcribe.cpp reported no CPU device: {[d.kind for d in devices]}"
                )
            model = transcribe_cpp.Model(
                _resolve_model_file(CPU_MODEL_FILE), device=cpu
            )
            return GgufParakeetEngine(
                model, model.session(n_threads=CPU_THREADS), device="cpu"
            )
    except Exception as exc:
        raise ModelError(
            MODEL_IMPORT_FAILED,
            f"Could not load Parakeet Ultra on transcribe.cpp: {exc}",
        ) from exc
