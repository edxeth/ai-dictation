from __future__ import annotations

import importlib
import io
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_ai_dictation import gguf as gguf_module
from local_ai_dictation import model as model_module
from local_ai_dictation.errors import (
    MODEL_IMPORT_FAILED,
    MODEL_TRANSCRIBE_FAILED,
    ModelError,
)
from local_ai_dictation.types import DictationConfig

GGUF_ENV_KEYS = ("TRANSCRIBE_BACKENDS", "VK_DRIVER_FILES")


class FakeSession:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []
        self.closed = False

    def run(self, pcm):
        self.inputs.append(np.array(pcm, copy=True))
        return SimpleNamespace(text="Run the focused tests.")

    def close(self) -> None:
        self.closed = True


class FakeModel:
    def __init__(self, path, *, device) -> None:
        self.path = Path(path)
        self.device = device
        self.session_options: dict = {}
        self.session_handle = FakeSession()
        self.closed = False

    def session(self, **options):
        self.session_options = options
        return self.session_handle

    def close(self) -> None:
        self.closed = True


def _device(kind: str, description: str):
    return SimpleNamespace(kind=kind, description=description)


class FakeTranscribeCpp:
    """Records the process environment each native entry point observed."""

    def __init__(self, devices, *, fail_load: Exception | None = None) -> None:
        self.devices = devices
        self.fail_load = fail_load
        self.environments: dict[str, dict[str, str | None]] = {}
        self.models: list[FakeModel] = []

    def snapshot(self, step: str) -> None:
        self.environments[step] = {key: os.environ.get(key) for key in GGUF_ENV_KEYS}

    def backends(self):
        self.snapshot("backends")
        return self.devices

    def Model(self, path, *, device):  # noqa: N802 - mirrors the binding's class
        self.snapshot("model")
        if self.fail_load is not None:
            raise self.fail_load
        model = FakeModel(path, device=device)
        self.models.append(model)
        return model


@pytest.fixture
def downloads(monkeypatch):
    requests: list[dict] = []

    def hf_hub_download(repo_id, filename, *, revision, local_files_only=False):
        requests.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        return f"/hf-cache/{filename}"

    monkeypatch.setattr(gguf_module, "_hf_hub_download", hf_hub_download)
    return requests


def _install(monkeypatch, fake: FakeTranscribeCpp, *, radv_driver: Path | None):
    def import_transcribe_cpp():
        fake.snapshot("import")
        return fake

    monkeypatch.setattr(gguf_module, "_import_transcribe_cpp", import_transcribe_cpp)
    monkeypatch.setattr(
        gguf_module,
        "RADV_DRIVER_FILE",
        radv_driver if radv_driver is not None else Path("/missing/radeon_icd.json"),
    )
    for key in GGUF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _radv_driver(tmp_path: Path) -> Path:
    driver = tmp_path / "radeon_icd.json"
    driver.write_text("{}", encoding="utf-8")
    return driver


def _write_wav(path: Path, frames: bytes, *, rate: int, channels: int, width: int = 2):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(rate)
        wav_file.writeframes(frames)
    return path


def test_backend_environment_limits_vulkan_to_radv_when_its_driver_is_installed(
    tmp_path,
):
    driver = _radv_driver(tmp_path)

    assert gguf_module.backend_environment(driver) == {
        "TRANSCRIBE_BACKENDS": "vulkan",
        "VK_DRIVER_FILES": str(driver),
    }
    assert gguf_module.backend_environment(tmp_path / "absent.json") == {
        "TRANSCRIBE_BACKENDS": "cpu"
    }


def test_igpu_load_applies_the_backend_environment_before_import_and_restores_it(
    monkeypatch, tmp_path, downloads
):
    driver = _radv_driver(tmp_path)
    fake = FakeTranscribeCpp(
        [_device("vulkan", "AMD Radeon Graphics (RADV RENOIR)"), _device("cpu", "CPU")]
    )
    _install(monkeypatch, fake, radv_driver=driver)
    monkeypatch.setenv("VK_DRIVER_FILES", "/user/choice.json")

    engine = gguf_module.load_engine()

    expected = {"TRANSCRIBE_BACKENDS": "vulkan", "VK_DRIVER_FILES": str(driver)}
    assert fake.environments == {
        "import": expected,
        "backends": expected,
        "model": expected,
    }
    assert os.environ.get("VK_DRIVER_FILES") == "/user/choice.json"
    assert "TRANSCRIBE_BACKENDS" not in os.environ
    (model,) = fake.models
    assert model.device.kind == "vulkan"
    assert model.path == Path("/hf-cache/parakeet-ultra-Q8_0.gguf")
    assert model.session_options == {}
    assert engine._device == "igpu"


def test_cpu_load_uses_f16_on_eight_threads_without_a_radv_driver(
    monkeypatch, downloads
):
    fake = FakeTranscribeCpp([_device("cpu", "AMD Ryzen 7 5800H")])
    _install(monkeypatch, fake, radv_driver=None)

    engine = gguf_module.load_engine()

    assert fake.environments["import"] == {
        "TRANSCRIBE_BACKENDS": "cpu",
        "VK_DRIVER_FILES": None,
    }
    (model,) = fake.models
    assert model.device.kind == "cpu"
    assert model.path == Path("/hf-cache/parakeet-ultra-F16.gguf")
    assert model.session_options == {"n_threads": 8}
    assert engine._device == "cpu"


def test_radv_driver_without_a_usable_vulkan_device_falls_back_to_the_cpu(
    monkeypatch, tmp_path, downloads
):
    fake = FakeTranscribeCpp([_device("cpu", "AMD Ryzen 7 5800H")])
    _install(monkeypatch, fake, radv_driver=_radv_driver(tmp_path))

    engine = gguf_module.load_engine()

    assert fake.models[0].path == Path("/hf-cache/parakeet-ultra-F16.gguf")
    assert engine._device == "cpu"


def test_model_files_are_read_from_the_cache_before_any_download(
    monkeypatch, downloads
):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def cache_then_network(repo_id, filename, *, revision, local_files_only=False):
        downloads.append((filename, revision, local_files_only))
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        return f"/downloaded/{filename}"

    monkeypatch.setattr(gguf_module, "_hf_hub_download", cache_then_network)
    fake = FakeTranscribeCpp([_device("cpu", "CPU")])
    _install(monkeypatch, fake, radv_driver=None)

    gguf_module.load_engine()

    pin = "a1539be1b849ea7463b88343e5277b373d0c4ea9"
    assert downloads == [
        ("parakeet-ultra-F16.gguf", pin, True),
        ("parakeet-ultra-F16.gguf", pin, False),
    ]
    assert fake.models[0].path == Path("/downloaded/parakeet-ultra-F16.gguf")


def test_cached_model_file_needs_no_network(monkeypatch, downloads):
    fake = FakeTranscribeCpp([_device("cpu", "CPU")])
    _install(monkeypatch, fake, radv_driver=None)

    gguf_module.load_engine()

    assert downloads == [
        {
            "repo_id": "handy-computer/parakeet-ultra-gguf",
            "filename": "parakeet-ultra-F16.gguf",
            "revision": "a1539be1b849ea7463b88343e5277b373d0c4ea9",
            "local_files_only": True,
        }
    ]


def test_load_errors_keep_the_model_error_contract_and_restore_the_environment(
    monkeypatch, downloads
):
    fake = FakeTranscribeCpp(
        [_device("cpu", "CPU")], fail_load=RuntimeError("bad GGUF header")
    )
    _install(monkeypatch, fake, radv_driver=None)

    with pytest.raises(ModelError, match="bad GGUF header") as error:
        gguf_module.load_engine()

    assert error.value.code == MODEL_IMPORT_FAILED
    assert "TRANSCRIBE_BACKENDS" not in os.environ


def _loaded_cpu_engine(monkeypatch):
    fake = FakeTranscribeCpp([_device("cpu", "CPU")])
    _install(monkeypatch, fake, radv_driver=None)
    monkeypatch.setattr(
        gguf_module,
        "_hf_hub_download",
        lambda repo_id, filename, *, revision, local_files_only=False: filename,
    )
    return gguf_module.load_engine(), fake


def test_transcribe_downmixes_and_resamples_capture_wav_to_16k_float(
    monkeypatch, tmp_path
):
    engine, fake = _loaded_cpu_engine(monkeypatch)
    # One second of 48 kHz stereo at a constant half of full scale.
    half_scale = (16384).to_bytes(2, "little", signed=True)
    capture = _write_wav(
        tmp_path / "capture.wav", half_scale * 2 * 48000, rate=48000, channels=2
    )

    result = model_module.transcribe_wav(engine, capture)

    assert result.text == "Run the focused tests."
    assert result.device == "cpu"
    (pcm,) = fake.models[0].session_handle.inputs
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert abs(len(pcm) - 16000) <= 1
    # Resampling may ramp the first samples; the steady state is 16384/32768.
    assert np.allclose(pcm[16:], 0.5, atol=1e-3)


def test_transcribe_rejects_wav_that_is_not_16_bit_pcm(monkeypatch, tmp_path):
    engine, _fake = _loaded_cpu_engine(monkeypatch)
    eight_bit = _write_wav(
        tmp_path / "eight-bit.wav", b"\x80" * 1600, rate=16000, channels=1, width=1
    )

    with pytest.raises(ModelError, match="eight-bit.wav.*8-bit") as error:
        model_module.transcribe_wav(engine, eight_bit)

    assert error.value.code == MODEL_TRANSCRIBE_FAILED


def test_warmup_runs_half_a_second_of_silence_through_the_session(monkeypatch):
    engine, fake = _loaded_cpu_engine(monkeypatch)

    model_module.warmup(engine)

    (pcm,) = fake.models[0].session_handle.inputs
    assert len(pcm) == 8000
    assert not pcm.any()


def test_close_releases_the_session_and_the_model(monkeypatch):
    engine, fake = _loaded_cpu_engine(monkeypatch)

    engine.close()

    assert fake.models[0].session_handle.closed
    assert fake.models[0].closed


def test_ultra_on_the_cpu_runs_on_transcribe_cpp_through_the_dictation_batch_path(
    monkeypatch,
):
    dictation = importlib.import_module("local_ai_dictation.dictation")
    fake = FakeTranscribeCpp([_device("cpu", "CPU")])
    _install(monkeypatch, fake, radv_driver=None)
    monkeypatch.setattr(
        gguf_module,
        "_hf_hub_download",
        lambda repo_id, filename, *, revision, local_files_only=False: filename,
    )

    def kestrel_must_not_load(config):
        raise AssertionError("Kestrel must not build a CPU runtime for Ultra")

    monkeypatch.setattr(
        model_module,
        "_load_runtime_dependencies",
        lambda: (kestrel_must_not_load, SimpleNamespace, object()),
    )
    monkeypatch.setattr(dictation, "spinner_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(dictation, "has_probable_speech", lambda *args, **kwargs: True)
    config = DictationConfig(backend="parakeet", cpu=True, debug=True)
    engine, use_cuda, _, _ = dictation._load_model(
        config, model_module, object(), status_stream=io.StringIO()
    )

    # 0.1 s of 48 kHz capture audio, as the bridge records it before resampling.
    result, path, _, _ = dictation._transcribe_once(
        config, engine, b"\x00\x10" * 4800, 48000, status_stream=io.StringIO()
    )
    try:
        assert result.text == "Run the focused tests."
        assert use_cuda is False
        assert len(fake.models[0].session_handle.inputs[0]) == 1600
    finally:
        Path(path).unlink()


def _cache_lookup(cached: set[str]):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def hf_hub_download(repo_id, filename, *, revision, local_files_only=False):
        assert local_files_only, "the cache check must never download"
        if filename not in cached:
            raise LocalEntryNotFoundError(f"{filename} is not cached")
        return f"/hf-cache/{revision}/{filename}"

    return hf_hub_download


def _forbid_library_import(monkeypatch):
    # Importing transcribe_cpp enumerates GPUs, which can wake a suspended dGPU.
    def forbidden():
        raise AssertionError("the cache check must not import transcribe_cpp")

    monkeypatch.setattr(gguf_module, "_import_transcribe_cpp", forbidden)


def test_cache_check_reports_the_igpu_file_without_loading_the_library(
    monkeypatch, tmp_path
):
    _forbid_library_import(monkeypatch)
    monkeypatch.setattr(gguf_module, "RADV_DRIVER_FILE", _radv_driver(tmp_path))
    monkeypatch.setattr(
        gguf_module, "_hf_hub_download", _cache_lookup({"parakeet-ultra-Q8_0.gguf"})
    )
    monkeypatch.setattr(gguf_module, "_library_installed", lambda: True)

    pin = "a1539be1b849ea7463b88343e5277b373d0c4ea9"
    assert gguf_module.check_model_cache() == {
        "runtime": "transcribe.cpp",
        "device": "igpu",
        "model_id": "handy-computer/parakeet-ultra-gguf",
        "revision": pin,
        "file": "parakeet-ultra-Q8_0.gguf",
        "cache_present": True,
        "cache_path": f"/hf-cache/{pin}/parakeet-ultra-Q8_0.gguf",
        "import_ready": True,
    }


def test_cache_check_reports_a_missing_cpu_file_and_a_missing_library(monkeypatch):
    _forbid_library_import(monkeypatch)
    monkeypatch.setattr(gguf_module, "RADV_DRIVER_FILE", Path("/missing/radeon.json"))
    monkeypatch.setattr(
        gguf_module, "_hf_hub_download", _cache_lookup({"parakeet-ultra-Q8_0.gguf"})
    )
    monkeypatch.setattr(gguf_module, "_library_installed", lambda: False)

    status = gguf_module.check_model_cache()

    assert status["device"] == "cpu"
    assert status["file"] == "parakeet-ultra-F16.gguf"
    assert status["cache_present"] is False
    assert status["cache_path"] is None
    assert status["import_ready"] is False
