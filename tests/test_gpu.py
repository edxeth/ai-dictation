from __future__ import annotations

import importlib
import io
from types import SimpleNamespace

import pytest

from local_ai_dictation.gpu import nvidia_driver_loaded
from local_ai_dictation.types import DictationConfig


dictation_module = importlib.import_module("local_ai_dictation.dictation")
doctor_module = importlib.import_module("local_ai_dictation.doctor")
model_module = importlib.import_module("local_ai_dictation.model")
whisper_module = importlib.import_module("local_ai_dictation.whisper")


class _ForbiddenCudaProbe:
    def is_available(self) -> bool:
        raise AssertionError("torch CUDA must not be queried without a loaded NVIDIA driver")


class _FakeTorch:
    cuda = _ForbiddenCudaProbe()


class _RecordingCudaProbe:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.calls = 0

    def is_available(self) -> bool:
        self.calls += 1
        return self.available


class _RecordingTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = _RecordingCudaProbe(available)


class _FakeParakeetModel:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        return None


def test_nvidia_driver_loaded_accepts_core_and_companion_modules():
    assert nvidia_driver_loaded(["nvidia 123 0 - Live 0x0\n"]) is True
    assert nvidia_driver_loaded(["nvidia_drm 123 0 - Live 0x0\n"]) is True


def test_nvidia_driver_loaded_rejects_similarly_named_framebuffer_module():
    assert nvidia_driver_loaded(["nvidiafb 123 0 - Live 0x0\n"]) is False


def test_nvidia_driver_loaded_reads_an_optional_modules_path(tmp_path):
    modules_path = tmp_path / "modules"
    modules_path.write_text("snd_hda_intel 123 0 - Live 0x0\nnvidia_uvm 456 0 - Live 0x0\n", encoding="utf-8")

    assert nvidia_driver_loaded(modules_path) is True


def test_nvidia_driver_loaded_treats_a_missing_modules_path_as_not_loaded(tmp_path):
    assert nvidia_driver_loaded(tmp_path / "missing-modules") is False


@pytest.mark.parametrize("backend", ["whisper", "parakeet"])
def test_model_load_uses_cpu_without_probing_cuda_when_nvidia_driver_is_absent(monkeypatch, backend):
    monkeypatch.setattr(dictation_module, "nvidia_driver_loaded", lambda: False)
    monkeypatch.setattr(dictation_module, "spinner_animation", lambda *args, **kwargs: None)

    if backend == "whisper":
        selected_device: dict[str, str] = {}

        def build_whisper_model(model_id: str, *, device: str, compute_type: str):
            selected_device["device"] = device
            return object()

        runtime_module = SimpleNamespace(WhisperModel=build_whisper_model)
    else:
        parakeet_model = _FakeParakeetModel()
        runtime_module = SimpleNamespace(
            models=SimpleNamespace(
                ASRModel=SimpleNamespace(from_pretrained=lambda model_id: parakeet_model),
            )
        )

    model, use_cuda, _start, _end = dictation_module._load_model(
        DictationConfig(backend=backend, debug=True),
        runtime_module,
        _FakeTorch(),
        status_stream=io.StringIO(),
    )

    assert use_cuda is False
    if backend == "whisper":
        assert selected_device["device"] == "cpu"
        assert model._parakeet_device == "cpu"
    else:
        assert parakeet_model.device == "cpu"


def test_loaded_nvidia_driver_allows_cuda_probe_and_gpu_model(monkeypatch):
    monkeypatch.setattr(dictation_module, "nvidia_driver_loaded", lambda: True)
    monkeypatch.setattr(dictation_module, "spinner_animation", lambda *args, **kwargs: None)
    parakeet_model = _FakeParakeetModel()
    runtime_module = SimpleNamespace(
        models=SimpleNamespace(
            ASRModel=SimpleNamespace(from_pretrained=lambda model_id: parakeet_model),
        )
    )
    torch_module = _RecordingTorch(available=True)

    _model, use_cuda, _start, _end = dictation_module._load_model(
        DictationConfig(backend="parakeet", debug=True),
        runtime_module,
        torch_module,
        status_stream=io.StringIO(),
    )

    assert torch_module.cuda.calls == 1
    assert use_cuda is True
    assert parakeet_model.device == "cuda"


def test_cpu_override_keeps_model_on_cpu_when_nvidia_driver_is_loaded(monkeypatch):
    monkeypatch.setattr(dictation_module, "nvidia_driver_loaded", lambda: True)
    monkeypatch.setattr(dictation_module, "spinner_animation", lambda *args, **kwargs: None)
    parakeet_model = _FakeParakeetModel()
    runtime_module = SimpleNamespace(
        models=SimpleNamespace(
            ASRModel=SimpleNamespace(from_pretrained=lambda model_id: parakeet_model),
        )
    )
    torch_module = _RecordingTorch(available=True)

    _model, use_cuda, _start, _end = dictation_module._load_model(
        DictationConfig(backend="parakeet", cpu=True, debug=True),
        runtime_module,
        torch_module,
        status_stream=io.StringIO(),
    )

    assert use_cuda is False
    assert parakeet_model.device == "cpu"


def test_standalone_parakeet_loader_does_not_probe_cuda_without_nvidia_driver(monkeypatch):
    parakeet_model = _FakeParakeetModel()
    runtime_module = SimpleNamespace(
        models=SimpleNamespace(
            ASRModel=SimpleNamespace(from_pretrained=lambda model_id: parakeet_model),
        )
    )
    monkeypatch.setattr(model_module, "nvidia_driver_loaded", lambda: False)
    monkeypatch.setattr(model_module, "_load_runtime_dependencies", lambda: (runtime_module, _FakeTorch()))

    model_module.load_engine(DictationConfig(backend="parakeet"))

    assert parakeet_model.device == "cpu"


def test_standalone_whisper_loader_does_not_probe_cuda_without_nvidia_driver(monkeypatch):
    selected_device: dict[str, str] = {}

    def build_whisper_model(model_id: str, *, device: str, compute_type: str):
        selected_device["device"] = device
        return object()

    runtime_module = SimpleNamespace(WhisperModel=build_whisper_model)
    monkeypatch.setattr(whisper_module, "nvidia_driver_loaded", lambda: False)
    monkeypatch.setattr(whisper_module, "_load_runtime_dependencies", lambda: (runtime_module, _FakeTorch()))

    engine = whisper_module.load_engine(DictationConfig(backend="whisper"))

    assert selected_device["device"] == "cpu"
    assert engine._parakeet_device == "cpu"


def test_doctor_does_not_probe_cuda_without_nvidia_driver(monkeypatch):
    monkeypatch.setattr(doctor_module, "nvidia_driver_loaded", lambda: False)
    monkeypatch.setattr(doctor_module.importlib, "import_module", lambda name: _FakeTorch())

    status = doctor_module._collect_cuda_status()

    assert status == {
        "available": False,
        "selected_device": "cpu",
        "device_name": None,
        "detail": "CUDA runtime is unavailable; CPU fallback remains usable",
    }
