from __future__ import annotations

import importlib
import io
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_ai_dictation import model as model_module
from local_ai_dictation.backend_state import KESTREL_BACKENDS
from local_ai_dictation.errors import (
    MODEL_IMPORT_FAILED,
    MODEL_TRANSCRIBE_FAILED,
    ModelError,
)
from local_ai_dictation.model import MODEL_ID, check_model_cache, warmup
from local_ai_dictation.types import DictationConfig


def test_parakeet_warmup_runs_real_inference_and_removes_fixture():
    paths: list[Path] = []

    class Engine:
        def transcribe(self, audio):
            path = Path(audio[0])
            assert path.is_file()
            paths.append(path)
            return []

    warmup(Engine())

    assert len(paths) == 1
    assert paths[0].exists() is False


def test_check_model_cache_detects_local_snapshot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    snapshot_dir = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--moondream--parakeet-ultra"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (snapshot_dir / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "local_ai_dictation.model._check_import_readiness", lambda: (True, None)
    )

    status = check_model_cache(home=home)

    assert status == {
        "checked": True,
        "cache_present": True,
        "cache_path": str(snapshot_dir.parent.parent),
        "model_id": MODEL_ID,
        "import_ready": True,
        "import_error": None,
        "detail": "Local cache and model imports look ready",
    }


def test_check_model_cache_reports_missing_cache_and_import_failure(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setattr(
        "local_ai_dictation.model._check_import_readiness",
        lambda: (False, "kestrel import failed"),
    )

    status = check_model_cache(home=home)

    assert status == {
        "checked": True,
        "cache_present": False,
        "cache_path": None,
        "model_id": MODEL_ID,
        "import_ready": False,
        "import_error": "kestrel import failed",
        "detail": "Local cache is missing and model imports failed",
    }


def test_ultra_load_uses_direct_runtime_and_cached_checkpoint(monkeypatch):
    configurations = []
    runtime = SimpleNamespace(forward=lambda task, inputs: [{"text": "Hello, world!"}])

    def build_runtime(config):
        configurations.append(config)
        return runtime

    monkeypatch.setattr(
        model_module,
        "_load_runtime_dependencies",
        lambda: (
            build_runtime,
            SimpleNamespace,
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        model_module, "_resolve_model_path", lambda spec: "/cached/ultra"
    )

    engine = model_module.load_engine(DictationConfig(backend="parakeet", cpu=True))

    assert MODEL_ID == "moondream/parakeet-ultra"
    assert vars(configurations[0]) == {
        "model": "moondream/parakeet-ultra",
        "model_path": "/cached/ultra",
        "device": "cpu",
    }
    assert model_module.transcribe_wav(engine, "speech.wav").text == "Hello, world!"
    assert engine._device == "cpu"


def test_every_kestrel_backend_has_a_model_spec():
    assert set(model_module.KESTREL_MODELS) == KESTREL_BACKENDS


@pytest.mark.parametrize(
    ("backend", "runtime_model", "repo_id"),
    [
        ("parakeet-redux", "moondream/parakeet-redux", "moondream/parakeet-redux"),
        # Bark checkpoints are the Parakeet TDT 0.6B v3 architecture, which
        # Kestrel registers under NVIDIA's model id.
        ("bark", "nvidia/parakeet-tdt-0.6b-v3", "ConwayResearch/Underdog-Bark-3.0"),
        (
            "bark-pro",
            "nvidia/parakeet-tdt-0.6b-v3",
            "ConwayResearch/Underdog-Bark-3.0-Pro",
        ),
    ],
)
def test_each_local_backend_loads_its_own_checkpoint(
    monkeypatch, backend, runtime_model, repo_id
):
    configurations = []
    resolved = []
    monkeypatch.setattr(
        model_module,
        "_load_runtime_dependencies",
        lambda: (
            lambda config: configurations.append(config) or SimpleNamespace(),
            SimpleNamespace,
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        model_module,
        "_resolve_model_path",
        lambda spec: resolved.append(spec.repo_id) or f"/cached/{backend}",
    )

    model_module.load_engine(DictationConfig(backend=backend, cpu=True))

    assert resolved == [repo_id]
    assert vars(configurations[0]) == {
        "model": runtime_model,
        "model_path": f"/cached/{backend}",
        "device": "cpu",
    }


def test_load_engine_rejects_a_backend_kestrel_does_not_serve():
    with pytest.raises(ValueError, match="'whisper'"):
        model_module.load_engine(DictationConfig(backend="whisper"))


@pytest.mark.parametrize(
    ("backend", "revision", "files"),
    [
        # Kestrel marks a checkpoint as ternary only when ternary.json exists.
        (
            "parakeet-redux",
            None,
            ["config.json", "tokenizer.json", "model.safetensors", "ternary.json"],
        ),
        (
            "bark-pro",
            "4b5c51e04aa8d16dda9ecdff71bc29f7360149e4",
            ["config.json", "tokenizer.json", "model.safetensors"],
        ),
    ],
)
def test_snapshot_downloads_request_each_checkpoints_files_and_pin(
    tmp_path, monkeypatch, backend, revision, files
):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    requests = []

    def download(repo_id, *, revision, allow_patterns, local_files_only=False):
        requests.append((repo_id, revision, allow_patterns, local_files_only))
        if local_files_only:
            raise LocalEntryNotFoundError("no local snapshot")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    spec = model_module.KESTREL_MODELS[backend]

    assert model_module._resolve_model_path(spec) == str(tmp_path)
    assert requests == [
        (spec.repo_id, revision, files, True),
        (spec.repo_id, revision, files, False),
    ]


def test_pinned_cache_check_ignores_snapshots_of_other_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "_check_import_readiness", lambda: (True, None))
    snapshots = (
        tmp_path
        / ".cache/huggingface/hub/models--ConwayResearch--Underdog-Bark-3.0-Pro"
        / "snapshots"
    )
    for revision in ("0000000000000000000000000000000000000000",):
        (snapshots / revision).mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (snapshots / revision / name).write_text("{}", encoding="utf-8")

    stale = check_model_cache(home=tmp_path, backend="bark-pro")
    pinned = snapshots / "4b5c51e04aa8d16dda9ecdff71bc29f7360149e4"
    pinned.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (pinned / name).write_text("{}", encoding="utf-8")
    ready = check_model_cache(home=tmp_path, backend="bark-pro")

    assert stale["cache_present"] is False
    assert ready["cache_present"] is True
    assert ready["model_id"] == "ConwayResearch/Underdog-Bark-3.0-Pro"


def test_bark_cache_check_reads_the_decoded_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "_check_import_readiness", lambda: (True, None))
    decoded = (
        tmp_path
        / ".cache/local-ai-dictation/models/ConwayResearch--Underdog-Bark-3.0"
        / "85db2ad18ab41e534a222ee253c29191d77ed919"
    )

    missing = check_model_cache(home=tmp_path, backend="bark")
    decoded.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (decoded / name).write_text("{}", encoding="utf-8")
    ready = check_model_cache(home=tmp_path, backend="bark")

    assert missing["cache_present"] is False
    assert missing["cache_path"] is None
    assert ready == {
        "checked": True,
        "cache_present": True,
        "cache_path": str(decoded),
        "model_id": "ConwayResearch/Underdog-Bark-3.0",
        "import_ready": True,
        "import_error": None,
        "detail": "Local cache and model imports look ready",
    }


def test_incomplete_ultra_cache_is_not_ready(tmp_path, monkeypatch):
    snapshot = (
        tmp_path
        / ".cache/huggingface/hub/models--moondream--parakeet-ultra/snapshots/revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model_module, "_check_import_readiness", lambda: (True, None))

    assert check_model_cache(home=tmp_path)["cache_present"] is False


def test_ultra_runtime_load_errors_keep_the_model_error_contract(monkeypatch):
    def fail(config):
        raise RuntimeError("weights cannot be loaded")

    monkeypatch.setattr(
        model_module,
        "_load_runtime_dependencies",
        lambda: (fail, SimpleNamespace, object()),
    )
    monkeypatch.setattr(
        model_module, "_resolve_model_path", lambda spec: "/cached/ultra"
    )

    with pytest.raises(ModelError, match="weights cannot be loaded") as error:
        model_module.load_engine(DictationConfig(backend="parakeet", cpu=True))
    assert error.value.code == MODEL_IMPORT_FAILED


def test_ultra_runtime_result_errors_keep_the_transcription_error_contract():
    runtime = SimpleNamespace(forward=lambda task, inputs: [ValueError("invalid WAV")])
    engine = model_module.ParakeetTdtEngine(runtime, device="cpu")

    with pytest.raises(ModelError, match="invalid WAV") as error:
        model_module.transcribe_wav(engine, "broken.wav")
    assert error.value.code == MODEL_TRANSCRIBE_FAILED


def test_runtime_dependencies_use_the_exported_model_not_the_telemetry_wrapper(
    monkeypatch,
):
    runtime_class = object()
    config_class = object()
    torch_module = object()
    allowed_modules = {
        "kestrel.models.parakeet_tdt": SimpleNamespace(
            ParakeetTdtRuntime=runtime_class
        ),
        "kestrel.config": SimpleNamespace(RuntimeConfig=config_class),
        "torch": torch_module,
    }
    monkeypatch.setattr(
        model_module.importlib, "import_module", allowed_modules.__getitem__
    )

    assert model_module._load_runtime_dependencies() == (
        runtime_class,
        config_class,
        torch_module,
    )


@pytest.mark.parametrize(
    "cached_files",
    [(), ("config.json",), ("config.json", "tokenizer.json", "model.safetensors")],
)
def test_checkpoint_resolution_downloads_only_when_cache_is_missing_or_partial(
    tmp_path, monkeypatch, cached_files
):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = []
    for name in cached_files:
        (tmp_path / name).write_text("cached", encoding="utf-8")

    def download(model_id, *, revision, allow_patterns, local_files_only=False):
        calls.append(local_files_only)
        assert model_id == "moondream/parakeet-ultra"
        assert revision is None
        assert allow_patterns == ["config.json", "tokenizer.json", "model.safetensors"]
        if local_files_only and not cached_files:
            raise LocalEntryNotFoundError("no local snapshot")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    assert model_module._resolve_model_path(
        model_module.KESTREL_MODELS["parakeet"]
    ) == str(tmp_path)
    assert calls == ([True] if len(cached_files) == 3 else [True, False])


def test_direct_runtime_serializes_calls_and_preserves_each_transcript():
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_entered_runtime = threading.Event()
    transcripts = []

    def forward(task, inputs):
        assert task == "transcribe"
        path = inputs[0]["audio"]
        if path == "warmup.wav":
            entered.set()
            assert release.wait(timeout=3)
        else:
            second_entered_runtime.set()
        return [{"text": path}]

    engine = model_module.ParakeetTdtEngine(
        SimpleNamespace(forward=forward), device="cuda"
    )

    def transcribe(path):
        if path == "recording.wav":
            second_started.set()
        transcripts.append(engine.transcribe([path])[0].text)

    first = threading.Thread(target=transcribe, args=("warmup.wav",))
    second = threading.Thread(target=transcribe, args=("recording.wav",))
    first.start()
    try:
        assert entered.wait(timeout=3)
        second.start()
        assert second_started.wait(timeout=3)
        assert not second_entered_runtime.wait(timeout=0.05)
    finally:
        release.set()
        first.join(timeout=3)
        if second.ident is not None:
            second.join(timeout=3)

    assert sorted(transcripts) == ["recording.wav", "warmup.wav"]


def test_direct_runtime_receives_file_paths_and_is_closed():
    calls = []
    runtime = SimpleNamespace(
        forward=lambda task, inputs: (
            calls.append((task, inputs)) or [{"text": "First."}, {"text": "Second."}]
        ),
        shutdown=lambda: calls.append("shutdown"),
    )
    engine = model_module.ParakeetTdtEngine(runtime, device="cpu")

    results = engine.transcribe([Path("first.wav"), "second.wav"])
    engine.close()

    assert [item.text for item in results] == ["First.", "Second."]
    assert all(item.device == "cpu" for item in results)
    assert calls == [
        ("transcribe", [{"audio": "first.wav"}, {"audio": "second.wav"}]),
        "shutdown",
    ]


def test_dictation_and_bridge_batch_path_load_and_transcribe_with_the_ultra_adapter(
    monkeypatch,
):
    dictation = importlib.import_module("local_ai_dictation.dictation")
    configurations = []
    observed_audio = []

    def forward(task, inputs):
        assert task == "transcribe"
        path = Path(inputs[0]["audio"])
        with wave.open(str(path)) as audio:
            observed_audio.append(
                (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
            )
        return [{"text": "A complete sentence."}]

    def build_runtime(config):
        configurations.append(config)
        return SimpleNamespace(forward=forward)

    monkeypatch.setattr(
        model_module,
        "_load_runtime_dependencies",
        lambda: (build_runtime, SimpleNamespace, object()),
    )
    monkeypatch.setattr(
        model_module, "_resolve_model_path", lambda spec: "/cached/ultra"
    )
    monkeypatch.setattr(dictation, "spinner_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(dictation, "has_probable_speech", lambda *args, **kwargs: True)
    config = DictationConfig(backend="parakeet", cpu=True, debug=True)
    engine, use_cuda, _, _ = dictation._load_model(
        config, model_module, object(), status_stream=io.StringIO()
    )

    result, path, _, _ = dictation._transcribe_once(
        config,
        engine,
        b"\x00\x00" * 160,
        16000,
        status_stream=io.StringIO(),
    )
    try:
        assert result.text == "A complete sentence."
        assert result.metadata == {"backend": "parakeet"}
        assert use_cuda is False
        assert configurations[0].model == "moondream/parakeet-ultra"
        assert observed_audio == [(16000, 1, 2)]
    finally:
        Path(path).unlink()


@pytest.mark.parametrize("backend", ["parakeet-redux", "bark", "bark-pro"])
def test_dictation_loads_every_local_backend_through_the_kestrel_adapter(
    monkeypatch, backend
):
    dictation = importlib.import_module("local_ai_dictation.dictation")
    imported = []
    monkeypatch.setattr(
        dictation.importlib,
        "import_module",
        lambda name: imported.append(name) or SimpleNamespace(),
    )
    runtime_module, *_ = dictation._load_runtime_dependencies(backend, True)
    configurations = []
    monkeypatch.setattr(
        model_module,
        "_load_runtime_dependencies",
        lambda: (
            lambda config: configurations.append(config) or SimpleNamespace(),
            SimpleNamespace,
            object(),
        ),
    )
    monkeypatch.setattr(model_module, "_resolve_model_path", lambda spec: "/cached")
    monkeypatch.setattr(dictation, "spinner_animation", lambda *args, **kwargs: None)

    engine, use_cuda, _, _ = dictation._load_model(
        DictationConfig(backend=backend, cpu=True, debug=True),
        model_module,
        object(),
        status_stream=io.StringIO(),
    )

    assert imported[0] == "local_ai_dictation.model"
    assert isinstance(runtime_module, SimpleNamespace)
    assert isinstance(engine, model_module.ParakeetTdtEngine)
    assert use_cuda is False
    assert (
        configurations[0].model == model_module.KESTREL_MODELS[backend].runtime_model_id
    )
