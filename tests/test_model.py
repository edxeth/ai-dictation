from __future__ import annotations

from pathlib import Path

from local_ai_dictation.model import MODEL_ID, check_model_cache, warmup



def test_parakeet_warmup_runs_real_inference_and_removes_fixture():
    paths: list[Path] = []

    class Engine:
        def eval(self):
            return self

        def transcribe(self, audio, *, verbose=False):
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
        / "models--nvidia--parakeet-tdt-0.6b-v3"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("local_ai_dictation.model._check_import_readiness", lambda: (True, None))

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



def test_check_model_cache_reports_missing_cache_and_import_failure(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr("local_ai_dictation.model._check_import_readiness", lambda: (False, "nemo import failed"))

    status = check_model_cache(home=home)

    assert status == {
        "checked": True,
        "cache_present": False,
        "cache_path": None,
        "model_id": MODEL_ID,
        "import_ready": False,
        "import_error": "nemo import failed",
        "detail": "Local cache is missing and model imports failed",
    }
