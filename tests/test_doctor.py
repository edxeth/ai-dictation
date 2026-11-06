from __future__ import annotations

import json

import pytest

from local_ai_dictation.cli import main
from local_ai_dictation.doctor import collect_doctor_report, doctor_exit_code
from local_ai_dictation.model import MODEL_ID
from local_ai_dictation.types import AudioDevice

_DEFAULT_WSL = {
    "is_wsl": False,
    "has_wslg_socket": False,
    "detected_via": [],
}


def _stub_common(monkeypatch):
    monkeypatch.setattr(
        "local_ai_dictation.doctor._detect_wsl", lambda: dict(_DEFAULT_WSL)
    )
    monkeypatch.setattr("local_ai_dictation.doctor.get_backend", lambda: "parakeet")


def _stub_ready_runtime(monkeypatch):
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.probe_audio_backend",
        lambda: {
            "status": "reachable",
            "transport": "tcp",
            "detail": "pactl info succeeded",
        },
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor.list_input_devices",
        lambda: [
            AudioDevice(
                id=2,
                name="Microphone",
                default_sample_rate=48000,
                max_input_channels=1,
                host_api="ALSA",
                is_default_candidate=True,
            )
        ],
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_clipboard_status",
        lambda: {"status": "ok", "backend": "pyperclip"},
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_cuda_status",
        lambda: {
            "available": True,
            "selected_device": "cuda",
            "device_name": "NVIDIA Test GPU",
        },
    )


def test_collect_doctor_report_ok_when_audio_and_runtime_are_ready(monkeypatch):
    _stub_ready_runtime(monkeypatch)

    report = collect_doctor_report()

    assert report.schema_version == 1
    assert report.model == {
        "checked": False,
        "cache_present": None,
        "model_id": MODEL_ID,
        "backend": "parakeet",
    }
    assert report.status["overall"] == "ok"
    assert report.status["exit_code"] == 0
    assert report.status["issues"] == []
    assert doctor_exit_code(report) == 0


def test_collect_doctor_report_skips_model_check_by_default(monkeypatch):
    _stub_ready_runtime(monkeypatch)

    def _unexpected_model_check():
        raise AssertionError("default doctor mode should not check the model cache")

    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache", _unexpected_model_check
    )

    report = collect_doctor_report()

    assert report.model == {
        "checked": False,
        "cache_present": None,
        "model_id": MODEL_ID,
        "backend": "parakeet",
    }
    assert report.status["overall"] == "ok"


@pytest.mark.parametrize(
    ("backend", "model_id"),
    [
        ("parakeet-redux", "moondream/parakeet-redux"),
        ("bark", "ConwayResearch/Underdog-Bark-3.0"),
        ("bark-pro", "ConwayResearch/Underdog-Bark-3.0-Pro"),
    ],
)
def test_collect_doctor_report_checks_the_selected_local_model(
    monkeypatch, backend, model_id
):
    _stub_ready_runtime(monkeypatch)
    checked = []

    def check(*, backend):
        checked.append(backend)
        return {"checked": True, "cache_present": True, "model_id": model_id}

    monkeypatch.setattr("local_ai_dictation.doctor.check_model_cache", check)

    unchecked = collect_doctor_report(backend=backend)
    report = collect_doctor_report(True, backend=backend)

    assert unchecked.model == {
        "checked": False,
        "cache_present": None,
        "model_id": model_id,
        "backend": backend,
    }
    assert checked == [backend]
    assert report.model["backend"] == backend
    assert report.model["model_id"] == model_id


def test_collect_doctor_report_targets_willow_backend(monkeypatch):
    _stub_ready_runtime(monkeypatch)

    report = collect_doctor_report(backend="willow")

    assert report.model == {
        "checked": False,
        "cache_present": None,
        "model_id": "willow/frontier-auto",
        "backend": "willow",
    }


def test_collect_doctor_report_fails_when_pulse_is_unreachable_and_no_devices_exist(
    monkeypatch,
):
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.probe_audio_backend",
        lambda: {
            "status": "unreachable",
            "transport": "unix",
            "detail": "Connection refused",
        },
    )
    monkeypatch.setattr("local_ai_dictation.doctor.list_input_devices", lambda: [])
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_clipboard_status",
        lambda: {"status": "ok", "backend": "pyperclip"},
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_cuda_status",
        lambda: {
            "available": True,
            "selected_device": "cuda",
            "device_name": "NVIDIA Test GPU",
        },
    )

    report = collect_doctor_report()
    issue_codes = {issue["code"] for issue in report.status["issues"]}

    assert report.status["overall"] == "fail"
    assert report.status["exit_code"] == 2
    assert issue_codes == {"AUDIO_BACKEND_UNREACHABLE", "AUDIO_NO_INPUT_DEVICE"}
    assert doctor_exit_code(report) == 2


def test_collect_doctor_report_warns_for_clipboard_and_cuda_without_blocking_recording(
    monkeypatch,
):
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.probe_audio_backend",
        lambda: {
            "status": "reachable",
            "transport": "tcp",
            "detail": "pactl info succeeded",
        },
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor.list_input_devices",
        lambda: [
            AudioDevice(
                id=7,
                name="USB Mic",
                default_sample_rate=44100,
                max_input_channels=1,
                host_api="PulseAudio",
            )
        ],
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_clipboard_status",
        lambda: {"status": "unavailable", "backend": "pyperclip"},
    )
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_cuda_status",
        lambda: {
            "available": False,
            "selected_device": "cpu",
            "device_name": None,
            "detail": "CUDA runtime is unavailable",
        },
    )

    report = collect_doctor_report()
    issue_codes = {issue["code"] for issue in report.status["issues"]}

    assert report.status["overall"] == "warn"
    assert report.status["exit_code"] == 3
    assert issue_codes == {"CLIPBOARD_UNAVAILABLE", "CUDA_UNAVAILABLE"}
    assert doctor_exit_code(report) == 3


_GGUF_READY = {
    "runtime": "transcribe.cpp",
    "device": "igpu",
    "model_id": "handy-computer/parakeet-ultra-gguf",
    "revision": "a1539be1b849ea7463b88343e5277b373d0c4ea9",
    "file": "parakeet-ultra-Q8_0.gguf",
    "cache_present": True,
    "cache_path": "/tmp/parakeet-ultra-Q8_0.gguf",
    "import_ready": True,
}


def _stub_cpu_fallback(monkeypatch, **overrides):
    monkeypatch.setattr(
        "local_ai_dictation.gguf.check_model_cache",
        lambda: {**_GGUF_READY, **overrides},
    )


def test_collect_doctor_report_check_model_cache_ok(monkeypatch):
    _stub_ready_runtime(monkeypatch)
    _stub_cpu_fallback(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda *, backend: {
            "checked": True,
            "cache_present": True,
            "cache_path": "/tmp/model-cache",
            "model_id": MODEL_ID,
            "import_ready": True,
            "import_error": None,
            "detail": "Local cache and model imports look ready",
        },
    )

    report = collect_doctor_report(check_model_cache=True)

    assert report.model == {
        "checked": True,
        "cache_present": True,
        "cache_path": "/tmp/model-cache",
        "model_id": MODEL_ID,
        "import_ready": True,
        "import_error": None,
        "detail": "Local cache and model imports look ready",
        "backend": "parakeet",
        "cpu_fallback": _GGUF_READY,
    }
    assert report.status["overall"] == "ok"
    assert report.status["exit_code"] == 0
    assert report.status["issues"] == []


def _stub_kestrel_cache_ready(monkeypatch):
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda *, backend: {
            "checked": True,
            "cache_present": True,
            "model_id": MODEL_ID,
            "import_ready": True,
        },
    )


def _stub_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(
        "local_ai_dictation.doctor._collect_cuda_status",
        lambda: {
            "available": False,
            "selected_device": "cpu",
            "device_name": None,
            "detail": "CUDA runtime is unavailable",
        },
    )


def test_missing_cpu_fallback_file_does_not_matter_while_cuda_runs_ultra(monkeypatch):
    _stub_ready_runtime(monkeypatch)
    _stub_kestrel_cache_ready(monkeypatch)
    _stub_cpu_fallback(monkeypatch, cache_present=False, cache_path=None)

    report = collect_doctor_report(check_model_cache=True)

    assert report.model["cpu_fallback"]["cache_present"] is False
    assert report.status["issues"] == []


def test_missing_cpu_fallback_file_fails_when_ultra_must_run_without_cuda(
    monkeypatch,
):
    _stub_ready_runtime(monkeypatch)
    _stub_cuda_unavailable(monkeypatch)
    _stub_kestrel_cache_ready(monkeypatch)
    _stub_cpu_fallback(monkeypatch, cache_present=False, cache_path=None)

    report = collect_doctor_report(check_model_cache=True)
    issues = {issue["code"]: issue for issue in report.status["issues"]}

    assert report.status["overall"] == "fail"
    assert set(issues) == {"CUDA_UNAVAILABLE", "MODEL_CACHE_MISSING"}
    assert "parakeet-ultra-Q8_0.gguf" in issues["MODEL_CACHE_MISSING"]["message"]


def test_missing_transcribe_cpp_fails_when_ultra_must_run_without_cuda(monkeypatch):
    _stub_ready_runtime(monkeypatch)
    _stub_cuda_unavailable(monkeypatch)
    _stub_kestrel_cache_ready(monkeypatch)
    _stub_cpu_fallback(monkeypatch, import_ready=False)

    report = collect_doctor_report(check_model_cache=True)
    issues = {issue["code"]: issue for issue in report.status["issues"]}

    assert set(issues) == {"CUDA_UNAVAILABLE", "MODEL_IMPORT_FAILED"}
    assert "transcribe-cpp" in issues["MODEL_IMPORT_FAILED"]["message"]


def test_collect_doctor_report_check_model_cache_surfaces_missing_cache_and_import_failures(
    monkeypatch,
):
    _stub_ready_runtime(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda *, backend: {
            "checked": True,
            "cache_present": False,
            "cache_path": None,
            "model_id": MODEL_ID,
            "import_ready": False,
            "import_error": "kestrel import failed",
            "detail": "Local cache is missing and model imports failed",
        },
    )

    report = collect_doctor_report(check_model_cache=True)
    issue_codes = {issue["code"] for issue in report.status["issues"]}

    assert report.status["overall"] == "fail"
    assert report.status["exit_code"] == 2
    assert issue_codes == {"MODEL_CACHE_MISSING", "MODEL_IMPORT_FAILED"}


def test_doctor_command_emits_json_schema(monkeypatch, capsys):
    _stub_ready_runtime(monkeypatch)

    exit_code = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema_version"] == 1
    assert payload["pulse"] == {
        "status": "reachable",
        "transport": "tcp",
        "detail": "pactl info succeeded",
    }
    assert payload["audio_devices"] == [
        {
            "id": 2,
            "name": "Microphone",
            "default_sample_rate": 48000,
            "max_input_channels": 1,
            "host_api": "ALSA",
            "is_default_candidate": True,
        }
    ]
    assert payload["model"] == {
        "checked": False,
        "cache_present": None,
        "model_id": MODEL_ID,
        "backend": "parakeet",
    }
    assert payload["status"] == {
        "overall": "ok",
        "exit_code": 0,
        "issues": [],
    }


def test_doctor_command_emits_model_check_json(monkeypatch, capsys):
    _stub_ready_runtime(monkeypatch)
    _stub_cpu_fallback(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda *, backend: {
            "checked": True,
            "cache_present": True,
            "cache_path": "/tmp/model-cache",
            "model_id": MODEL_ID,
            "import_ready": True,
            "import_error": None,
            "detail": "Local cache and model imports look ready",
        },
    )

    exit_code = main(["doctor", "--check-model-cache", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model"] == {
        "checked": True,
        "cache_present": True,
        "cache_path": "/tmp/model-cache",
        "model_id": MODEL_ID,
        "import_ready": True,
        "import_error": None,
        "detail": "Local cache and model imports look ready",
        "backend": "parakeet",
        "cpu_fallback": _GGUF_READY,
    }
    assert payload["status"] == {
        "overall": "ok",
        "exit_code": 0,
        "issues": [],
    }


def test_doctor_text_names_the_cpu_fallback_runtime_device_and_file(
    monkeypatch, capsys
):
    _stub_ready_runtime(monkeypatch)
    _stub_kestrel_cache_ready(monkeypatch)
    _stub_cpu_fallback(monkeypatch, cache_present=False)

    main(["doctor", "--check-model-cache"])

    assert (
        "cpu fallback: transcribe.cpp on igpu file=parakeet-ultra-Q8_0.gguf "
        "cache_present=False import_ready=True"
    ) in capsys.readouterr().out.splitlines()


def _stub_kestrel_cache_missing(monkeypatch):
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda *, backend: {
            "checked": True,
            "cache_present": False,
            "cache_path": None,
            "model_id": MODEL_ID,
            "import_ready": True,
        },
    )


def test_unused_kestrel_ultra_cache_does_not_fail_when_the_cpu_fallback_is_ready(
    monkeypatch,
):
    _stub_ready_runtime(monkeypatch)
    _stub_cuda_unavailable(monkeypatch)
    _stub_kestrel_cache_missing(monkeypatch)
    _stub_cpu_fallback(monkeypatch)

    report = collect_doctor_report(check_model_cache=True)

    assert {issue["code"] for issue in report.status["issues"]} == {"CUDA_UNAVAILABLE"}
    assert report.status["overall"] == "warn"


def test_missing_kestrel_ultra_cache_still_fails_while_cuda_runs_ultra(monkeypatch):
    _stub_ready_runtime(monkeypatch)
    _stub_kestrel_cache_missing(monkeypatch)
    _stub_cpu_fallback(monkeypatch)

    report = collect_doctor_report(check_model_cache=True)

    assert {issue["code"] for issue in report.status["issues"]} == {
        "MODEL_CACHE_MISSING"
    }
