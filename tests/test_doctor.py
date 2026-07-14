from __future__ import annotations

import json

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
    monkeypatch.setattr("local_ai_dictation.doctor._detect_wsl", lambda: dict(_DEFAULT_WSL))
    monkeypatch.setattr("local_ai_dictation.doctor.get_backend", lambda: "parakeet")


def _stub_ready_runtime(monkeypatch):
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.probe_audio_backend",
        lambda: {"status": "reachable", "transport": "tcp", "detail": "pactl info succeeded"},
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
        lambda: {"available": True, "selected_device": "cuda", "device_name": "NVIDIA Test GPU"},
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

    monkeypatch.setattr("local_ai_dictation.doctor.check_model_cache", _unexpected_model_check)

    report = collect_doctor_report()

    assert report.model == {
        "checked": False,
        "cache_present": None,
        "model_id": MODEL_ID,
        "backend": "parakeet",
    }
    assert report.status["overall"] == "ok"



def test_collect_doctor_report_targets_willow_backend(monkeypatch):
    _stub_ready_runtime(monkeypatch)

    report = collect_doctor_report(backend="willow")

    assert report.model == {
        "checked": False,
        "cache_present": None,
        "model_id": "willow/frontier-auto",
        "backend": "willow",
    }


def test_collect_doctor_report_fails_when_pulse_is_unreachable_and_no_devices_exist(monkeypatch):
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
        lambda: {"available": True, "selected_device": "cuda", "device_name": "NVIDIA Test GPU"},
    )

    report = collect_doctor_report()
    issue_codes = {issue["code"] for issue in report.status["issues"]}

    assert report.status["overall"] == "fail"
    assert report.status["exit_code"] == 2
    assert issue_codes == {"AUDIO_BACKEND_UNREACHABLE", "AUDIO_NO_INPUT_DEVICE"}
    assert doctor_exit_code(report) == 2



def test_collect_doctor_report_warns_for_clipboard_and_cuda_without_blocking_recording(monkeypatch):
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.probe_audio_backend",
        lambda: {"status": "reachable", "transport": "tcp", "detail": "pactl info succeeded"},
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



def test_collect_doctor_report_check_model_cache_ok(monkeypatch):
    _stub_ready_runtime(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda: {
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
    }
    assert report.status["overall"] == "ok"
    assert report.status["exit_code"] == 0
    assert report.status["issues"] == []



def test_collect_doctor_report_check_model_cache_surfaces_missing_cache_and_import_failures(monkeypatch):
    _stub_ready_runtime(monkeypatch)
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda: {
            "checked": True,
            "cache_present": False,
            "cache_path": None,
            "model_id": MODEL_ID,
            "import_ready": False,
            "import_error": "nemo import failed",
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
    monkeypatch.setattr(
        "local_ai_dictation.doctor.check_model_cache",
        lambda: {
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
    }
    assert payload["status"] == {
        "overall": "ok",
        "exit_code": 0,
        "issues": [],
    }
