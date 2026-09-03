"""Diagnostics collection and rendering for AI Dictation."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Mapping, Sequence

from local_ai_dictation.audio import list_input_devices, probe_audio_backend
from local_ai_dictation.backend_state import get_backend
from local_ai_dictation.errors import (
    AUDIO_BACKEND_UNREACHABLE,
    AUDIO_NO_INPUT_DEVICE,
    CLIPBOARD_UNAVAILABLE,
    CUDA_UNAVAILABLE,
    MODEL_CACHE_MISSING,
    MODEL_IMPORT_FAILED,
    ExitCode,
)
from local_ai_dictation.gpu import nvidia_driver_loaded
from local_ai_dictation.model import MODEL_ID, check_model_cache
from local_ai_dictation.types import AudioDevice, DoctorIssue, DoctorReport


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _detect_wsl(
    env: Mapping[str, str] | None = None,
    *,
    proc_version_path: Path = Path("/proc/version"),
    osrelease_path: Path = Path("/proc/sys/kernel/osrelease"),
    wslg_socket_path: Path = Path("/mnt/wslg/PulseServer"),
) -> dict[str, Any]:
    source = os.environ if env is None else env
    detected_via: list[str] = []

    if "microsoft" in _read_text(proc_version_path).lower():
        detected_via.append("proc_version")
    if "microsoft" in _read_text(osrelease_path).lower():
        detected_via.append("osrelease")
    if source.get("WSL_DISTRO_NAME"):
        detected_via.append("env")

    return {
        "is_wsl": bool(detected_via),
        "has_wslg_socket": wslg_socket_path.exists(),
        "detected_via": detected_via,
    }


def _collect_env(source: Mapping[str, str] | None = None) -> dict[str, str | None]:
    env = os.environ if source is None else source
    return {
        "pulse_server": env.get("PULSE_SERVER"),
        "display": env.get("DISPLAY"),
        "wayland_display": env.get("WAYLAND_DISPLAY"),
    }


def _collect_clipboard_status() -> dict[str, Any]:
    try:
        pyperclip = importlib.import_module("pyperclip")
    except ModuleNotFoundError:
        return {"status": "missing", "backend": None}
    except Exception as exc:
        return {"status": "unavailable", "backend": "pyperclip", "detail": str(exc)}

    determine_clipboard = getattr(pyperclip, "determine_clipboard", None)
    if not callable(determine_clipboard):
        return {"status": "ok", "backend": "pyperclip"}

    try:
        copy_impl, _ = determine_clipboard()
    except Exception as exc:
        return {"status": "unavailable", "backend": "pyperclip", "detail": str(exc)}

    backend_name = getattr(copy_impl, "__name__", "copy_unknown")
    if backend_name == "copy_no":
        return {"status": "missing", "backend": None}
    if backend_name == "copy_wsl" and shutil.which("clip.exe") is None:
        return {"status": "unavailable", "backend": "pyperclip-wsl", "detail": "clip.exe is not available"}

    backend = backend_name.removeprefix("copy_") or "pyperclip"
    return {"status": "ok", "backend": backend}


def _collect_cuda_status() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return {
            "available": False,
            "selected_device": "cpu",
            "device_name": None,
            "detail": "torch is not installed",
        }
    except Exception as exc:
        return {
            "available": False,
            "selected_device": "cpu",
            "device_name": None,
            "detail": str(exc),
        }

    try:
        available = nvidia_driver_loaded() and bool(torch.cuda.is_available())
    except Exception as exc:
        return {
            "available": False,
            "selected_device": "cpu",
            "device_name": None,
            "detail": str(exc),
        }

    if available:
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            device_name = None
        return {
            "available": True,
            "selected_device": "cuda",
            "device_name": device_name,
        }

    return {
        "available": False,
        "selected_device": "cpu",
        "device_name": None,
        "detail": "CUDA runtime is unavailable; CPU fallback remains usable",
    }


def _collect_model_status(check_model_cache_enabled: bool, backend: str) -> dict[str, Any]:
    if backend == "whisper":
        from local_ai_dictation.whisper import WHISPER_MODEL_ID, check_model_cache as selected_check

        model_id = WHISPER_MODEL_ID
    elif backend == "willow":
        from local_ai_dictation.willow import check_model_cache as selected_check

        model_id = "willow/frontier-auto"
    else:
        selected_check = check_model_cache
        model_id = MODEL_ID

    if not check_model_cache_enabled:
        return {
            "checked": False,
            "cache_present": None,
            "model_id": model_id,
            "backend": backend,
        }

    return {**selected_check(), "backend": backend}


def _build_issues(
    *,
    pulse: Mapping[str, Any],
    audio_devices: Sequence[AudioDevice],
    clipboard: Mapping[str, Any],
    cuda: Mapping[str, Any],
    model: Mapping[str, Any],
    device_error: Exception | None,
    backend: str,
) -> list[DoctorIssue]:
    issues: list[DoctorIssue] = []

    if not audio_devices:
        issues.append(
            DoctorIssue(
                code=AUDIO_NO_INPUT_DEVICE,
                severity="fail",
                message="No audio input devices were detected.",
                remediation="Check microphone routing, Pulse/ALSA configuration, or device permissions.",
            )
        )

    if (pulse.get("status") == "unreachable" and not audio_devices) or device_error is not None:
        detail = pulse.get("detail") or (str(device_error) if device_error is not None else "Audio backend probe failed")
        issues.append(
            DoctorIssue(
                code=AUDIO_BACKEND_UNREACHABLE,
                severity="fail",
                message=f"Audio backend appears unreachable: {detail}",
                remediation="Verify PulseAudio/WSLg connectivity and that the selected backend is reachable from this shell.",
            )
        )

    if clipboard.get("status") != "ok":
        issues.append(
            DoctorIssue(
                code=CLIPBOARD_UNAVAILABLE,
                severity="warn",
                message="Clipboard copy is unavailable in this environment.",
                remediation="Install xclip/wl-clipboard or disable clipboard copying with --no-clipboard.",
            )
        )

    if backend != "willow" and not bool(cuda.get("available")):
        issues.append(
            DoctorIssue(
                code=CUDA_UNAVAILABLE,
                severity="warn",
                message="CUDA is unavailable; dictation can still run on CPU.",
                remediation="Install a CUDA-enabled PyTorch build and verify NVIDIA driver access if GPU inference is required.",
            )
        )

    if model.get("checked") and not bool(model.get("import_ready", True)):
        detail = model.get("import_error") or model.get("detail") or "Model imports failed"
        issues.append(
            DoctorIssue(
                code=MODEL_IMPORT_FAILED,
                severity="fail",
                message=f"Model imports are not ready: {detail}",
                remediation=(
                    "Run `local-ai-dictation willow-session login`, import a fresh session, or provide WILLOW_ACCESS_TOKEN."
                    if backend == "willow"
                    else "Install the required runtime dependencies for the selected local model."
                ),
            )
        )

    if model.get("checked") and model.get("cache_present") is False:
        detail = model.get("cache_path") or model.get("detail") or "Model cache is missing"
        issues.append(
            DoctorIssue(
                code=MODEL_CACHE_MISSING,
                severity="fail",
                message=f"Local model cache is missing: {detail}",
                remediation="Populate the local Hugging Face cache for the configured model before relying on offline readiness checks.",
            )
        )

    return issues


def _status_from_issues(issues: Sequence[DoctorIssue]) -> dict[str, Any]:
    if any(issue.severity == "fail" for issue in issues):
        overall = "fail"
        exit_code = int(ExitCode.RECORDING_BLOCKED)
    elif issues:
        overall = "warn"
        exit_code = int(ExitCode.DEGRADED)
    else:
        overall = "ok"
        exit_code = int(ExitCode.OK)

    return {
        "overall": overall,
        "exit_code": exit_code,
        "issues": [asdict(issue) for issue in issues],
    }


def collect_doctor_report(check_model_cache: bool = False, *, backend: str | None = None) -> DoctorReport:
    env = _collect_env()
    wsl = _detect_wsl()
    pulse = probe_audio_backend()

    device_error: Exception | None = None
    try:
        audio_devices = list_input_devices()
    except Exception as exc:
        audio_devices = []
        device_error = exc

    clipboard = _collect_clipboard_status()
    cuda = _collect_cuda_status()
    selected_backend = get_backend() if backend in {None, ""} else str(backend)
    model = _collect_model_status(check_model_cache, selected_backend)
    issues = _build_issues(
        pulse=pulse,
        audio_devices=audio_devices,
        clipboard=clipboard,
        cuda=cuda,
        model=model,
        device_error=device_error,
        backend=selected_backend,
    )

    return DoctorReport(
        platform={
            "system": platform.system(),
            "release": platform.release(),
        },
        wsl=wsl,
        env=env,
        pulse=pulse,
        audio_devices=list(audio_devices),
        clipboard=clipboard,
        cuda=cuda,
        model=model,
        status=_status_from_issues(issues),
    )


def render_doctor_text(report: DoctorReport) -> str:
    lines = [
        "AI Dictation doctor",
        f"overall: {report.status['overall']}",
        f"platform: {report.platform.get('system', 'unknown')} {report.platform.get('release', '')}".rstrip(),
        f"pulse: {report.pulse.get('status', 'unknown')} ({report.pulse.get('detail', 'no detail')})",
        f"audio_devices: {len(report.audio_devices)}",
        f"clipboard: {report.clipboard.get('status', 'unknown')}",
        f"cuda: {'available' if report.cuda.get('available') else 'unavailable'}",
        (
            f"model: {'checked' if report.model.get('checked') else 'skipped'}"
            if not report.model.get('checked')
            else f"model: checked cache_present={report.model.get('cache_present')} import_ready={report.model.get('import_ready')}"
        ),
    ]

    issues = report.status.get("issues", [])
    if issues:
        lines.append("issues:")
        for issue in issues:
            lines.append(f"- [{issue['severity']}] {issue['code']}: {issue['message']}")
    else:
        lines.append("issues: none")

    return "\n".join(lines)


def doctor_exit_code(report: DoctorReport) -> int:
    return int(report.status.get("exit_code", ExitCode.ERROR))
