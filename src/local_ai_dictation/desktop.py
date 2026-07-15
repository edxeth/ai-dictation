"""Desktop app launch helpers for Local AI Dictation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from local_ai_dictation.backend_state import get_backend


class DesktopAppError(RuntimeError):
    """Raised when the desktop GUI cannot be launched."""


DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 8765
DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS = 120.0
DEFAULT_GUI_AUTO_EXIT_MS = 1500
DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS = 10.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def desktop_app_dir(root: Path | None = None) -> Path:
    resolved_root = repo_root() if root is None else root
    return resolved_root / "desktop" / "electrobun"


def bridge_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def bridge_start_command(host: str, port: int, *, backend: str | None = None) -> str:
    resolved_backend = get_backend() if backend in {None, ""} else str(backend)
    command = f"local-ai-dictation bridge --host {host} --port {port}"
    if resolved_backend != "parakeet":
        command += f" --backend {resolved_backend}"
    return command


def ensure_desktop_app_available(app_dir: Path | None = None) -> Path:
    resolved_app_dir = desktop_app_dir() if app_dir is None else app_dir
    if not resolved_app_dir.exists():
        raise DesktopAppError(f"Local AI Dictation GUI app not found at {resolved_app_dir}")
    package_json = resolved_app_dir / "package.json"
    if not package_json.exists():
        raise DesktopAppError(f"Local AI Dictation GUI package manifest not found at {package_json}")
    return resolved_app_dir


def ensure_windows_interop_command(command_name: str) -> str:
    command_path = shutil.which(command_name)
    if command_path is None:
        raise DesktopAppError(f"Windows staging requires `{command_name}` to be available from WSL.")
    return command_path


def run_text_command(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise DesktopAppError(stderr or f"Command failed: {' '.join(command)}")
    return completed.stdout.strip()


def read_windows_env_var(name: str) -> str:
    cmd_exe = ensure_windows_interop_command("cmd.exe")
    value = run_text_command([cmd_exe, "/c", "echo", f"%{name}%"])
    if not value or value == f"%{name}%":
        raise DesktopAppError(f"Windows environment variable `{name}` is not available from WSL.")
    return value.splitlines()[-1].strip()


def wsl_path_from_windows(path: str) -> Path:
    wslpath = ensure_windows_interop_command("wslpath")
    return Path(run_text_command([wslpath, "-u", path]))


def windows_path_from_wsl(path: Path) -> str:
    wslpath = ensure_windows_interop_command("wslpath")
    return run_text_command([wslpath, "-w", str(path)])


def windows_stage_root() -> Path:
    local_appdata = read_windows_env_var("LOCALAPPDATA")
    return wsl_path_from_windows(local_appdata) / "LocalAIDictation" / "staging"


def windows_local_appdata_root() -> Path:
    return wsl_path_from_windows(read_windows_env_var("LOCALAPPDATA"))


def stage_windows_desktop_app(app_dir: Path | None = None) -> dict[str, str]:
    source_app_dir = ensure_desktop_app_available(app_dir)
    source_root = source_app_dir.parents[2]
    digest = hashlib.sha256(str(source_root).encode("utf-8")).hexdigest()[:12]
    stage_root = windows_stage_root() / f"{source_root.name}-{digest}"
    stage_app_dir = stage_root / "desktop" / "electrobun"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_app_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_app_dir,
        stage_app_dir,
        ignore=shutil.ignore_patterns("node_modules", ".tmp-check", "dist"),
    )
    return {
        "source_app_dir": str(source_app_dir),
        "stage_root": str(stage_root),
        "desktop_app_dir": str(stage_app_dir),
        "windows_stage_root": windows_path_from_wsl(stage_root),
        "windows_desktop_app_dir": windows_path_from_wsl(stage_app_dir),
    }


def ensure_gui_dependencies(app_dir: Path, bun_path: str) -> None:
    if (app_dir / "node_modules").exists():
        return
    print(f"Installing Local AI Dictation GUI dependencies in {app_dir}...")
    completed = subprocess.run([bun_path, "install"], cwd=app_dir, check=False)
    if completed.returncode != 0:
        raise DesktopAppError("Failed to install Local AI Dictation GUI dependencies with `bun install`.")


def default_gui_log_dir(env: dict[str, str] | None = None) -> Path | None:
    source = os.environ if env is None else env
    local_appdata = source.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "local-ai-dictation.desktop.local" / "stable" / "logs"
    xdg_state_home = source.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "local-ai-dictation" / "desktop"
    home = source.get("HOME")
    if home:
        return Path(home) / ".local" / "state" / "local-ai-dictation" / "desktop"
    return None


def build_gui_environment(
    host: str,
    port: int,
    *,
    bridge_command: str | None = None,
    hotkey: str | None = None,
    backend: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["LOCAL_AI_DICTATION_BRIDGE_URL"] = bridge_url(host, port)
    env["LOCAL_AI_DICTATION_BRIDGE_COMMAND"] = bridge_start_command(host, port, backend=backend) if bridge_command is None else bridge_command
    if hotkey:
        env["LOCAL_AI_DICTATION_HOTKEY"] = hotkey
    log_dir = default_gui_log_dir(env)
    if log_dir is not None:
        env.setdefault("LOCAL_AI_DICTATION_GUI_LOG_PATH", str(log_dir / "startup.log"))
        env.setdefault("LOCAL_AI_DICTATION_GUI_STARTUP_DIAGNOSTICS_PATH", str(log_dir / "startup-diagnostics.json"))
    return env


def ensure_bun_available() -> str:
    bun_path = shutil.which("bun")
    if bun_path is None:
        raise DesktopAppError("Native GUI launch requires `bun` to be installed and available on PATH.")
    return bun_path


def _terminate_existing_native_gui_processes() -> None:
    current_pid = os.getpid()
    patterns = [
        "local-ai-dictation gui --host",
        "bun run start",
        "electrobun dev",
        "local-ai-dictation-desktop-dev/bin/launcher",
        "local-ai-dictation-desktop-dev/bin/bun",
    ]
    for pattern in patterns:
        completed = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        if completed.returncode not in {0, 1}:
            continue
        for line in completed.stdout.splitlines():
            pid_text = line.strip()
            if not pid_text.isdigit():
                continue
            pid = int(pid_text)
            if pid == current_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue


def ensure_gui_bridge_running(namespace: Any) -> None:
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    port = int(getattr(namespace, "port", DEFAULT_BRIDGE_PORT))
    if bridge_healthy(host, port):
        return
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return

    subprocess.Popen(
        build_bridge_command(namespace),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not wait_for_bridge(host, port):
        raise DesktopAppError(f"Local AI Dictation bridge did not become ready at {bridge_url(host, port)}.")


def run_gui_command(namespace: Any) -> int:
    app_dir = ensure_desktop_app_available()
    bun_path = ensure_bun_available()
    ensure_gui_dependencies(app_dir, bun_path)
    ensure_gui_bridge_running(namespace)
    _terminate_existing_native_gui_processes()
    completed = subprocess.run(
        [bun_path, "run", "start"],
        cwd=app_dir,
        env=build_gui_environment(
            str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST)),
            int(getattr(namespace, "port", DEFAULT_BRIDGE_PORT)),
            bridge_command=getattr(namespace, "bridge_command", None),
            hotkey=getattr(namespace, "hotkey", None),
            backend=getattr(namespace, "backend", None),
        ),
        check=False,
    )
    return int(completed.returncode)


def bridge_healthy(host: str, port: int, *, timeout: float = 1.0) -> bool:
    health_url = f"{bridge_url(host, port)}/health"
    try:
        with urlopen(health_url, timeout=timeout) as response:
            return int(getattr(response, "status", 200)) == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def wait_for_bridge(host: str, port: int, *, timeout_seconds: float = 10.0, poll_interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if bridge_healthy(host, port, timeout=poll_interval):
            return True
        time.sleep(poll_interval)
    return False


def run_bridge_toggle_command(namespace: Any) -> int:
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    port = int(getattr(namespace, "port", DEFAULT_BRIDGE_PORT))
    request = Request(
        f"{bridge_url(host, port)}/session/toggle",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise DesktopAppError(detail or f"Bridge toggle request failed: {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise DesktopAppError(f"Bridge toggle request failed: {error}") from error

    if not isinstance(payload, dict):
        raise DesktopAppError("Bridge toggle response was not a JSON object.")

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
    else:
        print(str(payload.get("state", "ok")))
    return 0


def build_bridge_command(namespace: Any) -> list[str]:
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    port = int(getattr(namespace, "port", DEFAULT_BRIDGE_PORT))
    command = [
        sys.executable,
        "-m",
        "local_ai_dictation.cli",
        "bridge",
    ]
    namespace_backend = getattr(namespace, "backend", None)
    backend = get_backend() if namespace_backend in {None, ""} else str(namespace_backend)
    if backend != "parakeet":
        command.extend(["--backend", backend])
    command.extend([
        "--host",
        host,
        "--port",
        str(port),
        "--max-silence-ms",
        str(int(getattr(namespace, "max_silence_ms", 1200))),
        "--min-speech-ms",
        str(int(getattr(namespace, "min_speech_ms", 300))),
        "--vad-mode",
        str(int(getattr(namespace, "vad_mode", 2))),
        "--log-file",
        str(getattr(namespace, "log_file", "transcriber.debug.log")),
    ])
    if bool(getattr(namespace, "cpu", False)):
        command.append("--cpu")
    input_device = getattr(namespace, "input_device", None)
    if input_device is not None:
        command.extend(["--input-device", str(input_device)])
    if bool(getattr(namespace, "vad", False)):
        command.append("--vad")
    if not bool(getattr(namespace, "clipboard", True)):
        command.append("--no-clipboard")
    if bool(getattr(namespace, "debug", False)):
        command.append("--debug")
    return command


def run_gui_stage_command(namespace: Any) -> int:
    payload = stage_windows_desktop_app()
    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0
    print(f"Staged desktop app at {payload['desktop_app_dir']}")
    print(payload["windows_desktop_app_dir"])
    return 0


def ensure_windows_bun_available() -> str:
    cmd_exe = ensure_windows_interop_command("cmd.exe")
    output = run_text_command([cmd_exe, "/d", "/c", "where", "bun"])
    bun_path = output.splitlines()[0].strip() if output else ""
    if not bun_path:
        raise DesktopAppError("Windows packaging requires `bun.exe` to be available on the Windows PATH.")
    return bun_path


def find_windows_powershell() -> str | None:
    powershell_path = shutil.which("powershell.exe")
    if powershell_path is not None:
        return powershell_path
    fallback = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if fallback.exists():
        return str(fallback)
    return None


def run_windows_in_dir(windows_dir: str, commands: list[str]) -> None:
    powershell_path = find_windows_powershell()
    if powershell_path is not None:
        escaped_dir = windows_dir.replace("'", "''")
        script_parts = [f"Set-Location -LiteralPath '{escaped_dir}'"]
        for command in commands:
            script_parts.append(command)
            script_parts.append("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
        completed = subprocess.run(
            [
                powershell_path,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "; ".join(script_parts),
            ],
            check=False,
        )
    else:
        cmd_exe = ensure_windows_interop_command("cmd.exe")
        completed = subprocess.run(
            [cmd_exe, "/c", f"cd /d {windows_dir} && {' && '.join(commands)}"],
            check=False,
        )
    if completed.returncode != 0:
        raise DesktopAppError(f"Windows command failed in {windows_dir}: {' && '.join(commands)}")


def embed_windows_exe_icon(executable_path: Path, icon_path: Path, rcedit_path: Path) -> None:
    powershell_path = find_windows_powershell()
    if powershell_path is not None:
        completed = subprocess.run(
            [
                powershell_path,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"& '{_powershell_quote(windows_path_from_wsl(rcedit_path))}' '{_powershell_quote(windows_path_from_wsl(executable_path))}' --set-icon '{_powershell_quote(windows_path_from_wsl(icon_path))}'",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        completed = subprocess.run(
            [str(rcedit_path), str(executable_path), "--set-icon", str(icon_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DesktopAppError(detail or f"Failed to embed icon into {executable_path}")


def rebuild_windows_setup_archive_with_icons(archive_path: Path, icon_path: Path, rcedit_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="local-ai-dictation-icon-archive-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        completed = subprocess.run(
            ["tar", "--zstd", "-xf", str(archive_path), "-C", str(temp_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise DesktopAppError(detail or f"Failed to extract {archive_path}")

        extracted_roots = [path for path in temp_dir.iterdir() if path.is_dir()]
        if len(extracted_roots) != 1:
            raise DesktopAppError(f"Unexpected archive layout in {archive_path}")
        extracted_root = extracted_roots[0]

        for target in (extracted_root / "bin" / "launcher.exe", extracted_root / "bin" / "bun.exe"):
            if target.exists():
                embed_windows_exe_icon(target, icon_path, rcedit_path)

        rebuilt_archive_path = temp_dir / archive_path.name
        completed = subprocess.run(
            ["tar", "--zstd", "-cf", str(rebuilt_archive_path), "-C", str(temp_dir), extracted_root.name],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise DesktopAppError(detail or f"Failed to rebuild {archive_path}")

        shutil.move(str(rebuilt_archive_path), str(archive_path))


def rebuild_windows_artifact_zip(app_dir: Path) -> None:
    artifacts_dir = app_dir / "artifacts"
    zip_path = next(artifacts_dir.glob("stable-win-x64-*-Setup.zip"), None)
    if zip_path is None:
        return

    build_dir = app_dir / "build" / "stable-win-x64"
    installer_path = next(build_dir.glob("*-Setup.exe"), None)
    metadata_path = next(build_dir.glob("*-Setup.metadata.json"), None)
    archive_path = next(build_dir.glob("*-Setup.tar.zst"), None)
    if installer_path is None or metadata_path is None or archive_path is None:
        return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(metadata_path, f".installer\\{metadata_path.name}")
        bundle.write(archive_path, f".installer\\{archive_path.name}")
        bundle.write(installer_path, installer_path.name)


def apply_windows_packaged_icon_workaround(app_dir: Path) -> None:
    icon_path = app_dir / "src" / "mainview" / "assets" / "local-ai-dictation-icon.ico"
    rcedit_path = app_dir / "node_modules" / "rcedit" / "bin" / "rcedit-x64.exe"
    build_dir = app_dir / "build" / "stable-win-x64"
    if not icon_path.exists() or not rcedit_path.exists() or not build_dir.exists():
        return

    targets: list[Path] = []
    bundle_bin_dir = build_dir / "local-ai-dictation-desktop" / "bin"
    targets.extend(sorted(bundle_bin_dir.glob("*.exe")))
    targets.extend(
        [
            bundle_bin_dir / "launcher",
            bundle_bin_dir / "launcher.exe",
            bundle_bin_dir / "bun.exe",
        ]
    )
    targets.extend(sorted(build_dir.glob("*.exe")))

    seen: set[Path] = set()
    for target in targets:
        if target.exists() and target not in seen:
            embed_windows_exe_icon(target, icon_path, rcedit_path)
            seen.add(target)

    archive_path = next(build_dir.glob("*-Setup.tar.zst"), None)
    if archive_path is not None:
        rebuild_windows_setup_archive_with_icons(archive_path, icon_path, rcedit_path)
        rebuild_windows_artifact_zip(app_dir)


def _read_package_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DesktopAppError(f"Windows package metadata at {metadata_path} is invalid.")
    return payload



def collect_windows_package_artifacts(app_dir: Path) -> dict[str, Any]:
    build_dir = app_dir / "build" / "stable-win-x64"
    if not build_dir.exists():
        raise DesktopAppError(f"Windows build output not found at {build_dir}")

    installer_path = next(build_dir.glob("*-Setup.exe"), None)
    if installer_path is None:
        raise DesktopAppError(f"Windows installer not found in {build_dir}")

    metadata_path = next(build_dir.glob("*-Setup.metadata.json"), None)
    if metadata_path is None:
        raise DesktopAppError(f"Windows installer metadata not found in {build_dir}")

    archive_path = next(build_dir.glob("*-Setup.tar.zst"), None)
    if archive_path is None:
        raise DesktopAppError(f"Windows installer archive not found in {build_dir}")

    zip_path = next((app_dir / "artifacts").glob("stable-win-x64-*-Setup.zip"), None)
    if zip_path is None:
        raise DesktopAppError(f"Windows packaged artifact zip not found in {app_dir / 'artifacts'}")

    metadata = _read_package_metadata(metadata_path)
    identifier = str(metadata.get("identifier", ""))
    channel = str(metadata.get("channel", "stable"))
    app_name = str(metadata.get("name", "local-ai-dictation-desktop"))
    if not identifier:
        raise DesktopAppError(f"Windows installer metadata at {metadata_path} does not include an identifier.")

    return {
        "build_dir": str(build_dir),
        "windows_build_dir": windows_path_from_wsl(build_dir),
        "installer_path": str(installer_path),
        "windows_installer_path": windows_path_from_wsl(installer_path),
        "setup_archive_path": str(archive_path),
        "windows_setup_archive_path": windows_path_from_wsl(archive_path),
        "metadata_path": str(metadata_path),
        "windows_metadata_path": windows_path_from_wsl(metadata_path),
        "artifact_zip_path": str(zip_path),
        "windows_artifact_zip_path": windows_path_from_wsl(zip_path),
        "app_identifier": identifier,
        "app_channel": channel,
        "app_name": app_name,
    }


def build_windows_package_payload() -> dict[str, Any]:
    ensure_windows_bun_available()
    payload = stage_windows_desktop_app()
    app_dir = Path(payload["desktop_app_dir"])
    run_windows_in_dir(payload["windows_desktop_app_dir"], ["bun install", "bunx electrobun build --env=stable"])
    apply_windows_packaged_icon_workaround(app_dir)
    payload.update(collect_windows_package_artifacts(app_dir))
    return payload


def run_gui_package_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0
    print(f"Packaged Windows desktop app at {payload['installer_path']}")
    print(payload["windows_installer_path"])
    return 0


def run_wsl_windows_executable_capture(
    executable_path: Path,
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [str(executable_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        env=merged_env,
        cwd=None if cwd is None else str(cwd),
    )


def launch_wsl_windows_executable(
    executable_path: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.Popen[str]:
    powershell_path = find_windows_powershell()
    if powershell_path is not None and str(executable_path).startswith("/mnt/"):
        script_parts: list[str] = []
        if cwd is not None:
            script_parts.append(f"Set-Location -LiteralPath '{_powershell_quote(windows_path_from_wsl(cwd))}'")
        for name, value in (env or {}).items():
            script_parts.append(f"$env:{name} = '{_powershell_quote(value)}'")
        script_parts.append(f"& '{_powershell_quote(windows_path_from_wsl(executable_path))}'")
        return subprocess.Popen(
            [
                powershell_path,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "; ".join(script_parts),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.Popen(
        [str(executable_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
        cwd=None if cwd is None else str(cwd),
    )


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def installed_windows_app_paths(identifier: str, channel: str) -> dict[str, Path]:
    app_root = windows_local_appdata_root() / identifier / channel
    logs_dir = app_root / "logs"
    app_dir = app_root / "app"
    return {
        "app_root": app_root,
        "app_dir": app_dir,
        "launcher_path": app_dir / "bin" / "launcher.exe",
        "logs_dir": logs_dir,
        "diagnostics_path": logs_dir / "startup-diagnostics.json",
        "log_path": logs_dir / "startup.log",
    }


def build_gui_smoke_paths(stage_root: Path) -> dict[str, Path]:
    smoke_dir = stage_root / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    return {
        "smoke_dir": smoke_dir,
        "diagnostics_path": smoke_dir / "startup-diagnostics.json",
        "log_path": smoke_dir / "startup.log",
        "installer_stdout_path": smoke_dir / "installer.stdout.log",
        "installer_stderr_path": smoke_dir / "installer.stderr.log",
        "launcher_stdout_path": smoke_dir / "launcher.stdout.log",
        "launcher_stderr_path": smoke_dir / "launcher.stderr.log",
    }


def clear_existing_gui_startup_logs(installed_paths: dict[str, Path]) -> None:
    for key in ("diagnostics_path", "log_path"):
        path = installed_paths[key]
        if path.exists():
            path.unlink()


def _powershell_quote(value: str) -> str:
    return value.replace("'", "''")


def prepare_installed_windows_app_for_reinstall(installed_paths: dict[str, Path]) -> None:
    app_root = installed_paths["app_root"]
    powershell_path = find_windows_powershell()
    if powershell_path is not None:
        windows_app_root = windows_path_from_wsl(app_root)
        subprocess.run(
            [
                powershell_path,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "; ".join(
                    [
                        "$ErrorActionPreference = 'SilentlyContinue'",
                        f"$root = '{_powershell_quote(windows_app_root)}'",
                        "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
                    ]
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    for path in (installed_paths["app_dir"], app_root / "self-extraction"):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def wait_for_startup_readiness(path: Path, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if path.exists():
            try:
                payload = read_json_file(path)
            except json.JSONDecodeError:
                time.sleep(0.25)
                continue
            last_payload = payload
            if payload.get("bunReady") and payload.get("rendererReady") and payload.get("rendererRpcReady"):
                return payload
        time.sleep(0.25)
    if last_payload is not None:
        raise DesktopAppError(f"Packaged Windows app wrote incomplete startup diagnostics: {json.dumps(last_payload)}")
    raise DesktopAppError(f"Packaged Windows app did not write startup diagnostics to {path} within {timeout_seconds} seconds.")


def request_installed_gui_shutdown(installed_paths: dict[str, Path]) -> None:
    powershell_path = find_windows_powershell()
    if powershell_path is None:
        raise DesktopAppError("Packaged Windows app shutdown requires powershell.exe to locate the installed GUI process.")
    windows_app_root = windows_path_from_wsl(installed_paths["app_root"])
    completed = subprocess.run(
        [
            powershell_path,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "; ".join(
                [
                    "$ErrorActionPreference = 'SilentlyContinue'",
                    f"$root = '{_powershell_quote(windows_app_root)}'",
                    "$targets = Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase) }",
                    "$targets | ForEach-Object { $proc = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; if ($proc -and $proc.MainWindowHandle -ne 0) { $proc.CloseMainWindow() | Out-Null } }",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise DesktopAppError(stderr or "Failed to request a clean shutdown for the packaged Windows app.")


def wait_for_shutdown_reason(path: Path, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if path.exists():
            try:
                payload = read_json_file(path)
            except json.JSONDecodeError:
                time.sleep(0.25)
                continue
            last_payload = payload
            if payload.get("shutdownReason"):
                return payload
        time.sleep(0.25)
    if last_payload is not None:
        raise DesktopAppError(f"Packaged Windows app did not record a shutdown reason: {json.dumps(last_payload)}")
    raise DesktopAppError(f"Packaged Windows app did not update shutdown diagnostics at {path} within {timeout_seconds} seconds.")


def reserve_socket_localhost_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        return int(listener.getsockname()[1])


def reserve_localhost_port() -> int:
    powershell_path = find_windows_powershell()
    if powershell_path is not None:
        output = run_text_command(
            [
                powershell_path,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "; ".join(
                    [
                        "$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse('127.0.0.1'), 0)",
                        "$listener.Start()",
                        "$port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port",
                        "$listener.Stop()",
                        "Write-Output $port",
                    ]
                ),
            ]
        )
        return int(output.splitlines()[-1].strip())

    return reserve_socket_localhost_port()


def gui_e2e_base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def request_gui_e2e_json(
    port: int,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{gui_e2e_base_url(port)}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise DesktopAppError(detail or f"GUI automation request failed: {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise DesktopAppError(f"GUI automation request failed for {path}: {error}") from error

    decoded = json.loads(body) if body else {}
    if not isinstance(decoded, dict):
        raise DesktopAppError(f"GUI automation response at {path} was not a JSON object.")
    return decoded


def wait_for_gui_e2e_state(
    port: int,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: float = DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        payload = request_gui_e2e_json(port, "/state", timeout_seconds=poll_interval)
        state = payload.get("state")
        if isinstance(state, dict):
            last_state = state
            if predicate(state):
                return state
        time.sleep(poll_interval)
    if last_state is not None:
        raise DesktopAppError(f"GUI automation state did not reach the expected condition: {json.dumps(last_state)}")
    raise DesktopAppError(f"GUI automation state was unavailable on port {port} within {timeout_seconds} seconds.")


def invoke_gui_e2e_action(
    port: int,
    action: str,
    *,
    timeout_seconds: float = DEFAULT_GUI_E2E_ACTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        payload = request_gui_e2e_json(port, f"/actions/{action}", method="POST", payload={}, timeout_seconds=timeout_seconds)
    except DesktopAppError as error:
        if action in {"quit", "tray/quit"} and "Remote end closed connection without response" in str(error):
            return {"automationAction": action, "shutdownRequested": True}
        raise
    state = payload.get("state")
    if not isinstance(state, dict):
        raise DesktopAppError(f"GUI automation action `{action}` did not return state.")
    return state


def _tail_output(text: str, *, max_lines: int = 40) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    tail = lines[-max_lines:]
    prefix = "...\n" if len(lines) > max_lines else ""
    return prefix + "\n".join(tail)


def _decode_last_json_object(text: str, *, command: list[str]) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise DesktopAppError(
        f"Verification command {command!r} did not emit a JSON object. Stdout tail:\n{_tail_output(text)}"
    )


def run_repo_cli_json(command_args: list[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "local_ai_dictation.cli", *command_args, "--json"]
    completed = subprocess.run(
        command,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DesktopAppError(
            f"Verification command {command!r} failed with exit code {completed.returncode}. "
            f"Stdout tail:\n{_tail_output(completed.stdout)}\nStderr tail:\n{_tail_output(completed.stderr)}"
        )
    return _decode_last_json_object(completed.stdout, command=command)


def preserve_verification_artifacts(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    smoke_dir_value = payload.get("smoke_dir")
    stage_root_value = payload.get("stage_root")
    if not isinstance(smoke_dir_value, str) or not isinstance(stage_root_value, str):
        return payload

    smoke_dir = Path(smoke_dir_value)
    stage_root = Path(stage_root_value)
    if not smoke_dir.exists() or not stage_root.exists():
        return payload

    verify_dir = stage_root / "verify" / name
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    verify_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(smoke_dir, verify_dir)

    copied_payload = dict(payload)
    path_keys = [
        ("smoke_dir", "windows_smoke_dir"),
        ("diagnostics_path", "windows_diagnostics_path"),
        ("log_path", "windows_log_path"),
        ("installer_stdout_path", None),
        ("installer_stderr_path", None),
        ("launcher_stdout_path", None),
        ("launcher_stderr_path", None),
        ("bridge_stdout_path", None),
        ("bridge_stderr_path", None),
    ]
    for path_key, windows_key in path_keys:
        value = copied_payload.get(path_key)
        if not isinstance(value, str):
            continue
        source_path = Path(value)
        if source_path == smoke_dir:
            copied_path = verify_dir
        elif smoke_dir in source_path.parents:
            copied_path = verify_dir / source_path.relative_to(smoke_dir)
        else:
            continue
        copied_payload[path_key] = str(copied_path)
        if windows_key is not None:
            copied_payload[windows_key] = windows_path_from_wsl(copied_path)

    copied_payload["verify_dir"] = str(verify_dir)
    copied_payload["windows_verify_dir"] = windows_path_from_wsl(verify_dir)
    return copied_payload


def run_gui_package_verify_command(namespace: Any) -> int:
    timeout_seconds = float(getattr(namespace, "timeout_seconds", 240.0))
    check_commands = [
        ("smoke", ["gui-package-smoke", "--timeout-seconds", str(timeout_seconds)]),
        ("automation", ["gui-package-automation", "--timeout-seconds", str(timeout_seconds)]),
        ("bridge_recovery", ["gui-package-bridge-recovery", "--timeout-seconds", str(timeout_seconds)]),
        ("main_window", ["gui-package-main-window", "--timeout-seconds", str(timeout_seconds)]),
        ("tray", ["gui-package-tray", "--timeout-seconds", str(timeout_seconds)]),
        ("hotkey", ["gui-package-hotkey", "--timeout-seconds", str(timeout_seconds)]),
    ]
    checks: dict[str, dict[str, Any]] = {}
    for name, command_args in check_commands:
        checks[name] = preserve_verification_artifacts(name, run_repo_cli_json(command_args))

    smoke_payload = checks["smoke"]
    payload = {
        "command": "gui-package-verify",
        "timeout_seconds": timeout_seconds,
        "installer_path": smoke_payload.get("installer_path"),
        "windows_installer_path": smoke_payload.get("windows_installer_path"),
        "checks": checks,
    }

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print("Packaged Windows verification suite passed.")
    for name, result in checks.items():
        diagnostics_path = result.get("windows_diagnostics_path") or result.get("windows_log_path") or result.get("windows_smoke_dir")
        print(f"- {name}: {diagnostics_path}")
    return 0


def run_gui_package_automation_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    stage_root = Path(payload["stage_root"])
    smoke_paths = build_gui_smoke_paths(stage_root)
    timeout_seconds = float(getattr(namespace, "timeout_seconds", DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS))
    automation_port = int(getattr(namespace, "automation_port", 0)) or reserve_localhost_port()

    installed_paths = installed_windows_app_paths(str(payload["app_identifier"]), str(payload["app_channel"]))
    prepare_installed_windows_app_for_reinstall(installed_paths)
    clear_existing_gui_startup_logs(installed_paths)

    installer_completed = run_wsl_windows_executable_capture(
        Path(payload["installer_path"]),
        timeout_seconds=timeout_seconds,
    )
    smoke_paths["installer_stdout_path"].write_text(installer_completed.stdout, encoding="utf-8")
    smoke_paths["installer_stderr_path"].write_text(installer_completed.stderr, encoding="utf-8")
    if installer_completed.returncode != 0:
        raise DesktopAppError(
            f"Packaged Windows installer exited with code {installer_completed.returncode}. See {smoke_paths['installer_stdout_path']} and {smoke_paths['installer_stderr_path']}."
        )
    if not installed_paths["launcher_path"].exists():
        raise DesktopAppError(f"Installed Windows launcher not found at {installed_paths['launcher_path']}")

    launcher_process = launch_wsl_windows_executable(
        installed_paths["launcher_path"],
        env={
            "LOCAL_AI_DICTATION_GUI_E2E": "1",
            "LOCAL_AI_DICTATION_GUI_E2E_PORT": str(automation_port),
        },
    )
    initial_state: dict[str, Any] | None = None
    show_window_state: dict[str, Any] | None = None
    tray_open_state: dict[str, Any] | None = None
    quit_state: dict[str, Any] | None = None
    try:
        wait_for_startup_readiness(installed_paths["diagnostics_path"], timeout_seconds=timeout_seconds)
        initial_state = wait_for_gui_e2e_state(
            automation_port,
            lambda state: bool(state.get("diagnostics", {}).get("automationReady")),
            timeout_seconds=timeout_seconds,
        )
        show_window_state = invoke_gui_e2e_action(automation_port, "show-window")
        tray_open_state = invoke_gui_e2e_action(automation_port, "tray/open")
        quit_state = invoke_gui_e2e_action(automation_port, "quit")
        launcher_process.wait(timeout=10)
        diagnostics = wait_for_shutdown_reason(installed_paths["diagnostics_path"], timeout_seconds=10.0)
    finally:
        if launcher_process.poll() is None:
            try:
                launcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_process.terminate()
                try:
                    launcher_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()
                    launcher_process.wait(timeout=5)

    launcher_stdout, launcher_stderr = launcher_process.communicate()
    smoke_paths["launcher_stdout_path"].write_text(launcher_stdout, encoding="utf-8")
    smoke_paths["launcher_stderr_path"].write_text(launcher_stderr, encoding="utf-8")

    if installed_paths["log_path"].exists():
        shutil.copy2(installed_paths["log_path"], smoke_paths["log_path"])
    if installed_paths["diagnostics_path"].exists():
        shutil.copy2(installed_paths["diagnostics_path"], smoke_paths["diagnostics_path"])

    payload.update(
        {
            "smoke_dir": str(smoke_paths["smoke_dir"]),
            "windows_smoke_dir": windows_path_from_wsl(smoke_paths["smoke_dir"]),
            "diagnostics_path": str(smoke_paths["diagnostics_path"]),
            "windows_diagnostics_path": windows_path_from_wsl(smoke_paths["diagnostics_path"]),
            "log_path": str(smoke_paths["log_path"]),
            "windows_log_path": windows_path_from_wsl(smoke_paths["log_path"]),
            "installer_stdout_path": str(smoke_paths["installer_stdout_path"]),
            "installer_stderr_path": str(smoke_paths["installer_stderr_path"]),
            "launcher_stdout_path": str(smoke_paths["launcher_stdout_path"]),
            "launcher_stderr_path": str(smoke_paths["launcher_stderr_path"]),
            "installed_launcher_path": str(installed_paths["launcher_path"]),
            "windows_installed_launcher_path": windows_path_from_wsl(installed_paths["launcher_path"]),
            "launcher_exit_code": launcher_process.returncode,
            "automation_port": automation_port,
            "initial_state": initial_state,
            "show_window_state": show_window_state,
            "tray_open_state": tray_open_state,
            "quit_state": quit_state,
            "startup_diagnostics": diagnostics,
        }
    )

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print(f"Packaged Windows desktop app automation check passed with diagnostics at {payload['diagnostics_path']}")
    print(payload["windows_diagnostics_path"])
    return 0


def run_gui_package_bridge_recovery_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    stage_root = Path(payload["stage_root"])
    smoke_paths = build_gui_smoke_paths(stage_root)
    timeout_seconds = float(getattr(namespace, "timeout_seconds", DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS))
    automation_port = int(getattr(namespace, "automation_port", 0)) or reserve_localhost_port()
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    bridge_port = int(getattr(namespace, "bridge_port", 0)) or reserve_socket_localhost_port()
    expected_bridge_url = bridge_url(host, bridge_port)
    expected_bridge_command = bridge_start_command(host, bridge_port)
    bridge_stdout_path = smoke_paths["smoke_dir"] / "bridge.stdout.log"
    bridge_stderr_path = smoke_paths["smoke_dir"] / "bridge.stderr.log"

    installed_paths = installed_windows_app_paths(str(payload["app_identifier"]), str(payload["app_channel"]))
    prepare_installed_windows_app_for_reinstall(installed_paths)
    clear_existing_gui_startup_logs(installed_paths)

    installer_completed = run_wsl_windows_executable_capture(
        Path(payload["installer_path"]),
        timeout_seconds=timeout_seconds,
    )
    smoke_paths["installer_stdout_path"].write_text(installer_completed.stdout, encoding="utf-8")
    smoke_paths["installer_stderr_path"].write_text(installer_completed.stderr, encoding="utf-8")
    if installer_completed.returncode != 0:
        raise DesktopAppError(
            f"Packaged Windows installer exited with code {installer_completed.returncode}. See {smoke_paths['installer_stdout_path']} and {smoke_paths['installer_stderr_path']}."
        )
    if not installed_paths["launcher_path"].exists():
        raise DesktopAppError(f"Installed Windows launcher not found at {installed_paths['launcher_path']}")

    launcher_process = launch_wsl_windows_executable(
        installed_paths["launcher_path"],
        env={
            "LOCAL_AI_DICTATION_GUI_E2E": "1",
            "LOCAL_AI_DICTATION_GUI_E2E_PORT": str(automation_port),
            "LOCAL_AI_DICTATION_BRIDGE_URL": expected_bridge_url,
            "LOCAL_AI_DICTATION_BRIDGE_COMMAND": expected_bridge_command,
        },
    )
    bridge_process: subprocess.Popen[str] | None = None
    initial_state: dict[str, Any] | None = None
    recovered_state: dict[str, Any] | None = None
    quit_state: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    try:
        wait_for_startup_readiness(installed_paths["diagnostics_path"], timeout_seconds=timeout_seconds)

        def _offline_ready(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            if not isinstance(bridge_state, dict) or not isinstance(snapshot, dict):
                return False
            return (
                bridge_state.get("connected") is False
                and snapshot.get("toggleButtonText") == "Bridge offline"
                and snapshot.get("bridgeUrl") == expected_bridge_url
                and snapshot.get("bridgeCommand") == expected_bridge_command
                and "Start the bridge command below" in str(snapshot.get("statusLine", ""))
            )

        initial_state = wait_for_gui_e2e_state(
            automation_port,
            _offline_ready,
            timeout_seconds=timeout_seconds,
        )

        bridge_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_ai_dictation.cli",
                "bridge",
                "--host",
                host,
                "--port",
                str(bridge_port),
            ],
            cwd=repo_root(),
            env={
                **os.environ,
                "LOCAL_AI_DICTATION_E2E_MODE": "1",
                "LOCAL_AI_DICTATION_E2E_TRANSCRIPT": "offline online recovery transcript",
                "LOCAL_AI_DICTATION_E2E_START_DELAY_MS": "30",
                "LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS": "40",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not wait_for_bridge(host, bridge_port, timeout_seconds=10.0):
            exit_code = bridge_process.poll()
            raise DesktopAppError(
                f"Local AI Dictation bridge did not become ready at {expected_bridge_url} within 10 seconds (exit code {exit_code})."
            )

        def _recovered(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            if not isinstance(bridge_state, dict) or not isinstance(snapshot, dict):
                return False
            return (
                bridge_state.get("connected") is True
                and snapshot.get("bridgeUrl") == expected_bridge_url
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and "Model ready. Press the button or the hotkey to begin." in str(snapshot.get("statusLine", ""))
            )

        recovered_state = wait_for_gui_e2e_state(
            automation_port,
            _recovered,
            timeout_seconds=timeout_seconds,
        )
        quit_state = invoke_gui_e2e_action(automation_port, "quit")
        launcher_process.wait(timeout=10)
        diagnostics = wait_for_shutdown_reason(installed_paths["diagnostics_path"], timeout_seconds=10.0)
    finally:
        if bridge_process is not None:
            _stop_process(bridge_process)
        if launcher_process.poll() is None:
            try:
                launcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_process.terminate()
                try:
                    launcher_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()
                    launcher_process.wait(timeout=5)

    launcher_stdout, launcher_stderr = launcher_process.communicate()
    smoke_paths["launcher_stdout_path"].write_text(launcher_stdout, encoding="utf-8")
    smoke_paths["launcher_stderr_path"].write_text(launcher_stderr, encoding="utf-8")

    bridge_stdout = ""
    bridge_stderr = ""
    bridge_exit_code: int | None = None
    if bridge_process is not None:
        bridge_stdout, bridge_stderr = bridge_process.communicate()
        bridge_exit_code = bridge_process.returncode
    bridge_stdout_path.write_text(bridge_stdout, encoding="utf-8")
    bridge_stderr_path.write_text(bridge_stderr, encoding="utf-8")

    if installed_paths["log_path"].exists():
        shutil.copy2(installed_paths["log_path"], smoke_paths["log_path"])
    if installed_paths["diagnostics_path"].exists():
        shutil.copy2(installed_paths["diagnostics_path"], smoke_paths["diagnostics_path"])

    payload.update(
        {
            "smoke_dir": str(smoke_paths["smoke_dir"]),
            "windows_smoke_dir": windows_path_from_wsl(smoke_paths["smoke_dir"]),
            "diagnostics_path": str(smoke_paths["diagnostics_path"]),
            "windows_diagnostics_path": windows_path_from_wsl(smoke_paths["diagnostics_path"]),
            "log_path": str(smoke_paths["log_path"]),
            "windows_log_path": windows_path_from_wsl(smoke_paths["log_path"]),
            "installer_stdout_path": str(smoke_paths["installer_stdout_path"]),
            "installer_stderr_path": str(smoke_paths["installer_stderr_path"]),
            "launcher_stdout_path": str(smoke_paths["launcher_stdout_path"]),
            "launcher_stderr_path": str(smoke_paths["launcher_stderr_path"]),
            "bridge_stdout_path": str(bridge_stdout_path),
            "bridge_stderr_path": str(bridge_stderr_path),
            "installed_launcher_path": str(installed_paths["launcher_path"]),
            "windows_installed_launcher_path": windows_path_from_wsl(installed_paths["launcher_path"]),
            "launcher_exit_code": launcher_process.returncode,
            "bridge_exit_code": bridge_exit_code,
            "automation_port": automation_port,
            "bridge_host": host,
            "bridge_port": bridge_port,
            "bridge_url": expected_bridge_url,
            "bridge_command": expected_bridge_command,
            "initial_state": initial_state,
            "recovered_state": recovered_state,
            "quit_state": quit_state,
            "startup_diagnostics": diagnostics,
        }
    )

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print(f"Packaged Windows desktop bridge recovery check passed with diagnostics at {payload['diagnostics_path']}")
    print(payload["windows_diagnostics_path"])
    return 0


def run_gui_package_main_window_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    stage_root = Path(payload["stage_root"])
    smoke_paths = build_gui_smoke_paths(stage_root)
    timeout_seconds = float(getattr(namespace, "timeout_seconds", DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS))
    automation_port = int(getattr(namespace, "automation_port", 0)) or reserve_localhost_port()
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    bridge_port = int(getattr(namespace, "bridge_port", 0)) or reserve_socket_localhost_port()
    expected_bridge_url = bridge_url(host, bridge_port)
    expected_bridge_command = bridge_start_command(host, bridge_port)
    expected_transcript = "main window deterministic transcript"
    bridge_stdout_path = smoke_paths["smoke_dir"] / "bridge.stdout.log"
    bridge_stderr_path = smoke_paths["smoke_dir"] / "bridge.stderr.log"

    installed_paths = installed_windows_app_paths(str(payload["app_identifier"]), str(payload["app_channel"]))
    prepare_installed_windows_app_for_reinstall(installed_paths)
    clear_existing_gui_startup_logs(installed_paths)

    installer_completed = run_wsl_windows_executable_capture(
        Path(payload["installer_path"]),
        timeout_seconds=timeout_seconds,
    )
    smoke_paths["installer_stdout_path"].write_text(installer_completed.stdout, encoding="utf-8")
    smoke_paths["installer_stderr_path"].write_text(installer_completed.stderr, encoding="utf-8")
    if installer_completed.returncode != 0:
        raise DesktopAppError(
            f"Packaged Windows installer exited with code {installer_completed.returncode}. See {smoke_paths['installer_stdout_path']} and {smoke_paths['installer_stderr_path']}."
        )
    if not installed_paths["launcher_path"].exists():
        raise DesktopAppError(f"Installed Windows launcher not found at {installed_paths['launcher_path']}")

    bridge_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_ai_dictation.cli",
            "bridge",
            "--host",
            host,
            "--port",
            str(bridge_port),
        ],
        cwd=repo_root(),
        env={
            **os.environ,
            "LOCAL_AI_DICTATION_E2E_MODE": "1",
            "LOCAL_AI_DICTATION_E2E_TRANSCRIPT": expected_transcript,
            "LOCAL_AI_DICTATION_E2E_START_DELAY_MS": "30",
            "LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS": "40",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not wait_for_bridge(host, bridge_port, timeout_seconds=10.0):
        exit_code = bridge_process.poll()
        _stop_process(bridge_process)
        raise DesktopAppError(
            f"Local AI Dictation bridge did not become ready at {expected_bridge_url} within 10 seconds (exit code {exit_code})."
        )

    launcher_process = launch_wsl_windows_executable(
        installed_paths["launcher_path"],
        env={
            "LOCAL_AI_DICTATION_GUI_E2E": "1",
            "LOCAL_AI_DICTATION_GUI_E2E_PORT": str(automation_port),
            "LOCAL_AI_DICTATION_BRIDGE_URL": expected_bridge_url,
            "LOCAL_AI_DICTATION_BRIDGE_COMMAND": expected_bridge_command,
        },
    )
    initial_state: dict[str, Any] | None = None
    show_window_state: dict[str, Any] | None = None
    recording_state: dict[str, Any] | None = None
    completed_state: dict[str, Any] | None = None
    quit_state: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    try:
        wait_for_startup_readiness(installed_paths["diagnostics_path"], timeout_seconds=timeout_seconds)

        def _ready(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            if not isinstance(bridge_state, dict) or not isinstance(snapshot, dict) or not isinstance(session, dict):
                return False
            return (
                bridge_state.get("connected") is True
                and snapshot.get("bridgeUrl") == expected_bridge_url
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and snapshot.get("historyCount") == 0
                and session.get("state") in {"idle", "stopped"}
            )

        initial_state = wait_for_gui_e2e_state(
            automation_port,
            _ready,
            timeout_seconds=timeout_seconds,
        )
        show_window_state = invoke_gui_e2e_action(automation_port, "show-window")
        invoke_gui_e2e_action(automation_port, "window/toggle-recording")

        def _recording(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            if not isinstance(bridge_state, dict) or not isinstance(snapshot, dict) or not isinstance(session, dict):
                return False
            return (
                bridge_state.get("connected") is True
                and session.get("state") == "recording"
                and snapshot.get("toggleButtonText") == "Stop recording"
                and snapshot.get("historyCount") == 0
            )

        recording_state = wait_for_gui_e2e_state(
            automation_port,
            _recording,
            timeout_seconds=timeout_seconds,
        )
        invoke_gui_e2e_action(automation_port, "window/toggle-recording")

        def _completed(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            last_transcript = session.get("last_transcript") if isinstance(session, dict) else None
            history_texts = snapshot.get("historyTexts") if isinstance(snapshot, dict) else None
            if (
                not isinstance(bridge_state, dict)
                or not isinstance(snapshot, dict)
                or not isinstance(session, dict)
                or not isinstance(last_transcript, dict)
                or not isinstance(history_texts, list)
            ):
                return False
            return (
                bridge_state.get("connected") is True
                and session.get("state") == "idle"
                and last_transcript.get("transcript") == expected_transcript
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and snapshot.get("historyCount", 0) >= 1
                and expected_transcript in history_texts
            )

        completed_state = wait_for_gui_e2e_state(
            automation_port,
            _completed,
            timeout_seconds=timeout_seconds,
        )
        quit_state = invoke_gui_e2e_action(automation_port, "quit")
        launcher_process.wait(timeout=10)
        diagnostics = wait_for_shutdown_reason(installed_paths["diagnostics_path"], timeout_seconds=10.0)
    finally:
        _stop_process(bridge_process)
        if launcher_process.poll() is None:
            try:
                launcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_process.terminate()
                try:
                    launcher_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()
                    launcher_process.wait(timeout=5)

    launcher_stdout, launcher_stderr = launcher_process.communicate()
    smoke_paths["launcher_stdout_path"].write_text(launcher_stdout, encoding="utf-8")
    smoke_paths["launcher_stderr_path"].write_text(launcher_stderr, encoding="utf-8")

    bridge_stdout, bridge_stderr = bridge_process.communicate()
    bridge_stdout_path.write_text(bridge_stdout, encoding="utf-8")
    bridge_stderr_path.write_text(bridge_stderr, encoding="utf-8")

    if installed_paths["log_path"].exists():
        shutil.copy2(installed_paths["log_path"], smoke_paths["log_path"])
    if installed_paths["diagnostics_path"].exists():
        shutil.copy2(installed_paths["diagnostics_path"], smoke_paths["diagnostics_path"])

    payload.update(
        {
            "smoke_dir": str(smoke_paths["smoke_dir"]),
            "windows_smoke_dir": windows_path_from_wsl(smoke_paths["smoke_dir"]),
            "diagnostics_path": str(smoke_paths["diagnostics_path"]),
            "windows_diagnostics_path": windows_path_from_wsl(smoke_paths["diagnostics_path"]),
            "log_path": str(smoke_paths["log_path"]),
            "windows_log_path": windows_path_from_wsl(smoke_paths["log_path"]),
            "installer_stdout_path": str(smoke_paths["installer_stdout_path"]),
            "installer_stderr_path": str(smoke_paths["installer_stderr_path"]),
            "launcher_stdout_path": str(smoke_paths["launcher_stdout_path"]),
            "launcher_stderr_path": str(smoke_paths["launcher_stderr_path"]),
            "bridge_stdout_path": str(bridge_stdout_path),
            "bridge_stderr_path": str(bridge_stderr_path),
            "installed_launcher_path": str(installed_paths["launcher_path"]),
            "windows_installed_launcher_path": windows_path_from_wsl(installed_paths["launcher_path"]),
            "launcher_exit_code": launcher_process.returncode,
            "bridge_exit_code": bridge_process.returncode,
            "automation_port": automation_port,
            "bridge_host": host,
            "bridge_port": bridge_port,
            "bridge_url": expected_bridge_url,
            "bridge_command": expected_bridge_command,
            "expected_transcript": expected_transcript,
            "initial_state": initial_state,
            "show_window_state": show_window_state,
            "recording_state": recording_state,
            "completed_state": completed_state,
            "quit_state": quit_state,
            "startup_diagnostics": diagnostics,
        }
    )

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print(f"Packaged Windows main-window session check passed with diagnostics at {payload['diagnostics_path']}")
    print(payload["windows_diagnostics_path"])
    return 0


def run_gui_package_tray_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    stage_root = Path(payload["stage_root"])
    smoke_paths = build_gui_smoke_paths(stage_root)
    timeout_seconds = float(getattr(namespace, "timeout_seconds", DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS))
    automation_port = int(getattr(namespace, "automation_port", 0)) or reserve_localhost_port()
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    bridge_port = int(getattr(namespace, "bridge_port", 0)) or reserve_socket_localhost_port()
    expected_bridge_url = bridge_url(host, bridge_port)
    expected_bridge_command = bridge_start_command(host, bridge_port)
    expected_transcript = "tray deterministic transcript"
    bridge_stdout_path = smoke_paths["smoke_dir"] / "bridge.stdout.log"
    bridge_stderr_path = smoke_paths["smoke_dir"] / "bridge.stderr.log"

    installed_paths = installed_windows_app_paths(str(payload["app_identifier"]), str(payload["app_channel"]))
    prepare_installed_windows_app_for_reinstall(installed_paths)
    clear_existing_gui_startup_logs(installed_paths)

    installer_completed = run_wsl_windows_executable_capture(
        Path(payload["installer_path"]),
        timeout_seconds=timeout_seconds,
    )
    smoke_paths["installer_stdout_path"].write_text(installer_completed.stdout, encoding="utf-8")
    smoke_paths["installer_stderr_path"].write_text(installer_completed.stderr, encoding="utf-8")
    if installer_completed.returncode != 0:
        raise DesktopAppError(
            f"Packaged Windows installer exited with code {installer_completed.returncode}. See {smoke_paths['installer_stdout_path']} and {smoke_paths['installer_stderr_path']}."
        )
    if not installed_paths["launcher_path"].exists():
        raise DesktopAppError(f"Installed Windows launcher not found at {installed_paths['launcher_path']}")

    bridge_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_ai_dictation.cli",
            "bridge",
            "--host",
            host,
            "--port",
            str(bridge_port),
        ],
        cwd=repo_root(),
        env={
            **os.environ,
            "LOCAL_AI_DICTATION_E2E_MODE": "1",
            "LOCAL_AI_DICTATION_E2E_TRANSCRIPT": expected_transcript,
            "LOCAL_AI_DICTATION_E2E_START_DELAY_MS": "30",
            "LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS": "40",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not wait_for_bridge(host, bridge_port, timeout_seconds=10.0):
        exit_code = bridge_process.poll()
        _stop_process(bridge_process)
        raise DesktopAppError(
            f"Local AI Dictation bridge did not become ready at {expected_bridge_url} within 10 seconds (exit code {exit_code})."
        )

    launcher_process = launch_wsl_windows_executable(
        installed_paths["launcher_path"],
        env={
            "LOCAL_AI_DICTATION_GUI_E2E": "1",
            "LOCAL_AI_DICTATION_GUI_E2E_PORT": str(automation_port),
            "LOCAL_AI_DICTATION_BRIDGE_URL": expected_bridge_url,
            "LOCAL_AI_DICTATION_BRIDGE_COMMAND": expected_bridge_command,
        },
    )
    initial_state: dict[str, Any] | None = None
    tray_open_state: dict[str, Any] | None = None
    recording_state: dict[str, Any] | None = None
    completed_state: dict[str, Any] | None = None
    tray_quit_state: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    try:
        wait_for_startup_readiness(installed_paths["diagnostics_path"], timeout_seconds=timeout_seconds)

        def _ready(state: dict[str, Any]) -> bool:
            tray_state = state.get("tray")
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            actions = tray_state.get("actions") if isinstance(tray_state, dict) else None
            if (
                not isinstance(tray_state, dict)
                or not isinstance(actions, list)
                or not isinstance(bridge_state, dict)
                or not isinstance(snapshot, dict)
                or not isinstance(session, dict)
            ):
                return False
            return (
                tray_state.get("created") is True
                and set(actions) == {"open", "toggle", "quit"}
                and bridge_state.get("connected") is True
                and snapshot.get("bridgeUrl") == expected_bridge_url
                and snapshot.get("bridgeCommand") == expected_bridge_command
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and snapshot.get("historyCount") == 0
                and session.get("state") in {"idle", "stopped"}
            )

        initial_state = wait_for_gui_e2e_state(
            automation_port,
            _ready,
            timeout_seconds=timeout_seconds,
        )
        tray_open_state = invoke_gui_e2e_action(automation_port, "tray/open")
        invoke_gui_e2e_action(automation_port, "tray/toggle")

        def _recording(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            if not isinstance(bridge_state, dict) or not isinstance(snapshot, dict) or not isinstance(session, dict):
                return False
            return (
                bridge_state.get("connected") is True
                and session.get("state") == "recording"
                and snapshot.get("toggleButtonText") == "Stop recording"
                and snapshot.get("historyCount") == 0
            )

        recording_state = wait_for_gui_e2e_state(
            automation_port,
            _recording,
            timeout_seconds=timeout_seconds,
        )
        invoke_gui_e2e_action(automation_port, "tray/toggle")

        def _completed(state: dict[str, Any]) -> bool:
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            last_transcript = session.get("last_transcript") if isinstance(session, dict) else None
            history_texts = snapshot.get("historyTexts") if isinstance(snapshot, dict) else None
            if (
                not isinstance(bridge_state, dict)
                or not isinstance(snapshot, dict)
                or not isinstance(session, dict)
                or not isinstance(last_transcript, dict)
                or not isinstance(history_texts, list)
            ):
                return False
            return (
                bridge_state.get("connected") is True
                and session.get("state") == "idle"
                and last_transcript.get("transcript") == expected_transcript
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and snapshot.get("historyCount", 0) >= 1
                and expected_transcript in history_texts
            )

        completed_state = wait_for_gui_e2e_state(
            automation_port,
            _completed,
            timeout_seconds=timeout_seconds,
        )
        tray_quit_state = invoke_gui_e2e_action(automation_port, "tray/quit")
        launcher_process.wait(timeout=10)
        diagnostics = wait_for_shutdown_reason(installed_paths["diagnostics_path"], timeout_seconds=10.0)
    finally:
        _stop_process(bridge_process)
        if launcher_process.poll() is None:
            try:
                launcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_process.terminate()
                try:
                    launcher_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()
                    launcher_process.wait(timeout=5)

    launcher_stdout, launcher_stderr = launcher_process.communicate()
    smoke_paths["launcher_stdout_path"].write_text(launcher_stdout, encoding="utf-8")
    smoke_paths["launcher_stderr_path"].write_text(launcher_stderr, encoding="utf-8")

    bridge_stdout, bridge_stderr = bridge_process.communicate()
    bridge_stdout_path.write_text(bridge_stdout, encoding="utf-8")
    bridge_stderr_path.write_text(bridge_stderr, encoding="utf-8")

    if installed_paths["log_path"].exists():
        shutil.copy2(installed_paths["log_path"], smoke_paths["log_path"])
    if installed_paths["diagnostics_path"].exists():
        shutil.copy2(installed_paths["diagnostics_path"], smoke_paths["diagnostics_path"])

    payload.update(
        {
            "smoke_dir": str(smoke_paths["smoke_dir"]),
            "windows_smoke_dir": windows_path_from_wsl(smoke_paths["smoke_dir"]),
            "diagnostics_path": str(smoke_paths["diagnostics_path"]),
            "windows_diagnostics_path": windows_path_from_wsl(smoke_paths["diagnostics_path"]),
            "log_path": str(smoke_paths["log_path"]),
            "windows_log_path": windows_path_from_wsl(smoke_paths["log_path"]),
            "installer_stdout_path": str(smoke_paths["installer_stdout_path"]),
            "installer_stderr_path": str(smoke_paths["installer_stderr_path"]),
            "launcher_stdout_path": str(smoke_paths["launcher_stdout_path"]),
            "launcher_stderr_path": str(smoke_paths["launcher_stderr_path"]),
            "bridge_stdout_path": str(bridge_stdout_path),
            "bridge_stderr_path": str(bridge_stderr_path),
            "installed_launcher_path": str(installed_paths["launcher_path"]),
            "windows_installed_launcher_path": windows_path_from_wsl(installed_paths["launcher_path"]),
            "launcher_exit_code": launcher_process.returncode,
            "bridge_exit_code": bridge_process.returncode,
            "automation_port": automation_port,
            "bridge_host": host,
            "bridge_port": bridge_port,
            "bridge_url": expected_bridge_url,
            "bridge_command": expected_bridge_command,
            "expected_transcript": expected_transcript,
            "initial_state": initial_state,
            "tray_open_state": tray_open_state,
            "recording_state": recording_state,
            "completed_state": completed_state,
            "tray_quit_state": tray_quit_state,
            "startup_diagnostics": diagnostics,
        }
    )

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print(f"Packaged Windows tray session check passed with diagnostics at {payload['diagnostics_path']}")
    print(payload["windows_diagnostics_path"])
    return 0


def run_gui_package_hotkey_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    stage_root = Path(payload["stage_root"])
    smoke_paths = build_gui_smoke_paths(stage_root)
    timeout_seconds = float(getattr(namespace, "timeout_seconds", DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS))
    automation_port = int(getattr(namespace, "automation_port", 0)) or reserve_localhost_port()
    host = str(getattr(namespace, "host", DEFAULT_BRIDGE_HOST))
    bridge_port = int(getattr(namespace, "bridge_port", 0)) or reserve_socket_localhost_port()
    expected_bridge_url = bridge_url(host, bridge_port)
    expected_bridge_command = bridge_start_command(host, bridge_port)
    expected_hotkey = os.environ.get("LOCAL_AI_DICTATION_HOTKEY", "CommandOrControl+Alt+R")
    expected_transcript = "hotkey deterministic transcript"
    bridge_stdout_path = smoke_paths["smoke_dir"] / "bridge.stdout.log"
    bridge_stderr_path = smoke_paths["smoke_dir"] / "bridge.stderr.log"

    installed_paths = installed_windows_app_paths(str(payload["app_identifier"]), str(payload["app_channel"]))
    prepare_installed_windows_app_for_reinstall(installed_paths)
    clear_existing_gui_startup_logs(installed_paths)

    installer_completed = run_wsl_windows_executable_capture(
        Path(payload["installer_path"]),
        timeout_seconds=timeout_seconds,
    )
    smoke_paths["installer_stdout_path"].write_text(installer_completed.stdout, encoding="utf-8")
    smoke_paths["installer_stderr_path"].write_text(installer_completed.stderr, encoding="utf-8")
    if installer_completed.returncode != 0:
        raise DesktopAppError(
            f"Packaged Windows installer exited with code {installer_completed.returncode}. See {smoke_paths['installer_stdout_path']} and {smoke_paths['installer_stderr_path']}."
        )
    if not installed_paths["launcher_path"].exists():
        raise DesktopAppError(f"Installed Windows launcher not found at {installed_paths['launcher_path']}")

    bridge_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_ai_dictation.cli",
            "bridge",
            "--host",
            host,
            "--port",
            str(bridge_port),
        ],
        cwd=repo_root(),
        env={
            **os.environ,
            "LOCAL_AI_DICTATION_E2E_MODE": "1",
            "LOCAL_AI_DICTATION_E2E_TRANSCRIPT": expected_transcript,
            "LOCAL_AI_DICTATION_E2E_START_DELAY_MS": "30",
            "LOCAL_AI_DICTATION_E2E_STOP_DELAY_MS": "40",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not wait_for_bridge(host, bridge_port, timeout_seconds=10.0):
        exit_code = bridge_process.poll()
        _stop_process(bridge_process)
        raise DesktopAppError(
            f"Local AI Dictation bridge did not become ready at {expected_bridge_url} within 10 seconds (exit code {exit_code})."
        )

    launcher_process = launch_wsl_windows_executable(
        installed_paths["launcher_path"],
        env={
            "LOCAL_AI_DICTATION_GUI_E2E": "1",
            "LOCAL_AI_DICTATION_GUI_E2E_PORT": str(automation_port),
            "LOCAL_AI_DICTATION_BRIDGE_URL": expected_bridge_url,
            "LOCAL_AI_DICTATION_BRIDGE_COMMAND": expected_bridge_command,
            "LOCAL_AI_DICTATION_HOTKEY": expected_hotkey,
        },
    )
    initial_state: dict[str, Any] | None = None
    recording_state: dict[str, Any] | None = None
    completed_state: dict[str, Any] | None = None
    quit_state: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    try:
        wait_for_startup_readiness(installed_paths["diagnostics_path"], timeout_seconds=timeout_seconds)

        def _ready(state: dict[str, Any]) -> bool:
            hotkey_state = state.get("hotkey")
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            if (
                not isinstance(hotkey_state, dict)
                or not isinstance(bridge_state, dict)
                or not isinstance(snapshot, dict)
                or not isinstance(session, dict)
            ):
                return False
            return (
                hotkey_state.get("registered") is True
                and hotkey_state.get("accelerator") == expected_hotkey
                and bridge_state.get("connected") is True
                and snapshot.get("bridgeUrl") == expected_bridge_url
                and snapshot.get("bridgeCommand") == expected_bridge_command
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and snapshot.get("historyCount") == 0
                and session.get("state") in {"idle", "stopped"}
            )

        initial_state = wait_for_gui_e2e_state(
            automation_port,
            _ready,
            timeout_seconds=timeout_seconds,
        )
        invoke_gui_e2e_action(automation_port, "hotkey/trigger")

        def _recording(state: dict[str, Any]) -> bool:
            hotkey_state = state.get("hotkey")
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer")
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            if (
                not isinstance(hotkey_state, dict)
                or not isinstance(bridge_state, dict)
                or not isinstance(snapshot, dict)
                or not isinstance(session, dict)
            ):
                return False
            return (
                hotkey_state.get("registered") is True
                and bridge_state.get("connected") is True
                and session.get("state") == "recording"
                and snapshot.get("toggleButtonText") == "Stop recording"
                and snapshot.get("historyCount") == 0
            )

        recording_state = wait_for_gui_e2e_state(
            automation_port,
            _recording,
            timeout_seconds=timeout_seconds,
        )
        invoke_gui_e2e_action(automation_port, "hotkey/trigger")

        def _completed(state: dict[str, Any]) -> bool:
            hotkey_state = state.get("hotkey")
            bridge_state = state.get("bridge")
            renderer_state = state.get("renderer") if isinstance(state.get("renderer"), dict) else None
            snapshot = renderer_state.get("snapshot") if isinstance(renderer_state, dict) else None
            session = bridge_state.get("session") if isinstance(bridge_state, dict) else None
            last_transcript = session.get("last_transcript") if isinstance(session, dict) else None
            history_texts = snapshot.get("historyTexts") if isinstance(snapshot, dict) else None
            if (
                not isinstance(hotkey_state, dict)
                or not isinstance(bridge_state, dict)
                or not isinstance(snapshot, dict)
                or not isinstance(session, dict)
                or not isinstance(last_transcript, dict)
                or not isinstance(history_texts, list)
            ):
                return False
            return (
                hotkey_state.get("registered") is True
                and bridge_state.get("connected") is True
                and session.get("state") == "idle"
                and last_transcript.get("transcript") == expected_transcript
                and snapshot.get("toggleButtonText") in {"Start recording", "Load + record"}
                and snapshot.get("historyCount", 0) >= 1
                and expected_transcript in history_texts
            )

        completed_state = wait_for_gui_e2e_state(
            automation_port,
            _completed,
            timeout_seconds=timeout_seconds,
        )
        quit_state = invoke_gui_e2e_action(automation_port, "quit")
        launcher_process.wait(timeout=10)
        diagnostics = wait_for_shutdown_reason(installed_paths["diagnostics_path"], timeout_seconds=10.0)
    finally:
        _stop_process(bridge_process)
        if launcher_process.poll() is None:
            try:
                launcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_process.terminate()
                try:
                    launcher_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()
                    launcher_process.wait(timeout=5)

    launcher_stdout, launcher_stderr = launcher_process.communicate()
    smoke_paths["launcher_stdout_path"].write_text(launcher_stdout, encoding="utf-8")
    smoke_paths["launcher_stderr_path"].write_text(launcher_stderr, encoding="utf-8")

    bridge_stdout, bridge_stderr = bridge_process.communicate()
    bridge_stdout_path.write_text(bridge_stdout, encoding="utf-8")
    bridge_stderr_path.write_text(bridge_stderr, encoding="utf-8")

    if installed_paths["log_path"].exists():
        shutil.copy2(installed_paths["log_path"], smoke_paths["log_path"])
    if installed_paths["diagnostics_path"].exists():
        shutil.copy2(installed_paths["diagnostics_path"], smoke_paths["diagnostics_path"])

    payload.update(
        {
            "smoke_dir": str(smoke_paths["smoke_dir"]),
            "windows_smoke_dir": windows_path_from_wsl(smoke_paths["smoke_dir"]),
            "diagnostics_path": str(smoke_paths["diagnostics_path"]),
            "windows_diagnostics_path": windows_path_from_wsl(smoke_paths["diagnostics_path"]),
            "log_path": str(smoke_paths["log_path"]),
            "windows_log_path": windows_path_from_wsl(smoke_paths["log_path"]),
            "installer_stdout_path": str(smoke_paths["installer_stdout_path"]),
            "installer_stderr_path": str(smoke_paths["installer_stderr_path"]),
            "launcher_stdout_path": str(smoke_paths["launcher_stdout_path"]),
            "launcher_stderr_path": str(smoke_paths["launcher_stderr_path"]),
            "bridge_stdout_path": str(bridge_stdout_path),
            "bridge_stderr_path": str(bridge_stderr_path),
            "installed_launcher_path": str(installed_paths["launcher_path"]),
            "windows_installed_launcher_path": windows_path_from_wsl(installed_paths["launcher_path"]),
            "launcher_exit_code": launcher_process.returncode,
            "bridge_exit_code": bridge_process.returncode,
            "automation_port": automation_port,
            "bridge_host": host,
            "bridge_port": bridge_port,
            "bridge_url": expected_bridge_url,
            "bridge_command": expected_bridge_command,
            "expected_hotkey": expected_hotkey,
            "expected_transcript": expected_transcript,
            "initial_state": initial_state,
            "recording_state": recording_state,
            "completed_state": completed_state,
            "quit_state": quit_state,
            "startup_diagnostics": diagnostics,
        }
    )

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print(f"Packaged Windows hotkey session check passed with diagnostics at {payload['diagnostics_path']}")
    print(payload["windows_diagnostics_path"])
    return 0


def run_gui_package_smoke_command(namespace: Any) -> int:
    payload = build_windows_package_payload()
    stage_root = Path(payload["stage_root"])
    smoke_paths = build_gui_smoke_paths(stage_root)
    timeout_seconds = float(getattr(namespace, "timeout_seconds", DEFAULT_GUI_SMOKE_TIMEOUT_SECONDS))
    auto_exit_ms = int(getattr(namespace, "auto_exit_ms", DEFAULT_GUI_AUTO_EXIT_MS))

    installed_paths = installed_windows_app_paths(str(payload["app_identifier"]), str(payload["app_channel"]))
    prepare_installed_windows_app_for_reinstall(installed_paths)
    clear_existing_gui_startup_logs(installed_paths)

    installer_completed = run_wsl_windows_executable_capture(
        Path(payload["installer_path"]),
        timeout_seconds=timeout_seconds,
    )
    smoke_paths["installer_stdout_path"].write_text(installer_completed.stdout, encoding="utf-8")
    smoke_paths["installer_stderr_path"].write_text(installer_completed.stderr, encoding="utf-8")
    if installer_completed.returncode != 0:
        raise DesktopAppError(
            f"Packaged Windows installer exited with code {installer_completed.returncode}. See {smoke_paths['installer_stdout_path']} and {smoke_paths['installer_stderr_path']}."
        )
    if not installed_paths["launcher_path"].exists():
        raise DesktopAppError(f"Installed Windows launcher not found at {installed_paths['launcher_path']}")

    launcher_process = launch_wsl_windows_executable(
        installed_paths["launcher_path"],
        env={
            "LOCAL_AI_DICTATION_GUI_E2E": "1",
            "LOCAL_AI_DICTATION_GUI_AUTO_EXIT_MS": str(auto_exit_ms),
        },
    )
    try:
        wait_for_startup_readiness(installed_paths["diagnostics_path"], timeout_seconds=timeout_seconds)
        request_installed_gui_shutdown(installed_paths)
        diagnostics = wait_for_shutdown_reason(installed_paths["diagnostics_path"], timeout_seconds=10.0)
    finally:
        if launcher_process.poll() is None:
            try:
                launcher_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher_process.terminate()
                try:
                    launcher_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()
                    launcher_process.wait(timeout=5)

    launcher_stdout, launcher_stderr = launcher_process.communicate()
    smoke_paths["launcher_stdout_path"].write_text(launcher_stdout, encoding="utf-8")
    smoke_paths["launcher_stderr_path"].write_text(launcher_stderr, encoding="utf-8")

    if installed_paths["log_path"].exists():
        shutil.copy2(installed_paths["log_path"], smoke_paths["log_path"])
    if installed_paths["diagnostics_path"].exists():
        shutil.copy2(installed_paths["diagnostics_path"], smoke_paths["diagnostics_path"])

    payload.update(
        {
            "smoke_dir": str(smoke_paths["smoke_dir"]),
            "windows_smoke_dir": windows_path_from_wsl(smoke_paths["smoke_dir"]),
            "diagnostics_path": str(smoke_paths["diagnostics_path"]),
            "windows_diagnostics_path": windows_path_from_wsl(smoke_paths["diagnostics_path"]),
            "log_path": str(smoke_paths["log_path"]),
            "windows_log_path": windows_path_from_wsl(smoke_paths["log_path"]),
            "installer_stdout_path": str(smoke_paths["installer_stdout_path"]),
            "installer_stderr_path": str(smoke_paths["installer_stderr_path"]),
            "launcher_stdout_path": str(smoke_paths["launcher_stdout_path"]),
            "launcher_stderr_path": str(smoke_paths["launcher_stderr_path"]),
            "installed_launcher_path": str(installed_paths["launcher_path"]),
            "windows_installed_launcher_path": windows_path_from_wsl(installed_paths["launcher_path"]),
            "launcher_exit_code": launcher_process.returncode,
            "startup_diagnostics": diagnostics,
        }
    )

    if bool(getattr(namespace, "json_output", False)):
        print(json.dumps(payload))
        return 0

    print(f"Packaged Windows desktop app smoke check passed with diagnostics at {payload['diagnostics_path']}")
    print(payload["windows_diagnostics_path"])
    return 0


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


