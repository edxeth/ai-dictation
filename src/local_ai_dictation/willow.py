"""Willow Voice cloud transcription adapter.

The wire protocol is derived from the unofficial willow-arch Linux client. Willow
selects Frontier Mini or Frontier Pro server-side according to the account plan.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
import wave
import webbrowser

import msgpack
from websockets.exceptions import ConnectionClosedOK
from websockets.sync.client import connect

from local_ai_dictation.errors import MODEL_IMPORT_FAILED, MODEL_TRANSCRIBE_FAILED, ModelError
from local_ai_dictation.types import DictationConfig, TranscriptionResult


PRODUCTION_TRANSCRIBE_URL = "wss://middleware.willowvoice.com/transcribe"
SUPABASE_TOKEN_URL = "https://db.willowvoice.com/auth/v1/token?grant_type=refresh_token"
SUPABASE_PKCE_TOKEN_URL = "https://db.willowvoice.com/auth/v1/token?grant_type=pkce"
SUPABASE_AUTHORIZE_URL = "https://db.willowvoice.com/auth/v1/authorize"
SUPABASE_PUBLIC_KEY = "sb_publishable_zK_SHENTsKCbRZP0-1mkeA_HUlYW4wM"
WILLOW_OAUTH_REDIRECT_URL = "https://willowvoice.com/success-open-app"
DEFAULT_SESSION_PATH = Path.home() / ".local" / "state" / "local-ai-dictation" / "willow-session.json"
PCM_PACKET_BYTES = 1024
AUDIO_QUEUE_MAX_CHUNKS = 512
AUDIO_QUEUE_PUT_TIMEOUT = 1.0
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_RESULT_KEYS = {"dictation_result", "result_actions"}
_TEXT_ACTION_TYPES = {"paste", "save"}


@dataclass(frozen=True)
class WillowSession:
    access_token: str
    user_id: str | None = None
    refresh_token: str | None = None
    expires_at: float | None = None


def _state_home(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    state_home = source.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home)
    home = source.get("HOME")
    if home:
        return Path(home) / ".local" / "state"
    return DEFAULT_SESSION_PATH.parents[1]


def session_path(env: Mapping[str, str] | None = None) -> Path:
    return _state_home(env) / "local-ai-dictation" / "willow-session.json"


def legacy_session_path(env: Mapping[str, str] | None = None) -> Path:
    return _state_home(env) / "willow-linux" / "supabase-session.json"


def resolve_session_path(env: Mapping[str, str] | None = None) -> Path:
    owned_path = session_path(env)
    if owned_path.is_file() and not owned_path.is_symlink():
        return owned_path
    legacy_path = legacy_session_path(env)
    if legacy_path.is_file() and not legacy_path.is_symlink():
        return legacy_path
    return owned_path


def _find_session(value: Any) -> WillowSession | None:
    if isinstance(value, str):
        try:
            return _find_session(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(value, list):
        for item in value:
            found = _find_session(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None

    access_token = value.get("access_token")
    if isinstance(access_token, str) and access_token:
        user = value.get("user")
        user_id = user.get("id") if isinstance(user, dict) else value.get("user_id")
        refresh_token = value.get("refresh_token")
        expires_at = value.get("expires_at")
        return WillowSession(
            access_token=access_token,
            user_id=user_id if isinstance(user_id, str) and user_id else None,
            refresh_token=refresh_token if isinstance(refresh_token, str) and refresh_token else None,
            expires_at=float(expires_at) if isinstance(expires_at, (int, float)) else None,
        )
    for item in value.values():
        found = _find_session(item)
        if found is not None:
            return found
    return None


def _write_owned_session(session: WillowSession, path: Path) -> None:
    if not session.refresh_token:
        raise ModelError(MODEL_IMPORT_FAILED, "Willow session does not contain a refresh token")
    payload = {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "expires_at": session.expires_at,
        "user_id": session.user_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        os.chmod(temporary_path, 0o600)
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    os.chmod(path, 0o600)


def import_session_file(
    source_path: Path,
    *,
    destination: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    session = load_session(path=source_path, env=env)
    destination_path = session_path(env) if destination is None else destination
    _write_owned_session(session, destination_path)
    return destination_path


def _google_oauth_url(code_challenge: str) -> str:
    query = urlencode(
        {
            "provider": "google",
            "redirect_to": WILLOW_OAUTH_REDIRECT_URL,
            "code_challenge": code_challenge,
            "code_challenge_method": "s256",
        }
    )
    return f"{SUPABASE_AUTHORIZE_URL}?{query}"


def _oauth_callback_code(callback_url: str) -> str:
    parsed = urlparse(callback_url.strip())
    is_web_callback = (
        parsed.scheme == "https"
        and parsed.netloc == "willowvoice.com"
        and parsed.path == "/success-open-app"
    )
    is_app_callback = parsed.scheme == "willow" and parsed.netloc == "login-callback"
    if not (is_web_callback or is_app_callback):
        raise ModelError(
            MODEL_IMPORT_FAILED,
            "Willow login callback must be the success URL from willowvoice.com",
        )
    query = parse_qs(parsed.query)
    error = query.get("error_description", query.get("error", [""]))[0]
    if error:
        raise ModelError(MODEL_IMPORT_FAILED, f"Willow login failed: {error}")
    code = query.get("code", [""])[0].strip()
    if not code:
        raise ModelError(MODEL_IMPORT_FAILED, "Willow login callback does not contain an authorization code")
    return code


def _session_from_token_payload(payload: Any, missing_message: str) -> WillowSession:
    session = _find_session(payload)
    if session is None:
        raise ModelError(MODEL_IMPORT_FAILED, missing_message)
    if session.expires_at is None and isinstance(payload, dict) and isinstance(payload.get("expires_in"), (int, float)):
        session = WillowSession(
            access_token=session.access_token,
            user_id=session.user_id,
            refresh_token=session.refresh_token,
            expires_at=time.time() + float(payload["expires_in"]),
        )
    return session


def _http_error_message(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read(4096).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return f"HTTP {exc.code}: {exc.reason}"
    if isinstance(payload, dict):
        for key in ("msg", "message", "error_description", "error"):
            detail = payload.get(key)
            if isinstance(detail, str) and detail.strip():
                return f"HTTP {exc.code}: {detail.strip()}"
    return f"HTTP {exc.code}: {exc.reason}"


def _exchange_oauth_code(code: str, code_verifier: str) -> WillowSession:
    request = Request(
        SUPABASE_PKCE_TOKEN_URL,
        data=json.dumps({"auth_code": code, "code_verifier": code_verifier}).encode("utf-8"),
        method="POST",
        headers={
            "apikey": SUPABASE_PUBLIC_KEY,
            "Authorization": f"Bearer {SUPABASE_PUBLIC_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=8.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ModelError(MODEL_IMPORT_FAILED, f"Could not complete Willow Google login: {_http_error_message(exc)}") from exc
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, f"Could not complete Willow Google login: {exc}") from exc
    session = _session_from_token_payload(payload, "Willow Google login returned no session")
    if not session.refresh_token:
        raise ModelError(MODEL_IMPORT_FAILED, "Willow Google login returned no refreshable session")
    return session


def login_google_session(*, env: Mapping[str, str] | None = None) -> Path:
    code_verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    authorize_url = _google_oauth_url(code_challenge)

    opened = webbrowser.open(authorize_url)
    if not opened:
        print(f"Open this Willow login URL in your browser:\n{authorize_url}", file=sys.stderr)
    print(
        "After Google login, copy the complete https://willowvoice.com/success-open-app?code=... URL "
        "from the browser address bar and paste it below.",
        file=sys.stderr,
    )
    try:
        callback_url = getpass.getpass("").strip()
    except EOFError as exc:
        raise ModelError(MODEL_IMPORT_FAILED, "Willow login callback was not provided") from exc
    session = _exchange_oauth_code(_oauth_callback_code(callback_url), code_verifier)
    destination = session_path(env)
    _write_owned_session(session, destination)
    return destination


def load_session(
    *,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> WillowSession:
    source = os.environ if env is None else env
    explicit_token = source.get("WILLOW_ACCESS_TOKEN", "").strip()
    if explicit_token:
        user_id = source.get("WILLOW_USER_ID", "").strip() or None
        return WillowSession(explicit_token, user_id)

    resolved_path = resolve_session_path(source) if path is None else path
    try:
        if resolved_path.is_symlink() or not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelError(
            MODEL_IMPORT_FAILED,
            f"Willow session required. Run `local-ai-dictation willow-session login` or import a session. Session not found at {resolved_path}",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(MODEL_IMPORT_FAILED, f"Could not read Willow session at {resolved_path}: {exc}") from exc

    found = _find_session(payload)
    if found is None:
        raise ModelError(MODEL_IMPORT_FAILED, f"Willow session at {resolved_path} does not contain an access token")
    if path is None and resolved_path == legacy_session_path(source):
        _write_owned_session(found, session_path(source))
    return found


def _refresh_session(session: WillowSession) -> WillowSession:
    if not session.refresh_token:
        raise ModelError(MODEL_IMPORT_FAILED, "Willow session expired. Run `local-ai-dictation willow-session login` or import a fresh session.")
    request = Request(
        SUPABASE_TOKEN_URL,
        data=json.dumps({"refresh_token": session.refresh_token}).encode("utf-8"),
        method="POST",
        headers={
            "apikey": SUPABASE_PUBLIC_KEY,
            "Authorization": f"Bearer {SUPABASE_PUBLIC_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=8.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ModelError(MODEL_IMPORT_FAILED, f"Could not refresh Willow login: {_http_error_message(exc)}") from exc
    except Exception as exc:
        raise ModelError(MODEL_IMPORT_FAILED, f"Could not refresh Willow login: {exc}") from exc
    return _session_from_token_payload(payload, "Willow login refresh returned no session")


def _replace_stored_session(value: Any, old_access_token: str, refreshed: WillowSession) -> tuple[Any, bool]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value, False
        updated, changed = _replace_stored_session(decoded, old_access_token, refreshed)
        return (json.dumps(updated, separators=(",", ":")), True) if changed else (value, False)
    if isinstance(value, list):
        changed = False
        updated_items = []
        for item in value:
            updated, item_changed = _replace_stored_session(item, old_access_token, refreshed)
            updated_items.append(updated)
            changed = changed or item_changed
        return updated_items, changed
    if not isinstance(value, dict):
        return value, False
    if value.get("access_token") == old_access_token:
        updated = dict(value)
        updated["access_token"] = refreshed.access_token
        if refreshed.refresh_token:
            updated["refresh_token"] = refreshed.refresh_token
        if refreshed.expires_at is not None:
            updated["expires_at"] = refreshed.expires_at
        if refreshed.user_id and isinstance(updated.get("user"), dict):
            updated["user"] = {**updated["user"], "id": refreshed.user_id}
        return updated, True
    changed = False
    updated_mapping = {}
    for key, item in value.items():
        updated, item_changed = _replace_stored_session(item, old_access_token, refreshed)
        updated_mapping[key] = updated
        changed = changed or item_changed
    return updated_mapping, changed


def _persist_refreshed_session(path: Path, old_access_token: str, refreshed: WillowSession) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated, changed = _replace_stored_session(payload, old_access_token, refreshed)
    if not changed:
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        os.chmod(temporary_path, 0o600)
        json.dump(updated, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    os.chmod(path, 0o600)


class WillowProtocol:
    def __init__(self, maximum_message_bytes: int = MAX_MESSAGE_BYTES) -> None:
        if maximum_message_bytes < 256:
            raise ValueError("Message limit is too small")
        self._maximum_message_bytes = maximum_message_bytes
        self._next_sequence = 0
        self._assemblies: dict[str, dict[str, Any]] = {}
        self._buffered_bytes = 0

    def encode(self, key: str, data: Any) -> list[bytes]:
        sequence = self._next_sequence
        self._next_sequence += 1
        packed = msgpack.packb({"seq": sequence, "d": {"key": key, "data": data}}, use_bin_type=True)
        if len(packed) <= self._maximum_message_bytes:
            return [packed]

        payload_size = self._maximum_message_bytes - 128
        chunks = [packed[offset : offset + payload_size] for offset in range(0, len(packed), payload_size)]
        chunk_id = str(uuid4())
        return [
            msgpack.packb(
                {"id": chunk_id, "seq": sequence, "idx": index, "tot": len(chunks), "bin": chunk},
                use_bin_type=True,
            )
            for index, chunk in enumerate(chunks)
        ]

    def accept(self, wire: bytes | bytearray | memoryview) -> dict[str, Any] | None:
        if len(wire) > MAX_MESSAGE_BYTES:
            raise ValueError("Protocol message exceeds safe byte limit")
        decoded = msgpack.unpackb(bytes(wire), raw=False)
        if isinstance(decoded, dict) and isinstance(decoded.get("d"), dict):
            return decoded
        if not isinstance(decoded, dict):
            raise ValueError("Unknown protocol envelope")

        chunk_id = decoded.get("id")
        sequence = decoded.get("seq")
        index = decoded.get("idx")
        total = decoded.get("tot")
        part = decoded.get("bin")
        if not (
            isinstance(chunk_id, str)
            and isinstance(sequence, int)
            and isinstance(index, int)
            and isinstance(total, int)
            and 0 <= index < total <= 1024
            and isinstance(part, bytes)
        ):
            raise ValueError("Invalid chunk metadata")

        if chunk_id not in self._assemblies and len(self._assemblies) >= 16:
            raise ValueError("Too many incomplete chunked messages")
        assembly = self._assemblies.setdefault(
            chunk_id,
            {"seq": sequence, "total": total, "parts": {}, "bytes": 0, "created": time.monotonic()},
        )
        if assembly["seq"] != sequence or assembly["total"] != total or index in assembly["parts"]:
            removed = self._assemblies.pop(chunk_id, None)
            if removed is not None:
                self._buffered_bytes -= removed["bytes"]
            raise ValueError("Inconsistent or duplicate protocol chunk")
        if assembly["bytes"] + len(part) > MAX_MESSAGE_BYTES or self._buffered_bytes + len(part) > MAX_MESSAGE_BYTES:
            removed = self._assemblies.pop(chunk_id, None)
            if removed is not None:
                self._buffered_bytes -= removed["bytes"]
            raise ValueError("Chunked message exceeds safe byte limit")
        assembly["parts"][index] = part
        assembly["bytes"] += len(part)
        self._buffered_bytes += len(part)
        if len(assembly["parts"]) != total:
            return None

        packed = b"".join(assembly["parts"][part_index] for part_index in range(total))
        self._assemblies.pop(chunk_id, None)
        self._buffered_bytes -= assembly["bytes"]
        envelope = msgpack.unpackb(packed, raw=False)
        if not isinstance(envelope, dict) or envelope.get("seq") != sequence or not isinstance(envelope.get("d"), dict):
            raise ValueError("Chunk payload is not a matching logical envelope")
        return envelope


def _extract_transcript(envelope: Mapping[str, Any]) -> str | None:
    descriptor = envelope.get("d")
    if not isinstance(descriptor, dict) or descriptor.get("key") not in _RESULT_KEYS:
        return None
    data = descriptor.get("data")
    if isinstance(data, list):
        actions = data
    elif isinstance(data, dict) and isinstance(data.get("actions"), list):
        actions = data["actions"]
    elif isinstance(data, dict) and isinstance(data.get("result"), dict) and isinstance(data["result"].get("actions"), list):
        actions = data["result"]["actions"]
    else:
        actions = []
    texts = [
        action["text"]
        for action in actions
        if isinstance(action, dict)
        and action.get("type") in _TEXT_ACTION_TYPES
        and isinstance(action.get("text"), str)
    ]
    return "".join(texts) if texts else None


_STREAM_FINISH = object()


class WillowStreamingSession:
    """Streams realtime PCM while capture is still in progress."""

    def __init__(self, engine: "WillowEngine") -> None:
        self._engine = engine
        self._audio_queue: queue.Queue[object] = queue.Queue(maxsize=AUDIO_QUEUE_MAX_CHUNKS)
        self._cancelled = threading.Event()
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._state_lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._socket: Any | None = None
        self._finish_requested = False
        self._transcript: str | None = None
        self._error: ModelError | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_ready(self) -> None:
        deadline = time.monotonic() + self._engine._connect_timeout + 1.0
        while not self._ready.is_set():
            if self._cancelled.is_set():
                raise ModelError(MODEL_TRANSCRIBE_FAILED, "Willow streaming session was cancelled")
            if self._finished.is_set():
                self._raise_if_failed()
                raise ModelError(MODEL_TRANSCRIBE_FAILED, "Willow streaming session ended before it was ready")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.cancel()
                raise ModelError(MODEL_TRANSCRIBE_FAILED, "Timed out preparing Willow streaming session")
            self._ready.wait(timeout=min(0.05, remaining))

    def send_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._state_lock:
            if self._finish_requested:
                self._raise_if_failed()
                raise RuntimeError("Cannot send Willow audio after finish")
            if self._finished.is_set():
                # The session ended on its own (handshake failure or cancellation)
                # while capture was still feeding it. Drop further audio quietly;
                # the underlying error surfaces through wait_ready()/finish().
                return
        self._enqueue(bytes(pcm), deadline=time.monotonic() + AUDIO_QUEUE_PUT_TIMEOUT)

    def finish(self) -> str:
        deadline = time.monotonic() + self._engine._connect_timeout + self._engine._result_timeout + 2.0
        enqueue_finish = False
        with self._state_lock:
            if not self._finish_requested:
                self._finish_requested = True
                enqueue_finish = True
        if enqueue_finish:
            self._enqueue(_STREAM_FINISH, deadline=deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._finished.wait(timeout=remaining):
            self.cancel()
            raise ModelError(MODEL_TRANSCRIBE_FAILED, "Timed out waiting for Willow streaming transcription")
        self._raise_if_failed()
        return self._transcript or ""

    def cancel(self) -> None:
        with self._state_lock:
            self._finish_requested = True
        self._cancelled.set()
        try:
            self._audio_queue.put_nowait(_STREAM_FINISH)
        except queue.Full:
            pass
        with self._socket_lock:
            socket = self._socket
        if socket is None:
            return
        try:
            socket.close()
        except Exception:
            pass
        self._finished.wait(timeout=1.0)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _enqueue(self, item: object, *, deadline: float) -> None:
        while not self._finished.is_set():
            if self._cancelled.is_set():
                raise ModelError(MODEL_TRANSCRIBE_FAILED, "Willow streaming session was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelError(MODEL_TRANSCRIBE_FAILED, "Timed out queueing Willow audio")
            try:
                self._audio_queue.put(item, timeout=min(0.1, remaining))
                return
            except queue.Full:
                continue
        self._raise_if_failed()
        raise RuntimeError("Willow streaming session ended before audio could be sent")

    def _run(self) -> None:
        try:
            self._engine._ensure_fresh_session()
            socket = self._engine._open_socket()
            with self._socket_lock:
                self._socket = socket
            with socket:
                if self._cancelled.is_set():
                    return
                protocol = WillowProtocol()
                self._engine._initialize_realtime_socket(socket, protocol)
                self._ready.set()
                pending = bytearray()
                while not self._cancelled.is_set():
                    item = self._audio_queue.get()
                    if item is _STREAM_FINISH:
                        break
                    pending.extend(item)
                    while len(pending) >= PCM_PACKET_BYTES:
                        packet = bytes(pending[:PCM_PACKET_BYTES])
                        del pending[:PCM_PACKET_BYTES]
                        self._engine._send(socket, protocol, "audio_packet", packet)

                if self._cancelled.is_set():
                    return
                if len(pending) % 2:
                    raise ValueError("Willow PCM stream ended with an incomplete sample")
                if pending:
                    self._engine._send(socket, protocol, "audio_packet", bytes(pending))
                self._engine._send(socket, protocol, "flush_packet", {})
                self._transcript = self._engine._wait_for_result(socket, protocol)
        except ModelError as exc:
            if not self._cancelled.is_set():
                self._error = exc
        except Exception as exc:
            if not self._cancelled.is_set():
                self._error = ModelError(MODEL_TRANSCRIBE_FAILED, f"Willow transcription failed: {exc}")
        finally:
            with self._socket_lock:
                self._socket = None
            self._finished.set()


class WillowEngine:
    def __init__(
        self,
        *,
        access_token: str,
        user_id: str | None = None,
        endpoint: str = PRODUCTION_TRANSCRIBE_URL,
        connect_timeout: float = 10.0,
        result_timeout: float = 15.0,
        refresh_token: str | None = None,
        expires_at: float | None = None,
        session_file: Path | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("A Willow access token is required")
        self._access_token = access_token
        self._user_id = user_id
        self._endpoint = endpoint
        self._connect_timeout = connect_timeout
        self._result_timeout = result_timeout
        self._refresh_token = refresh_token
        self._expires_at = expires_at
        self._session_file = session_file

    @classmethod
    def from_config(
        cls,
        config: DictationConfig,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "WillowEngine":
        session = load_session(env=env)
        source = os.environ if env is None else env
        return cls(
            access_token=session.access_token,
            user_id=session.user_id,
            endpoint=source.get("WILLOW_TRANSCRIBE_URL", PRODUCTION_TRANSCRIBE_URL),
            refresh_token=session.refresh_token,
            expires_at=session.expires_at,
            session_file=None if source.get("WILLOW_ACCESS_TOKEN") else resolve_session_path(source),
        )

    def to(self, device: str) -> "WillowEngine":
        return self

    def eval(self) -> "WillowEngine":
        return self

    def parameters(self) -> Iterable[Any]:
        return iter(())

    def transcribe(self, audio: Sequence[str], *, verbose: bool = False) -> list[Any]:
        text = self.transcribe_path(Path(audio[0])) if audio else ""
        return [type("WillowTranscript", (), {"text": text})()]

    def transcribe_path(self, path: Path) -> str:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16_000:
                raise ValueError("Willow requires 16 kHz mono signed 16-bit PCM WAV input")
            pcm = handle.readframes(handle.getnframes())
        return self.transcribe_pcm(pcm)

    def transcribe_pcm(self, pcm: bytes) -> str:
        session = self.start_streaming()
        session.send_audio(pcm)
        return session.finish()

    def start_streaming(self) -> WillowStreamingSession:
        return WillowStreamingSession(self)

    def _open_socket(self):
        query = {"interactionID": str(uuid4()), "client": "linux", "protocol_version": "3"}
        if self._user_id:
            query["userID"] = self._user_id
        endpoint = f"{self._endpoint}{'&' if '?' in self._endpoint else '?'}{urlencode(query)}"
        return connect(
            endpoint,
            additional_headers={"Authorization": f"Bearer {self._access_token}"},
            open_timeout=self._connect_timeout,
            close_timeout=0.5,
        )

    def _initialize_realtime_socket(self, socket: Any, protocol: WillowProtocol) -> None:
        self._send(socket, protocol, "health_packet", {})
        self._wait_for(socket, protocol, "health_packet", self._connect_timeout)
        for key, data in (
            ("selected_languages", []),
            ("glossary", []),
            ("type", {"type": "realtime"}),
            ("llm_type", "normal"),
            ("user_preferences", {"style_matching": {}, "scribe_style_matching": {}}),
            ("assistant_enabled", False),
            ("force_assistant", False),
            ("smart_insertion_enabled", False),
            ("press_enter_enabled", False),
        ):
            self._send(socket, protocol, key, data)

    def _ensure_fresh_session(self) -> None:
        if self._expires_at is None or self._expires_at > time.time() + 60:
            return
        previous_token = self._access_token
        refreshed = _refresh_session(
            WillowSession(
                access_token=self._access_token,
                user_id=self._user_id,
                refresh_token=self._refresh_token,
                expires_at=self._expires_at,
            )
        )
        self._access_token = refreshed.access_token
        self._user_id = refreshed.user_id or self._user_id
        self._refresh_token = refreshed.refresh_token
        self._expires_at = refreshed.expires_at
        if self._session_file is not None:
            _persist_refreshed_session(self._session_file, previous_token, refreshed)

    @staticmethod
    def _send(socket: Any, protocol: WillowProtocol, key: str, data: Any) -> None:
        for wire in protocol.encode(key, data):
            socket.send(wire)

    @staticmethod
    def _receive(socket: Any, protocol: WillowProtocol, timeout: float) -> dict[str, Any]:
        while True:
            wire = socket.recv(timeout=timeout)
            if isinstance(wire, str):
                raise ValueError("Willow sent a non-binary websocket message")
            envelope = protocol.accept(wire)
            if envelope is not None:
                return envelope

    def _wait_for(self, socket: Any, protocol: WillowProtocol, key: str, timeout: float) -> dict[str, Any]:
        while True:
            envelope = self._receive(socket, protocol, timeout)
            descriptor = envelope.get("d", {})
            if descriptor.get("key") == "error":
                raise ModelError(MODEL_TRANSCRIBE_FAILED, f"Willow protocol error: {descriptor.get('data')}")
            if descriptor.get("key") == key:
                return envelope

    def _wait_for_result(self, socket: Any, protocol: WillowProtocol) -> str:
        while True:
            try:
                envelope = self._receive(socket, protocol, self._result_timeout)
            except ConnectionClosedOK:
                return ""
            descriptor = envelope.get("d", {})
            if descriptor.get("key") == "error":
                raise ModelError(MODEL_TRANSCRIBE_FAILED, f"Willow protocol error: {descriptor.get('data')}")
            transcript = _extract_transcript(envelope)
            if transcript is not None:
                return transcript


def load_engine(config: DictationConfig) -> WillowEngine:
    return WillowEngine.from_config(config)


def warmup(engine: WillowEngine) -> None:
    engine.eval()


def transcribe_wav(engine: WillowEngine, path: str | Path) -> TranscriptionResult:
    try:
        text = engine.transcribe_path(Path(path))
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(MODEL_TRANSCRIBE_FAILED, str(exc)) from exc
    return TranscriptionResult(
        text=text,
        device="cloud",
        metadata={
            "backend": "willow",
            "model": "Automatic (server-selected Frontier Mini or Frontier Pro)",
        },
    )


def check_model_cache(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        session = load_session(env=env)
        if session.expires_at is not None and session.expires_at <= time.time() + 60 and not session.refresh_token:
            raise ModelError(MODEL_IMPORT_FAILED, "Willow session expired. Run `local-ai-dictation willow-session login` or import a fresh session.")
        ready = True
        error = None
    except ModelError as exc:
        ready = False
        error = str(exc)
    return {
        "checked": True,
        "cache_present": None,
        "cache_path": None,
        "model_id": "willow/frontier-auto",
        "import_ready": ready,
        "import_error": error,
        "detail": "Willow cloud session is ready" if ready else "Willow cloud login is required",
    }
