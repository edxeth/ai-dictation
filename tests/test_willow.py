from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import wave

import msgpack
import pytest
from websockets.sync.server import serve

from local_ai_dictation.cli import main
from local_ai_dictation.types import DictationConfig
import local_ai_dictation.willow as willow_module
from local_ai_dictation.willow import (
    WillowEngine,
    WillowProtocol,
    WillowSession,
    load_session,
    resolve_session_path,
    session_path,
    transcribe_wav,
)


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm)


def test_load_session_reads_willow_linux_supabase_storage(tmp_path: Path):
    session = {
        "access_token": "secret-access-token",
        "user": {"id": "user-123", "email": "person@example.com"},
    }
    path = tmp_path / "supabase-session.json"
    path.write_text(json.dumps({"sb-db-auth-token": json.dumps(session)}), encoding="utf-8")

    loaded = load_session(path=path, env={})

    assert loaded.access_token == "secret-access-token"
    assert loaded.user_id == "user-123"


def test_owned_session_path_is_preferred_over_legacy_path(tmp_path: Path):
    env = {"XDG_STATE_HOME": str(tmp_path / "state")}
    owned = session_path(env)
    legacy = tmp_path / "state" / "willow-linux" / "supabase-session.json"
    owned.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    owned.write_text(json.dumps({"token": "owned"}), encoding="utf-8")
    legacy.write_text(json.dumps({"token": "legacy"}), encoding="utf-8")

    assert resolve_session_path(env) == owned


def test_legacy_session_is_migrated_to_minimal_owned_file(tmp_path: Path):
    env = {"XDG_STATE_HOME": str(tmp_path / "state")}
    legacy = tmp_path / "state" / "willow-linux" / "supabase-session.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "sb-db-auth-token": json.dumps(
                    {
                        "access_token": "legacy-access",
                        "refresh_token": "legacy-refresh",
                        "provider_token": "must-not-be-copied",
                        "expires_at": 4_000_000_000,
                        "user": {"id": "legacy-user", "email": "private@example.com"},
                    }
                )
            }
        ),
        encoding="utf-8",
    )

    loaded = load_session(env=env)
    owned = session_path(env)
    stored = json.loads(owned.read_text(encoding="utf-8"))

    assert loaded.access_token == "legacy-access"
    assert resolve_session_path(env) == owned
    assert stored == {
        "access_token": "legacy-access",
        "refresh_token": "legacy-refresh",
        "expires_at": 4_000_000_000.0,
        "user_id": "legacy-user",
    }
    assert owned.stat().st_mode & 0o777 == 0o600


def test_willow_session_import_cli_writes_private_minimal_file(tmp_path: Path, monkeypatch, capsys):
    state_home = tmp_path / "state"
    source = tmp_path / "official-session.json"
    source.write_text(
        json.dumps(
            {
                "access_token": "official-access",
                "refresh_token": "official-refresh",
                "provider_token": "remove-me",
                "expires_at": 4_000_000_000,
                "user": {"id": "official-user", "email": "private@example.com"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    assert main(["willow-session", "import", str(source), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    owned = state_home / "local-ai-dictation" / "willow-session.json"
    stored = json.loads(owned.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["session_path"] == str(owned)
    assert "access_token" not in payload
    assert "refresh_token" not in payload
    assert "provider_token" not in stored
    assert stored["user_id"] == "official-user"
    assert owned.stat().st_mode & 0o777 == 0o600


def test_load_session_accepts_explicit_environment_token():
    loaded = load_session(env={"WILLOW_ACCESS_TOKEN": "environment-token", "WILLOW_USER_ID": "env-user"})

    assert loaded.access_token == "environment-token"
    assert loaded.user_id == "env-user"


def test_refresh_response_converts_expires_in_to_absolute_expiry(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "access_token": "fresh-token",
                    "refresh_token": "fresh-refresh-token",
                    "expires_in": 3600,
                    "user": {"id": "user-123"},
                }
            ).encode("utf-8")

    monkeypatch.setattr(willow_module, "urlopen", lambda request, timeout: Response())
    monkeypatch.setattr(willow_module.time, "time", lambda: 1_000.0)

    refreshed = willow_module._refresh_session(
        WillowSession("old-token", "user-123", "old-refresh", 1.0)
    )

    assert refreshed.expires_at == 4_600.0


def test_expired_session_is_refreshed_and_persisted(tmp_path: Path, monkeypatch):
    stored = {
        "access_token": "expired-token",
        "refresh_token": "old-refresh-token",
        "expires_at": 1,
        "user": {"id": "user-123"},
    }
    path = tmp_path / "supabase-session.json"
    path.write_text(json.dumps({"sb-db-auth-token": json.dumps(stored)}), encoding="utf-8")
    monkeypatch.setattr(
        willow_module,
        "_refresh_session",
        lambda session: WillowSession(
            access_token="fresh-token",
            user_id="user-123",
            refresh_token="fresh-refresh-token",
            expires_at=4_000_000_000,
        ),
    )
    engine = WillowEngine(
        access_token="expired-token",
        user_id="user-123",
        refresh_token="old-refresh-token",
        expires_at=1,
        session_file=path,
    )

    engine._ensure_fresh_session()

    persisted = json.loads(next(iter(json.loads(path.read_text(encoding="utf-8")).values())))
    assert persisted["access_token"] == "fresh-token"
    assert persisted["refresh_token"] == "fresh-refresh-token"
    assert persisted["expires_at"] == 4_000_000_000
    assert path.stat().st_mode & 0o777 == 0o600


def test_willow_protocol_reassembles_chunked_result():
    protocol = WillowProtocol(maximum_message_bytes=256)
    wires = protocol.encode(
        "dictation_result",
        {"actions": [{"type": "paste", "text": "Mock transcript"}], "padding": "x" * 500},
    )

    received = None
    for wire in reversed(wires):
        envelope = protocol.accept(wire)
        if envelope is not None:
            received = envelope

    assert received is not None
    assert received["d"]["key"] == "dictation_result"
    assert received["d"]["data"]["actions"][0]["text"] == "Mock transcript"


def test_streaming_session_waits_for_health_handshake():
    health_received = threading.Event()
    allow_health = threading.Event()

    def handler(socket):
        incoming = WillowProtocol()
        outgoing = WillowProtocol()
        while True:
            envelope = incoming.accept(socket.recv())
            if envelope is None or envelope["d"]["key"] != "health_packet":
                continue
            health_received.set()
            assert allow_health.wait(timeout=2)
            socket.send(outgoing.encode("health_packet", {"ok": True})[0])
            try:
                while socket.recv():
                    pass
            except Exception:
                return

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        engine = WillowEngine(access_token="test-token", endpoint=f"ws://127.0.0.1:{port}/transcribe")
        session = engine.start_streaming()
        ready = threading.Event()
        waiter = threading.Thread(target=lambda: (session.wait_ready(), ready.set()), daemon=True)
        waiter.start()

        assert health_received.wait(timeout=2)
        assert ready.is_set() is False
        allow_health.set()
        assert ready.wait(timeout=2)

        session.cancel()
        waiter.join(timeout=1)
        server.shutdown()
        thread.join(timeout=2)


def test_streaming_session_cancel_unblocks_readiness_during_connect(monkeypatch):
    open_started = threading.Event()
    release_open = threading.Event()
    engine = WillowEngine(access_token="test-token", connect_timeout=5.0, result_timeout=1.0)

    def blocked_open():
        open_started.set()
        assert release_open.wait(timeout=2)
        raise RuntimeError("cancelled connection")

    monkeypatch.setattr(engine, "_open_socket", blocked_open)
    session = engine.start_streaming()
    assert open_started.wait(timeout=1)
    ready_errors: list[Exception] = []

    def wait_ready():
        try:
            session.wait_ready()
        except Exception as exc:
            ready_errors.append(exc)

    waiter = threading.Thread(target=wait_ready, daemon=True)
    waiter.start()
    cancel_started = time.perf_counter()
    session.cancel()
    cancel_elapsed = time.perf_counter() - cancel_started
    waiter.join(timeout=0.5)
    release_open.set()

    assert cancel_elapsed < 0.2
    assert waiter.is_alive() is False
    assert ready_errors


def test_streaming_session_finish_unblocks_when_full_queue_is_cancelled(monkeypatch):
    open_started = threading.Event()
    release_open = threading.Event()
    engine = WillowEngine(access_token="test-token", connect_timeout=1.0, result_timeout=1.0)

    def blocked_open():
        open_started.set()
        assert release_open.wait(timeout=3)
        raise RuntimeError("cancelled connection")

    monkeypatch.setattr(engine, "_open_socket", blocked_open)
    session = engine.start_streaming()
    assert open_started.wait(timeout=1)
    for _ in range(willow_module.AUDIO_QUEUE_MAX_CHUNKS):
        session.send_audio(b"\x00\x00")

    finish_errors: list[Exception] = []

    def finish():
        try:
            session.finish()
        except Exception as exc:
            finish_errors.append(exc)

    finisher = threading.Thread(target=finish, daemon=True)
    finisher.start()
    time.sleep(0.15)
    session.cancel()

    finisher.join(timeout=1)
    release_open.set()

    assert finisher.is_alive() is False
    assert finish_errors


def test_wait_for_result_treats_normal_session_close_as_empty_transcript():
    from websockets.exceptions import ConnectionClosedOK

    class _NormallyClosedSocket:
        def recv(self, timeout=None):
            raise ConnectionClosedOK(None, None)

    engine = WillowEngine(access_token="test-token")

    assert engine._wait_for_result(_NormallyClosedSocket(), WillowProtocol()) == ""


def test_streaming_session_sends_audio_before_flush():
    pcm = bytes(range(256)) * 8
    audio_received = threading.Event()
    flush_received = threading.Event()
    received_audio: list[bytes] = []

    def handler(socket):
        incoming = WillowProtocol()
        outgoing = WillowProtocol()
        while True:
            envelope = incoming.accept(socket.recv())
            if envelope is None:
                continue
            key = envelope["d"]["key"]
            if key == "health_packet":
                socket.send(outgoing.encode("health_packet", {"ok": True})[0])
            elif key == "audio_packet":
                received_audio.append(envelope["d"]["data"])
                audio_received.set()
            elif key == "flush_packet":
                flush_received.set()
                socket.send(
                    outgoing.encode(
                        "dictation_result",
                        {"actions": [{"type": "paste", "text": "Realtime transcript"}]},
                    )[0]
                )
                return

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        engine = WillowEngine(access_token="test-token", endpoint=f"ws://127.0.0.1:{port}/transcribe")

        session = engine.start_streaming()
        session.wait_ready()
        session.send_audio(pcm[:1024])

        assert audio_received.wait(timeout=2)
        assert flush_received.is_set() is False

        session.send_audio(pcm[1024:])
        result = session.finish()
        server.shutdown()
        thread.join(timeout=2)

    assert result == "Realtime transcript"
    assert b"".join(received_audio) == pcm


def test_transcribe_wav_sends_willow_handshake_and_recording(tmp_path: Path):
    pcm = bytes(range(256)) * 10
    fixture = tmp_path / "spoken.wav"
    _write_wav(fixture, pcm)
    received: list[dict] = []
    request: dict[str, str] = {}

    def handler(socket):
        request["path"] = socket.request.path
        request["authorization"] = socket.request.headers.get("Authorization", "")
        incoming = WillowProtocol()
        outgoing = WillowProtocol(maximum_message_bytes=256)
        while True:
            envelope = incoming.accept(socket.recv())
            if envelope is None:
                continue
            received.append(envelope)
            key = envelope["d"]["key"]
            if key == "health_packet":
                socket.send(outgoing.encode("health_packet", {"ok": True})[0])
            if key == "flush_packet":
                for wire in reversed(
                    outgoing.encode(
                        "dictation_result",
                        {"actions": [{"type": "paste", "text": "Recorded speech transcript"}], "padding": "x" * 500},
                    )
                ):
                    socket.send(wire)
                return

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        engine = WillowEngine(
            access_token="test-token",
            user_id="user-123",
            endpoint=f"ws://127.0.0.1:{port}/transcribe",
        )
        result = transcribe_wav(engine, fixture)
        server.shutdown()
        thread.join(timeout=2)

    assert result.text == "Recorded speech transcript"
    assert result.device == "cloud"
    assert result.metadata["backend"] == "willow"
    assert request["authorization"] == "Bearer test-token"
    assert "client=linux" in request["path"]
    assert "protocol_version=3" in request["path"]
    assert "userID=user-123" in request["path"]
    keys = [envelope["d"]["key"] for envelope in received]
    assert keys == [
        "health_packet",
        "selected_languages",
        "glossary",
        "type",
        "llm_type",
        "user_preferences",
        "assistant_enabled",
        "force_assistant",
        "smart_insertion_enabled",
        "press_enter_enabled",
        "audio_packet",
        "audio_packet",
        "audio_packet",
        "flush_packet",
    ]
    audio = b"".join(envelope["d"]["data"] for envelope in received if envelope["d"]["key"] == "audio_packet")
    assert audio == pcm


def test_check_model_cache_rejects_expired_unrefreshable_session(tmp_path: Path):
    path = tmp_path / "state" / "willow-linux" / "supabase-session.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "sb-db-auth-token": json.dumps(
                    {"access_token": "expired-token", "expires_at": 1, "user": {"id": "user-123"}}
                )
            }
        ),
        encoding="utf-8",
    )

    status = willow_module.check_model_cache(env={"XDG_STATE_HOME": str(tmp_path / "state")})

    assert status["import_ready"] is False
    assert "refresh token" in status["import_error"]


def test_willow_engine_requires_a_session(monkeypatch):
    monkeypatch.delenv("WILLOW_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("HOME", "/path/that/does/not/exist")
    config = DictationConfig(backend="willow")

    with pytest.raises(Exception, match="Willow session required"):
        WillowEngine.from_config(config, env={"HOME": "/path/that/does/not/exist"})
