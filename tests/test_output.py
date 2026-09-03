from __future__ import annotations

from dataclasses import replace
import importlib
import io
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

main = importlib.import_module("local_ai_dictation.cli").main
dictation_module = importlib.import_module("local_ai_dictation.dictation")
output_module = importlib.import_module("local_ai_dictation.output")
DictationConfig = importlib.import_module("local_ai_dictation.types").DictationConfig
TranscriptionResult = importlib.import_module("local_ai_dictation.types").TranscriptionResult


class _ClipboardFailure:
    def copy(self, text: str) -> None:
        raise RuntimeError("clipboard backend unavailable")


class _ClipboardSuccess:
    def __init__(self):
        self.calls: list[str] = []

    def copy(self, text: str) -> None:
        self.calls.append(text)


class _FakeTorchCuda:
    @staticmethod
    def is_available() -> bool:
        raise AssertionError("torch CUDA must not be queried without a loaded NVIDIA driver")


class _FakeTorch:
    cuda = _FakeTorchCuda()


class _FakeModel:
    pass


class _GuardModel:
    def transcribe(self, audio, *, verbose=False):
        raise AssertionError("model.transcribe should not be called")



def test_emit_transcription_result_writes_json_stdout_and_output_file(tmp_path):
    output_path = tmp_path / "transcript.json"
    stdout = io.StringIO()
    status_stream = io.StringIO()
    clipboard = _ClipboardSuccess()
    config = DictationConfig(format="json", output_file=str(output_path), clipboard=True)

    warning = output_module.emit_transcription_result(
        TranscriptionResult(text="hello world", device="cpu"),
        config,
        pyperclip_module=clipboard,
        stdout=stdout,
        status_stream=status_stream,
    )

    rendered_stdout = stdout.getvalue()
    payload = json.loads(rendered_stdout)

    assert warning is None
    assert payload == {
        "schema_version": 1,
        "transcript": "hello world",
        "device": "cpu",
    }
    assert output_path.read_text(encoding="utf-8") == rendered_stdout
    assert clipboard.calls == ["hello world"]
    assert status_stream.getvalue() == ""


def test_emit_transcription_result_keeps_clipboard_failure_non_fatal_and_debug_only(tmp_path):
    output_path = tmp_path / "transcript.txt"
    stdout = io.StringIO()
    status_stream = io.StringIO()
    config = DictationConfig(format="text", output_file=str(output_path), clipboard=True)

    warning = output_module.emit_transcription_result(
        TranscriptionResult(text="hello world"),
        config,
        pyperclip_module=_ClipboardFailure(),
        stdout=stdout,
        status_stream=status_stream,
    )

    assert warning is not None
    assert warning.code == "CLIPBOARD_UNAVAILABLE"
    assert stdout.getvalue() == "hello world\n"
    assert output_path.read_text(encoding="utf-8") == "hello world\n"
    assert status_stream.getvalue() == ""

    debug_stdout = io.StringIO()
    debug_status = io.StringIO()
    debug_warning = output_module.emit_transcription_result(
        TranscriptionResult(text="hello world"),
        replace(config, debug=True, output_file=None),
        pyperclip_module=_ClipboardFailure(),
        stdout=debug_stdout,
        status_stream=debug_status,
    )

    assert debug_warning is not None
    assert debug_stdout.getvalue() == "hello world\n"
    assert "Clipboard warning: CLIPBOARD_UNAVAILABLE" in debug_status.getvalue()


def test_transcribe_once_skips_model_inference_for_silence(monkeypatch):
    monkeypatch.setattr("local_ai_dictation.dictation.has_probable_speech", lambda *args, **kwargs: False)

    transcription, temp_path, start, end = dictation_module._transcribe_once(
        DictationConfig(clipboard=False),
        _GuardModel(),
        b"\x00\x00" * 1024,
        16000,
    )

    assert transcription.text == ""
    assert transcription.metadata == {"backend": "whisper", "silence_filtered": True}
    assert temp_path == ""
    assert end >= start



def test_dictation_json_mode_keeps_status_off_stdout(monkeypatch, capsys):
    waits = iter([True, False])
    dictation_module._shutdown_event.clear()
    monkeypatch.setattr(dictation_module, "nvidia_driver_loaded", lambda: False)

    monkeypatch.setattr(
        "local_ai_dictation.dictation._load_runtime_dependencies",
        lambda debug: (object(), object(), _ClipboardSuccess(), _FakeTorch()),
    )
    monkeypatch.setattr(
        "local_ai_dictation.dictation._load_model",
        lambda config, nemo_asr, torch_module: (_FakeModel(), False, 0.0, 0.0),
    )
    monkeypatch.setattr(
        "local_ai_dictation.dictation.wait_for_enter_interruptible",
        lambda *args, **kwargs: next(waits),
    )
    monkeypatch.setattr(
        "local_ai_dictation.dictation.record_audio_interruptible",
        lambda config, pyaudio_module, sample_rate=16000: b"\x00\x00",
    )
    monkeypatch.setattr(
        "local_ai_dictation.dictation._transcribe_once",
        lambda config, model, audio_data, sample_rate: (
            TranscriptionResult(text="hello world", device="cpu"),
            "/tmp/fake.wav",
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr("local_ai_dictation.dictation.os.unlink", lambda path: None)

    exit_code = main(["dictation", "--format", "json", "--no-clipboard"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "schema_version": 1,
        "transcript": "hello world",
        "device": "cpu",
    }
    assert "Starting..." in captured.err
    assert "Press ENTER to start" in captured.err
    assert "hello world" not in captured.err
