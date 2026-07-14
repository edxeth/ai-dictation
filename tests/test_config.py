from __future__ import annotations

from local_ai_dictation.config import resolve_config
from local_ai_dictation.dictation import build_parser


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_defaults_apply_when_no_overrides(tmp_path):
    config = resolve_config(_parse([]), env={}, config_path=tmp_path / "missing.toml")

    assert config.backend == "whisper"
    assert config.cpu is False
    assert config.input_device is None
    assert config.vad is False
    assert config.max_silence_ms == 1200
    assert config.min_speech_ms == 300
    assert config.vad_mode == 2
    assert config.format == "text"
    assert config.output_file is None
    assert config.clipboard is True
    assert config.debug is False


def test_willow_backend_is_accepted_from_environment(tmp_path):
    config = resolve_config(
        _parse([]),
        env={"LOCAL_AI_DICTATION_BACKEND": "willow"},
        config_path=tmp_path / "missing.toml",
    )

    assert config.backend == "willow"


def test_config_file_overrides_defaults(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        backend = "whisper"
        cpu = true
        input_device = "USB Mic"
        vad = true
        max_silence_ms = 1800
        min_speech_ms = 450
        vad_mode = 1
        format = "json"
        output_file = "transcript.json"
        clipboard = false
        debug = true
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    config = resolve_config(_parse([]), env={}, config_path=config_path)

    assert config.backend == "whisper"
    assert config.cpu is True
    assert config.input_device == "USB Mic"
    assert config.vad is True
    assert config.max_silence_ms == 1800
    assert config.min_speech_ms == 450
    assert config.vad_mode == 1
    assert config.format == "json"
    assert config.output_file == "transcript.json"
    assert config.clipboard is False
    assert config.debug is True


def test_environment_overrides_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        backend = "whisper"
        cpu = false
        input_device = "USB Mic"
        vad = false
        max_silence_ms = 1800
        min_speech_ms = 450
        vad_mode = 1
        format = "text"
        output_file = "from-config.txt"
        clipboard = true
        debug = false
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    env = {
        "LOCAL_AI_DICTATION_BACKEND": "whisper",
        "LOCAL_AI_DICTATION_CPU": "1",
        "LOCAL_AI_DICTATION_INPUT_DEVICE": "9",
        "LOCAL_AI_DICTATION_VAD": "true",
        "LOCAL_AI_DICTATION_MAX_SILENCE_MS": "900",
        "LOCAL_AI_DICTATION_MIN_SPEECH_MS": "700",
        "LOCAL_AI_DICTATION_VAD_MODE": "3",
        "LOCAL_AI_DICTATION_FORMAT": "json",
        "LOCAL_AI_DICTATION_OUTPUT_FILE": "from-env.json",
        "LOCAL_AI_DICTATION_CLIPBOARD": "0",
        "LOCAL_AI_DICTATION_DEBUG": "yes",
    }

    config = resolve_config(_parse([]), env=env, config_path=config_path)

    assert config.backend == "whisper"
    assert config.cpu is True
    assert config.input_device == 9
    assert config.vad is True
    assert config.max_silence_ms == 900
    assert config.min_speech_ms == 700
    assert config.vad_mode == 3
    assert config.format == "json"
    assert config.output_file == "from-env.json"
    assert config.clipboard is False
    assert config.debug is True


def test_cli_overrides_environment(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        backend = "whisper"
        cpu = false
        input_device = "USB Mic"
        vad = false
        max_silence_ms = 1800
        min_speech_ms = 450
        vad_mode = 1
        format = "text"
        output_file = "from-config.txt"
        clipboard = true
        debug = false
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    env = {
        "LOCAL_AI_DICTATION_BACKEND": "whisper",
        "LOCAL_AI_DICTATION_CPU": "0",
        "LOCAL_AI_DICTATION_INPUT_DEVICE": "9",
        "LOCAL_AI_DICTATION_VAD": "false",
        "LOCAL_AI_DICTATION_MAX_SILENCE_MS": "900",
        "LOCAL_AI_DICTATION_MIN_SPEECH_MS": "700",
        "LOCAL_AI_DICTATION_VAD_MODE": "3",
        "LOCAL_AI_DICTATION_FORMAT": "text",
        "LOCAL_AI_DICTATION_OUTPUT_FILE": "from-env.json",
        "LOCAL_AI_DICTATION_CLIPBOARD": "1",
        "LOCAL_AI_DICTATION_DEBUG": "false",
    }

    args = _parse(
        [
            "--backend",
            "whisper",
            "--cpu",
            "--input-device",
            "Studio Mic",
            "--vad",
            "--max-silence-ms",
            "1500",
            "--min-speech-ms",
            "350",
            "--vad-mode",
            "2",
            "--format",
            "json",
            "--output-file",
            "from-cli.json",
            "--no-clipboard",
            "--debug",
        ]
    )
    config = resolve_config(args, env=env, config_path=config_path)

    assert config.backend == "whisper"
    assert config.cpu is True
    assert config.input_device == "Studio Mic"
    assert config.vad is True
    assert config.max_silence_ms == 1500
    assert config.min_speech_ms == 350
    assert config.vad_mode == 2
    assert config.format == "json"
    assert config.output_file == "from-cli.json"
    assert config.clipboard is False
    assert config.debug is True
