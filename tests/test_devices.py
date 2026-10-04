from __future__ import annotations

import json

from local_ai_dictation.audio import list_input_devices
from local_ai_dictation.cli import main
from local_ai_dictation.types import AudioDevice


class _FakePyAudioInstance:
    def __init__(
        self, devices: list[dict], host_apis: dict[int, str], default_index: int | None
    ):
        self._devices = devices
        self._host_apis = host_apis
        self._default_index = default_index
        self.terminated = False

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self._devices[index]

    def get_host_api_info_by_index(self, index: int) -> dict:
        return {"name": self._host_apis[index]}

    def get_default_input_device_info(self) -> dict:
        if self._default_index is None:
            raise OSError("no default input device")
        return {"index": self._default_index}

    def terminate(self) -> None:
        self.terminated = True


class _FakePyAudioModule:
    def __init__(
        self, devices: list[dict], host_apis: dict[int, str], default_index: int | None
    ):
        self._instance = _FakePyAudioInstance(devices, host_apis, default_index)

    def PyAudio(self) -> _FakePyAudioInstance:
        return self._instance


def test_list_input_devices_filters_and_sorts_by_device_id():
    fake_pyaudio = _FakePyAudioModule(
        devices=[
            {
                "index": 9,
                "name": "Speakers",
                "maxInputChannels": 0,
                "defaultSampleRate": 48000,
                "hostApi": 0,
            },
            {
                "index": 5,
                "name": "USB Mic",
                "maxInputChannels": 1,
                "defaultSampleRate": 44100,
                "hostApi": 0,
            },
            {
                "index": 2,
                "name": "Built-in Mic",
                "maxInputChannels": 2,
                "defaultSampleRate": 16000,
                "hostApi": 1,
            },
        ],
        host_apis={0: "PulseAudio", 1: "ALSA"},
        default_index=2,
    )

    devices = list_input_devices(fake_pyaudio)

    assert [device.id for device in devices] == [2, 5]
    assert devices[0].host_api == "ALSA"
    assert devices[0].is_default_candidate is True
    assert devices[1].host_api == "PulseAudio"
    assert devices[1].is_default_candidate is False
    assert fake_pyaudio._instance.terminated is True


def test_devices_command_emits_stable_json_schema(monkeypatch, capsys):
    monkeypatch.setattr(
        "local_ai_dictation.cli.list_input_devices",
        lambda: [
            AudioDevice(
                id=2,
                name="Built-in Mic",
                default_sample_rate=16000,
                max_input_channels=2,
                host_api="ALSA",
                is_default_candidate=True,
            )
        ],
    )

    exit_code = main(["devices", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {
        "schema_version": 1,
        "devices": [
            {
                "id": 2,
                "name": "Built-in Mic",
                "default_sample_rate": 16000,
                "max_input_channels": 2,
                "host_api": "ALSA",
                "is_default_candidate": True,
            }
        ],
    }


def test_devices_command_reports_empty_array_when_no_devices(monkeypatch, capsys):
    monkeypatch.setattr("local_ai_dictation.cli.list_input_devices", lambda: [])

    exit_code = main(["devices", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {"schema_version": 1, "devices": []}
