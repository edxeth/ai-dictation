"""Decoder and local-checkpoint tests for the packed Underdog Bark 3.0 format.

The fixtures are encoded here with constriction from known weights, following the
format description in Underdog's bark_loader.py, so each expected tensor comes
from the encoder's inputs rather than from the decoder under test.
"""

from __future__ import annotations

import json
import lzma
import threading
from pathlib import Path

import constriction
import numpy as np
import pytest
from safetensors.numpy import load_file

from local_ai_dictation import model as model_module
from local_ai_dictation.bark_format import COLUMN_CLASSES, BarkFormatError, decode

CONFIG = {"architectures": ["ParakeetForTDT"], "blank_token_id": 8192}
LEVELS = np.array([[0.5, 1.25], [2.0, 0.125], [3.5, 0.75]], np.float16)
# 0 is zero, 1..2 select +level, 3..4 select -level (K = 2 levels per sign).
CODES = np.array([[0, 1, 2, 3, 4], [4, 3, 0, 1, 2], [2, 0, 4, 1, 3]])
COLUMN_CLASS = np.array([0, 3, 0, 15, 3])
SCALES = np.array([0.5, 0.25], np.float16)
INTEGERS = np.array([[-15, 0, 15], [7, -3, 1]])  # 5-bit symmetric integers
RAW_HALF = np.array([1.5, -2.25, 0.0], np.float16)
RAW_LONG = np.array(7, np.int64)


def _signed_level(row: int, code: int) -> float:
    levels_per_sign = LEVELS.shape[1]
    if code == 0:
        return 0.0
    if code <= levels_per_sign:
        return float(LEVELS[row, code - 1])
    return -float(LEVELS[row, code - levels_per_sign - 1])


def _model(counts: np.ndarray):
    return constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)


def _levels_entry() -> tuple[dict, bytes]:
    rows, cols = CODES.shape
    alphabet = 2 * LEVELS.shape[1] + 1
    counts = np.ones((COLUMN_CLASSES, alphabet), np.uint32)
    coder = constriction.stream.stack.AnsCoder()
    # A stack coder decodes in reverse push order, so push the last class first.
    for group in reversed(range(COLUMN_CLASSES)):
        columns = np.nonzero(COLUMN_CLASS == group)[0]
        symbols = CODES[:, columns].T.reshape(-1).astype(np.int32)
        coder.encode_reverse(symbols, _model(counts[group].astype(np.float64)))
    bits = [(int(c) >> shift) & 1 for c in COLUMN_CLASS for shift in (3, 2, 1, 0)]
    meta = np.packbits(np.array(bits, np.uint8)).tobytes() + counts.tobytes()
    levels = lzma.compress(LEVELS.tobytes())
    words = coder.get_compressed().tobytes()
    entry = {
        "n": "encoder.linear.weight",
        "k": "levels2",
        "K": LEVELS.shape[1],
        "shape": [rows, cols],
        "lev": len(levels),
        "zl": 1,
        "meta": len(meta),
        "b": len(words),
    }
    return entry, levels + meta + words


def _integer_entry() -> tuple[dict, bytes]:
    counts = np.ones(31, np.uint32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(
        (INTEGERS.reshape(-1) + 15).astype(np.int32), _model(counts.astype(np.float64))
    )
    scale = lzma.compress(SCALES.tobytes())
    words = coder.get_compressed().tobytes()
    entry = {
        "n": "decoder.proj.weight",
        "k": "ansint5",
        "shape": list(INTEGERS.shape),
        "s": len(scale),
        "zs": 1,
        "meta": counts.nbytes,
        "b": len(words),
    }
    return entry, scale + counts.tobytes() + words


def _bark_bytes(*, format_name: str = "bark-levels-v3") -> bytes:
    levels, levels_blob = _levels_entry()
    integers, integers_blob = _integer_entry()
    half = lzma.compress(RAW_HALF.tobytes())
    raw_half = {
        "n": "norm.bias",
        "k": "raw",
        "dtype": "torch.float16",
        "shape": [3],
        "b": len(half),
        "zr": 1,
    }
    raw_long = {
        "n": "norm.num_batches_tracked",
        "k": "raw",
        "dtype": "torch.int64",
        "shape": [],
        "b": 8,
    }
    header = json.dumps(
        {
            "format": format_name,
            "index": [levels, integers, raw_half, raw_long],
            "config": json.dumps(CONFIG),
        }
    ).encode()
    return (
        len(header).to_bytes(8, "little")
        + header
        + levels_blob
        + integers_blob
        + half
        + RAW_LONG.tobytes()
    )


def test_decode_restores_every_tensor_kind_exactly():
    state, config = decode(_bark_bytes())

    expected_levels = [
        [_signed_level(row, int(code)) for code in CODES[row]]
        for row in range(CODES.shape[0])
    ]
    assert config == CONFIG
    assert state["encoder.linear.weight"].dtype == np.float16
    assert state["encoder.linear.weight"].tolist() == expected_levels
    assert state["decoder.proj.weight"].dtype == np.float32
    assert state["decoder.proj.weight"].tolist() == [
        [-7.5, 0.0, 7.5],
        [1.75, -0.75, 0.25],
    ]
    assert state["norm.bias"].dtype == np.float16
    assert state["norm.bias"].tolist() == [1.5, -2.25, 0.0]
    assert state["norm.num_batches_tracked"].dtype == np.int64
    assert state["norm.num_batches_tracked"].tolist() == 7


def test_decode_accepts_the_uncompressed_v2_format_name():
    state, _ = decode(_bark_bytes(format_name="bark-levels-v2"))

    assert state["norm.bias"].tolist() == [1.5, -2.25, 0.0]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda raw: raw[:4], "too short"),
        (lambda raw: raw[:-3], "truncated"),
        (lambda raw: (5).to_bytes(8, "little") + b"{oops", "not JSON"),
        (
            lambda raw: _bark_bytes(format_name="bark-levels-v9"),
            "Unsupported model.bark format: 'bark-levels-v9'",
        ),
    ],
)
def test_decode_rejects_damaged_or_unknown_files(damage, message):
    with pytest.raises(BarkFormatError, match=message):
        decode(damage(_bark_bytes()))


def test_decode_rejects_an_unknown_tensor_kind():
    header = json.dumps(
        {
            "format": "bark-levels-v3",
            "index": [{"n": "encoder.proj.weight", "k": "levels4", "shape": [1]}],
            "config": json.dumps(CONFIG),
        }
    ).encode()

    with pytest.raises(
        BarkFormatError, match="tensor kind 'levels4' for 'encoder.proj.weight'"
    ):
        decode(len(header).to_bytes(8, "little") + header)


class _Hub:
    """Fake Hugging Face downloads for the pinned Bark and tokenizer files."""

    def __init__(self, root: Path, packed: bytes) -> None:
        self.snapshot = root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "model.bark").write_bytes(packed)
        self.tokenizer = root / "tokenizer.json"
        self.tokenizer.write_text('{"model": "parakeet"}', encoding="utf-8")
        self.calls: list[tuple] = []

    def snapshot_download(
        self, repo_id, *, revision, allow_patterns, local_files_only=False
    ):
        self.calls.append(("snapshot", repo_id, revision, tuple(allow_patterns)))
        return str(self.snapshot)

    def hf_hub_download(self, repo_id, filename, *, revision):
        self.calls.append(("file", repo_id, filename, revision))
        return str(self.tokenizer)


@pytest.fixture
def bark_hub(tmp_path, monkeypatch):
    import huggingface_hub

    hub = _Hub(tmp_path, _bark_bytes())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return hub


def test_bark_is_decoded_once_into_a_kestrel_checkpoint(bark_hub, tmp_path):
    spec = model_module.KESTREL_MODELS["bark"]

    path = Path(model_module._resolve_model_path(spec))

    assert path == (
        tmp_path
        / "cache/local-ai-dictation/models/ConwayResearch--Underdog-Bark-3.0"
        / "85db2ad18ab41e534a222ee253c29191d77ed919"
    )
    assert sorted(item.name for item in path.iterdir()) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    weights = load_file(str(path / "model.safetensors"))
    assert weights["decoder.proj.weight"].tolist() == [
        [-7.5, 0.0, 7.5],
        [1.75, -0.75, 0.25],
    ]
    assert json.loads((path / "config.json").read_text()) == CONFIG
    assert (path / "tokenizer.json").read_text() == '{"model": "parakeet"}'
    assert bark_hub.calls == [
        (
            "snapshot",
            "ConwayResearch/Underdog-Bark-3.0",
            "85db2ad18ab41e534a222ee253c29191d77ed919",
            ("model.bark",),
        ),
        (
            "file",
            "nvidia/parakeet-tdt-0.6b-v3",
            "tokenizer.json",
            "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        ),
    ]
    # Only the published checkpoint and its builder lock; no staging left over.
    assert sorted(item.name for item in path.parent.iterdir()) == [
        f".{path.name}.lock",
        path.name,
    ]

    bark_hub.calls.clear()
    assert model_module._resolve_model_path(spec) == str(path)
    assert bark_hub.calls == []


def test_an_incomplete_bark_checkpoint_is_rebuilt(bark_hub):
    spec = model_module.KESTREL_MODELS["bark"]
    target = model_module._decoded_model_dir(spec)
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")

    assert model_module._resolve_model_path(spec) == str(target)

    assert json.loads((target / "config.json").read_text()) == CONFIG
    assert (target / "model.safetensors").is_file()


def test_a_failed_bark_decode_leaves_no_checkpoint_behind(bark_hub):
    (bark_hub.snapshot / "model.bark").write_bytes(b"not a bark file at all")
    spec = model_module.KESTREL_MODELS["bark"]

    with pytest.raises(BarkFormatError):
        model_module._resolve_model_path(spec)

    target = model_module._decoded_model_dir(spec)
    assert [item.name for item in target.parent.iterdir()] == [f".{target.name}.lock"]
    assert model_module.check_model_cache(backend="bark")["cache_present"] is False


def test_concurrent_first_loads_decode_once_and_never_expose_a_missing_checkpoint(
    bark_hub, monkeypatch
):
    import local_ai_dictation.bark_format as bark_format

    spec = model_module.KESTREL_MODELS["bark"]
    decoding = threading.Event()
    release = threading.Event()
    decodes = []
    real_decode = bark_format.decode

    def slow_decode(raw):
        decodes.append(threading.current_thread().name)
        decoding.set()
        assert release.wait(timeout=5)
        return real_decode(raw)

    monkeypatch.setattr(bark_format, "decode", slow_decode)
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def resolve(name):
        try:
            results[name] = model_module._resolve_model_path(spec)
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    first = threading.Thread(target=resolve, args=("first",), name="first")
    second = threading.Thread(target=resolve, args=("second",), name="second")
    first.start()
    assert decoding.wait(timeout=5)
    second.start()
    # The second builder waits for the first instead of decoding in parallel.
    second.join(timeout=0.3)
    assert second.is_alive()
    release.set()
    first.join(timeout=10)
    published = Path(results["first"])
    weights_inode = (published / "model.safetensors").stat().st_ino
    second.join(timeout=10)

    assert errors == []
    assert decodes == ["first"]
    assert results["second"] == results["first"]
    # The checkpoint the first caller received is the one still on disk.
    assert (published / "model.safetensors").stat().st_ino == weights_inode
    assert sorted(item.name for item in published.parent.iterdir()) == [
        ".85db2ad18ab41e534a222ee253c29191d77ed919.lock",
        published.name,
    ]


def test_a_write_failure_leaves_no_partial_checkpoint_and_releases_the_lock(
    bark_hub, monkeypatch
):
    import safetensors.numpy

    spec = model_module.KESTREL_MODELS["bark"]
    real_save = safetensors.numpy.save_file

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(safetensors.numpy, "save_file", disk_full)
    with pytest.raises(OSError, match="No space left on device"):
        model_module._resolve_model_path(spec)

    target = model_module._decoded_model_dir(spec)
    assert [item.name for item in target.parent.iterdir()] == [f".{target.name}.lock"]

    monkeypatch.setattr(safetensors.numpy, "save_file", real_save)
    assert model_module.check_model_cache(backend="bark")["cache_present"] is False
    path = model_module._resolve_model_path(spec)
    assert model_module.check_model_cache(backend="bark")["cache_present"] is True
    assert Path(path) == model_module._decoded_model_dir(spec)
