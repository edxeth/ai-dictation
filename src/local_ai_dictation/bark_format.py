"""Exact decoder for Underdog's packed ``model.bark`` weight format.

Adapted from ``bark_loader.py`` in ConwayResearch/Underdog-Bark-3.0 (CC-BY-4.0,
Conway Research), formats "bark-levels-v2" and "bark-levels-v3". The app vendors
the decoding steps instead of running the downloaded script, and returns tensors
in the narrowest dtype that holds every decoded value exactly:

- "levels2" encoder weights are signed fp16 levels, so they stay float16;
- "ansintN" matrices are N-bit integers times an fp16 row scale, whose product
  needs float32;
- "raw" tensors keep their stored dtype.
"""

from __future__ import annotations

import json
import lzma
from typing import Any

import numpy as np

FORMATS = ("bark-levels-v2", "bark-levels-v3")
# Each levels2 tensor splits its columns into this many classes, each with its
# own symbol model.
COLUMN_CLASSES = 16
_HEADER_LENGTH_BYTES = 8
_RAW_DTYPES = {"torch.int64": np.int64, "torch.int32": np.int32}


class BarkFormatError(ValueError):
    """The bytes are not a supported model.bark file."""


def _decompress(blob: bytes, compressed: Any) -> bytes:
    return lzma.decompress(blob) if compressed else blob


def _ans_decoder(words: np.ndarray):
    import constriction

    return constriction.stream.stack.AnsCoder(words)


def _categorical(counts: np.ndarray):
    import constriction

    return constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)


class _Reader:
    def __init__(self, raw: bytes, offset: int) -> None:
        self._raw = raw
        self._offset = offset

    def take(self, size: int) -> bytes:
        end = self._offset + size
        if size < 0 or end > len(self._raw):
            raise BarkFormatError(
                f"model.bark is truncated: needs bytes {self._offset}..{end}, has {len(self._raw)}"
            )
        blob = self._raw[self._offset : end]
        self._offset = end
        return blob


def _decode_levels(entry: dict[str, Any], reader: _Reader) -> np.ndarray:
    shape = entry["shape"]
    levels_per_sign = int(entry["K"])
    rows, cols = shape[0], int(np.prod(shape[1:]))
    levels = np.frombuffer(
        _decompress(reader.take(entry["lev"]), entry.get("zl")), np.float16
    ).reshape(rows, levels_per_sign)
    meta = reader.take(entry["meta"])
    class_bytes = (cols * 4 + 7) // 8
    bits = np.unpackbits(np.frombuffer(meta[:class_bytes], np.uint8))[: cols * 4]
    nibbles = np.concatenate(
        [np.zeros((cols, 4), np.uint8), bits.reshape(cols, 4)], axis=1
    )
    column_class = np.packbits(nibbles, axis=1)[:, 0]
    counts = (
        np.frombuffer(meta[class_bytes:], np.uint32)
        .reshape(COLUMN_CLASSES, 2 * levels_per_sign + 1)
        .astype(np.float64)
    )
    words = np.frombuffer(reader.take(entry["b"]), np.uint32).copy()

    decoder = _ans_decoder(words)
    codes = np.empty((rows, cols), np.int64)
    for column_group in range(COLUMN_CLASSES):
        columns = np.nonzero(column_class == column_group)[0]
        symbols = decoder.decode(
            _categorical(counts[column_group]), rows * len(columns)
        )
        codes[:, columns] = symbols.reshape(len(columns), rows).T

    # Code 0 is zero, 1..K select +level, K+1..2K select -level.
    table = np.concatenate([np.zeros((rows, 1), np.float16), levels], axis=1)
    magnitude = np.where(codes > levels_per_sign, codes - levels_per_sign, codes)
    values = np.take_along_axis(table, magnitude, axis=1)
    return np.where(codes > levels_per_sign, -values, values).reshape(shape)


def _decode_integers(entry: dict[str, Any], reader: _Reader) -> np.ndarray:
    shape = entry["shape"]
    bits = int(entry["k"][len("ansint") :])
    scale = np.frombuffer(
        _decompress(reader.take(entry["s"]), entry.get("zs")), np.float16
    ).astype(np.float32)
    counts = np.frombuffer(reader.take(entry["meta"]), np.uint32).astype(np.float64)
    words = np.frombuffer(reader.take(entry["b"]), np.uint32).copy()
    symbols = _ans_decoder(words).decode(_categorical(counts), int(np.prod(shape)))
    integers = (symbols - (2 ** (bits - 1) - 1)).astype(np.float32)
    return (integers.reshape(len(scale), -1) * scale[:, None]).reshape(shape)


def _decode_raw(entry: dict[str, Any], reader: _Reader) -> np.ndarray:
    dtype = _RAW_DTYPES.get(entry["dtype"], np.float16)
    blob = _decompress(reader.take(entry["b"]), entry.get("zr"))
    return np.frombuffer(blob, dtype).copy().reshape(entry["shape"])


def decode(raw: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the state dict (tensor name -> array) and the model config of a model.bark file."""
    if len(raw) < _HEADER_LENGTH_BYTES:
        raise BarkFormatError(f"model.bark is too short: {len(raw)} bytes")
    header_length = int.from_bytes(raw[:_HEADER_LENGTH_BYTES], "little")
    header_end = _HEADER_LENGTH_BYTES + header_length
    try:
        header = json.loads(raw[_HEADER_LENGTH_BYTES:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BarkFormatError(f"model.bark header is not JSON: {exc}") from exc
    if not isinstance(header, dict) or header.get("format") not in FORMATS:
        found = header.get("format") if isinstance(header, dict) else header
        raise BarkFormatError(f"Unsupported model.bark format: {found!r}")

    reader = _Reader(raw, header_end)
    state: dict[str, np.ndarray] = {}
    for entry in header["index"]:
        kind = entry["k"]
        if kind == "levels2":
            state[entry["n"]] = _decode_levels(entry, reader)
        elif kind.startswith("ansint"):
            state[entry["n"]] = _decode_integers(entry, reader)
        elif kind == "raw":
            state[entry["n"]] = _decode_raw(entry, reader)
        else:
            raise BarkFormatError(
                f"Unsupported model.bark tensor kind {kind!r} for {entry['n']!r}"
            )
    return state, json.loads(header["config"])
