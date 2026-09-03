"""GPU availability guards that avoid loading device drivers."""

from __future__ import annotations

from collections.abc import Iterable
import os
import re


_NVIDIA_MODULE_LINE = re.compile(r"^nvidia(?:_[^\s]+)?\s")


def nvidia_driver_loaded(
    source: str | os.PathLike[str] | Iterable[str] = "/proc/modules",
) -> bool:
    """Return whether a proprietary NVIDIA kernel module is already loaded."""
    if isinstance(source, (str, os.PathLike)):
        try:
            with open(source, encoding="utf-8") as modules:
                return any(_NVIDIA_MODULE_LINE.match(line) for line in modules)
        except OSError:
            return False

    return any(_NVIDIA_MODULE_LINE.match(line) for line in source)
