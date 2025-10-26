"""Data acquisition and generation helpers."""

from pathlib import Path
from typing import Iterable

from . import config


def list_raw_files(suffixes: Iterable[str] | None = None) -> list[Path]:
    """Return raw data files optionally filtered by suffix."""

    base = config.RAW_DATA_DIR
    if suffixes is None:
        return sorted(base.glob("*"))
    suffixes = tuple(suffixes)
    return sorted(path for path in base.glob("*") if path.suffix in suffixes)
