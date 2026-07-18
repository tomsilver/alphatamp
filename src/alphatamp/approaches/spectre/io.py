"""Atomic gzip-pickle IO and light filesystem helpers for episode records."""

from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphatamp.approaches.spectre.schema import EpisodeRecord


def atomic_write_pickle_gz(obj: object, path: Path) -> None:
    """Write ``obj`` to ``path`` via ``<path>.tmp`` + rename, with fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_episode(path: Path) -> "EpisodeRecord":
    """Load and return an ``EpisodeRecord`` from a gzip-pickle file."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def list_episodes(split_dir: Path) -> list[Path]:
    """Sorted list of ``ep_*.pkl.gz`` paths under ``<split_dir>/episodes/``."""
    episodes_dir = split_dir / "episodes"
    if not episodes_dir.exists():
        return []
    return sorted(episodes_dir.glob("ep_*.pkl.gz"))


def scrub_partial_writes(split_dir: Path) -> int:
    """Delete ``*.tmp`` files left by killed writers.

    Returns count removed.
    """
    episodes_dir = split_dir / "episodes"
    if not episodes_dir.exists():
        return 0
    tmps = list(episodes_dir.glob("*.tmp"))
    for t in tmps:
        t.unlink()
    return len(tmps)
