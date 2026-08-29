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


def _migrate(ep: "EpisodeRecord") -> "EpisodeRecord":
    """Fill trailing-nullable fields missing on pickles written before they existed.

    Frozen-dataclass pickles restore via ``__dict__`` and do NOT re-run ``__init__``/
    ``__post_init__``, so a record written before the geometry/evidence layer landed
    lacks ``scene_geometry`` / ``aux_labels`` (on the episode) and ``post_mortem`` (on
    each outcome); accessing them would raise ``AttributeError``. Set the field defaults
    in place via ``object.__setattr__`` (the dataclasses are frozen). This keeps existing
    RT2D/kinder corpora loadable unchanged; DD2D is re-collected regardless.

    v3 extends the same shim to ``ProvenanceBlock.gen_params``.
    """
    for obj, attrs in [
        (ep, ("scene_geometry", "aux_labels")),
        (getattr(ep, "provenance", None), ("gen_params",)),
        *[(o, ("post_mortem",)) for o in getattr(ep, "outcomes", ())],
    ]:
        if obj is None:
            continue
        for name in attrs:
            if not hasattr(obj, name):
                object.__setattr__(obj, name, None)
    return ep


def load_episode(path: Path) -> "EpisodeRecord":
    """Load and return an ``EpisodeRecord`` from a gzip-pickle file (migrating any
    pre-v2.2.1 record so the optional geometry/evidence fields are present)."""
    with gzip.open(path, "rb") as f:
        return _migrate(pickle.load(f))


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
