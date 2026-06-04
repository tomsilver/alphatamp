"""Tests for the parallel-collection refactor of ``experiments/spectre/spectre_collect``.

Covers two layers:

- ``_resolve_problem_ids`` / id-set precedence (fast).
- End-to-end parallel invocation of ``main`` on a live env (slow, marked
  ``@pytest.mark.slow``).
"""

# pylint: disable=protected-access

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf


def _load_spectre_collect_module() -> ModuleType:
    """Import ``experiments/spectre/spectre_collect.py`` without relying on sys.path.

    The ``experiments/`` directory is a scripts folder, not a package, so we
    load it by file path. Cached under ``"spectre_collect_module"`` to avoid
    re-importing on repeated calls within one pytest session.
    """
    if "spectre_collect_module" in sys.modules:
        return sys.modules["spectre_collect_module"]
    path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "spectre"
        / "spectre_collect.py"
    )
    spec = importlib.util.spec_from_file_location("spectre_collect_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["spectre_collect_module"] = module
    spec.loader.exec_module(module)
    return module


def _fake_spectre_cfg(start: int = 0, end: int = 5) -> object:
    """Build a stand-in ``CollectionConfig`` for id-resolution tests.

    ``_resolve_problem_ids`` only reads ``problem_seed_start`` and
    ``problem_seed_end`` off the object, so a namedtuple-ish stub suffices.
    """

    class _Stub:
        def __init__(self, s: int, e: int) -> None:
            self.problem_seed_start = s
            self.problem_seed_end = e

    return _Stub(start, end)


# ---------------------------------------------------------------------------
# _resolve_problem_ids (fast)
# ---------------------------------------------------------------------------


def test_resolve_uses_range_by_default() -> None:
    """With ``problem_ids`` unset, the [start,end) range is returned."""
    mod = _load_spectre_collect_module()
    cfg = OmegaConf.create({"problem_ids": None})
    ids = mod._resolve_problem_ids(cfg, _fake_spectre_cfg(0, 4))
    assert ids == [0, 1, 2, 3]


def test_resolve_override_takes_precedence() -> None:
    """Explicit ``problem_ids`` overrides the range."""
    mod = _load_spectre_collect_module()
    cfg = OmegaConf.create({"problem_ids": [10, 20, 30]})
    ids = mod._resolve_problem_ids(cfg, _fake_spectre_cfg(0, 100))
    assert ids == [10, 20, 30]


def test_resolve_dedups_and_sorts_override() -> None:
    """Duplicate/unsorted ids in the override are deduped and sorted."""
    mod = _load_spectre_collect_module()
    cfg = OmegaConf.create({"problem_ids": [30, 10, 10, 20]})
    ids = mod._resolve_problem_ids(cfg, _fake_spectre_cfg(0, 100))
    assert ids == [10, 20, 30]


def test_resolve_rejects_empty_override() -> None:
    """An empty override list raises — this is almost certainly a config bug."""
    mod = _load_spectre_collect_module()
    cfg = OmegaConf.create({"problem_ids": []})
    with pytest.raises(ValueError, match="non-empty"):
        mod._resolve_problem_ids(cfg, _fake_spectre_cfg(0, 5))


# ---------------------------------------------------------------------------
# End-to-end parallel run (slow, exercises ProcessPoolExecutor + spawn)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_parallel_collect_writes_all_episodes(tmp_path: Path) -> None:
    """Two problems with ``workers=2`` both land on disk via the spawn pool."""
    mod = _load_spectre_collect_module()
    cfg = OmegaConf.create(
        {
            "env": {
                "env_id": "kinder/ClutteredStorage2D-b5-v0",
                "env_variant": "clutteredstorage2d_b5",
                "model_name": "clutteredstorage2d",
                "model_kwargs": {"num_blocks": 5},
            },
            "split": "train",
            "num_problems": 2,
            "problem_seed_start": 0,
            "problem_seed_end": 2,
            "problem_ids": None,
            "workers": 2,
            "K_max": 3,
            "abstract_plan_timeout_s": 10.0,
            "refinement_timeout_s": 5.0,
            "num_sampling_attempts_per_step": 3,
            "max_trajectory_steps": 50,
            "heuristic_name": "hff",
            "refinement_seed_rule": "v1_blake2b_problem_skeleton",
            "collect_instrumentation": False,
            "state_path_depth": "s0_sL_only",
            "data_root": str(tmp_path),
        }
    )
    # Hydra's decorator wraps ``main`` into a CLI-driven function; call the
    # underlying wrapped implementation directly.
    mod.main.__wrapped__(cfg)

    episodes_dir = tmp_path / "raw" / "clutteredstorage2d_b5" / "train" / "episodes"
    files = sorted(episodes_dir.glob("ep_*.pkl.gz"))
    assert [p.name for p in files] == ["ep_00000.pkl.gz", "ep_00001.pkl.gz"]
    # No tmp residue from atomic writes.
    assert not list(episodes_dir.glob("*.tmp"))
    # Config YAML written exactly once by the parent, not by workers.
    configs = list((tmp_path / "configs").glob("collection_*.yaml"))
    assert len(configs) == 1
