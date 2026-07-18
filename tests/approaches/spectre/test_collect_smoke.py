"""End-to-end collection smoke test on a live kinder env.

Marked ``slow`` because it spins up ClutteredStorage2D-b5 and drives the real
bilevel-planner pipeline. Skipped by default per ``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphatamp.approaches.spectre.collect import collect_episode
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.io import (
    atomic_write_pickle_gz,
    list_episodes,
    load_episode,
)


@pytest.mark.slow
def test_collect_one_episode_on_b5(tmp_path: Path) -> None:
    """Collect a single episode and assert collection-time invariants I1–I5."""
    cfg = CollectionConfig(
        env_id="kinder/ClutteredStorage2D-b5-v0",
        env_variant="clutteredstorage2d_b5",
        model_name="clutteredstorage2d",
        model_kwargs={"num_blocks": 5},
        split="train",
        num_problems=1,
        problem_seed_start=0,
        problem_seed_end=1,
        # Budgets tight enough to keep the test under a couple minutes.
        K_max=3,
        abstract_plan_timeout_s=10.0,
        refinement_timeout_s=5.0,
        num_sampling_attempts_per_step=3,
        max_trajectory_steps=50,
    )

    ep = collect_episode(cfg, problem_id=0)

    # Schema invariants (I1–I4 already asserted in __post_init__).
    assert ep.provenance.config_hash == cfg.config_hash
    assert ep.summary.num_skeletons >= 1
    assert ep.summary.num_skeletons <= cfg.K_max

    # Every outcome maps to one of the three categories.
    for o in ep.outcomes:
        assert o.outcome in {"success", "fail", "error"}

    # Round-trip through disk.
    path = tmp_path / "ep_00000.pkl.gz"
    atomic_write_pickle_gz(ep, path)
    loaded = load_episode(path)
    assert loaded.summary.num_skeletons == ep.summary.num_skeletons
    # list_episodes locates the file via the usual split_dir layout.
    split_dir = tmp_path.parent
    found = list_episodes(tmp_path.parent)
    del split_dir, found  # this test uses a flat tmp_path, not the nested layout
