"""Tests for ``spectre.schema`` invariants and ``spectre.io`` pickle round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre.io import atomic_write_pickle_gz, load_episode
from alphatamp.approaches.spectre.schema import (
    EpisodeRecord,
    OutcomeRecord,
    SummaryBlock,
)


def test_build_toy_episode_validates() -> None:
    """Happy path: a legal episode passes ``__post_init__`` assertions."""
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    assert ep.summary.num_skeletons == 3
    assert ep.summary.first_success_idx == 2


def test_roundtrip_preserves_fields(tmp_path: Path) -> None:
    """Pickling + loading yields an equivalent record."""
    ep = build_toy_episode(problem_id=42, outcomes=("fail", "success", "fail"))
    path = tmp_path / "ep_00042.pkl.gz"
    atomic_write_pickle_gz(ep, path)
    loaded = load_episode(path)
    assert loaded.provenance.problem_id == 42
    assert len(loaded.skeleton_pool) == 3
    assert loaded.outcomes[1].outcome == "success"
    assert loaded.summary.first_success_idx == 1


def test_i1_mismatched_lengths() -> None:
    """I1: ``len(outcomes)`` must match ``len(skeleton_pool)``."""
    ep = build_toy_episode()
    with pytest.raises(AssertionError, match="I1"):
        EpisodeRecord(
            provenance=ep.provenance,
            initial_abstract_state=ep.initial_abstract_state,
            goal_atoms=ep.goal_atoms,
            object_registry=ep.object_registry,
            skeleton_pool=ep.skeleton_pool,
            outcomes=ep.outcomes[:-1],
            summary=ep.summary,
        )


def test_i3_summary_counts_must_sum() -> None:
    """I3: ``num_success + num_fail + num_error == num_skeletons``."""
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    bad_summary = SummaryBlock(
        num_skeletons=3,
        num_success=0,  # wrong
        num_fail=2,
        num_error=0,
        first_success_idx=None,
        total_wall_clock_s=0.3,
        pool_truncated=False,
    )
    with pytest.raises(AssertionError, match="I3"):
        EpisodeRecord(
            provenance=ep.provenance,
            initial_abstract_state=ep.initial_abstract_state,
            goal_atoms=ep.goal_atoms,
            object_registry=ep.object_registry,
            skeleton_pool=ep.skeleton_pool,
            outcomes=ep.outcomes,
            summary=bad_summary,
        )


def test_i4_first_success_idx_points_at_success() -> None:
    """I4: if ``first_success_idx`` is set it must reference a success outcome."""
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    # Mutate the summary's pointer to a non-success skeleton.
    bad_summary = SummaryBlock(
        num_skeletons=3,
        num_success=1,
        num_fail=2,
        num_error=0,
        first_success_idx=0,  # points to a fail
        total_wall_clock_s=0.6,
        pool_truncated=False,
    )
    with pytest.raises(AssertionError, match="I4"):
        EpisodeRecord(
            provenance=ep.provenance,
            initial_abstract_state=ep.initial_abstract_state,
            goal_atoms=ep.goal_atoms,
            object_registry=ep.object_registry,
            skeleton_pool=ep.skeleton_pool,
            outcomes=ep.outcomes,
            summary=bad_summary,
        )


def test_success_fail_error_indices() -> None:
    """Accessor helpers partition outcomes by type."""
    ep = build_toy_episode(outcomes=("fail", "success", "error"))
    assert ep.success_indices() == [1]
    assert ep.fail_indices() == [0]
    assert ep.error_indices() == [2]


def test_outcome_record_accepts_error_info() -> None:
    """``OutcomeRecord.error_info`` survives construction unchanged."""
    o = OutcomeRecord(
        skeleton_idx=0,
        outcome="error",
        refinement_wall_clock_s=0.5,
        refinement_seed=1,
        error_info={"cls": "RuntimeError", "msg": "boom"},
    )
    assert o.error_info == {"cls": "RuntimeError", "msg": "boom"}
