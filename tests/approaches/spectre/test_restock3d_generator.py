"""Unit tests for the Restock3D generator and stratum-id encoding (pure Python)."""

from __future__ import annotations

from alphatamp.approaches.spectre.compare import stratum_of
from alphatamp.approaches.spectre.envs.restock3d import generator as G
from alphatamp.approaches.spectre.envs.restock3d import strata as S


def test_build_spec_deterministic() -> None:
    a = G.build_spec(7, 2)
    b = G.build_spec(7, 2)
    assert a == b
    assert G.build_spec(8, 2) != a  # a different seed differs


def test_build_spec_counts_match_recipe() -> None:
    for stratum, (n_small, n_tall, n_tall_reg, n_short_reg) in G.STRATA.items():
        spec = G.build_spec(0, stratum)
        assert spec.n_small == n_small
        assert spec.n_tall == n_tall
        assert len(spec.tall_region_ys) == n_tall_reg
        assert len(spec.short_region_ys) == n_short_reg
        assert len(spec.small_floor) == n_small
        assert len(spec.tall_floor) == n_tall


def test_sigma_definitions() -> None:
    spec = G.build_spec(0, 2)  # (3 small, 1 tall, 2 tall reg, 4 short reg)
    assert spec.sigma_tall == 2 - 1
    assert spec.sigma_short == 4 + (2 - 1) - 3


def test_goal_object_names() -> None:
    # r3 = 4 small + 2 tall goal objects; helper lists them in cube-then-block order.
    names = G.goal_object_names(G.build_spec(3, 3))
    assert names == [f"cube_goal{i}" for i in range(1, 5)] + [
        f"block_goal{i}" for i in range(1, 3)
    ]


def test_region_sampling_exclusion_radius() -> None:
    # Objects are region-sampled in the object band (fully-lateral layout, decisions/07 2026-08-16):
    # every pair of goal objects is >= the exclusion radius apart, all inside the band, axis-aligned.
    # No clutter is placed -- blockers were dropped (reach-over ordering is the difficulty; the front
    # grasp is not obstructed by a floor neighbour at the grasp config).
    import math

    for stratum in G.STRATA:
        for seed in range(5):
            spec = G.build_spec(seed, stratum)
            floor = spec.small_floor + spec.tall_floor
            for i, a in enumerate(floor):
                assert G._OBJECT_REGION_X[0] <= a[0] <= G._OBJECT_REGION_X[1]
                assert G._OBJECT_REGION_Y[0] <= a[1] <= G._OBJECT_REGION_Y[1]
                for b in floor[i + 1 :]:
                    assert (
                        math.hypot(a[0] - b[0], a[1] - b[1])
                        >= G._EXCLUSION_RADIUS - 1e-9
                    )
            assert not spec.clutter_floor  # blockers dropped (reach-over-only)


def test_problem_id_recovers_stratum() -> None:
    for split in ("train", "val", "test"):
        for stratum in S.STRATA:
            for index in (0, 1, 123, S.STRATUM_BAND - 1):
                pid = S.problem_id(split, stratum, index)
                assert stratum_of(pid) == stratum
                assert S.decode(pid) == (split, stratum, index)


def test_problem_id_splits_disjoint() -> None:
    ids = {S.problem_id(split, 0, 0) for split in ("train", "val", "test")}
    assert len(ids) == 3  # no split shares a scene seed


def test_index_overflow_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        S.problem_id("train", 0, S.STRATUM_BAND)
