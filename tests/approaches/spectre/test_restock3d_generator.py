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


def test_build_task_config_structure() -> None:
    cfg = G.build_task_config(G.build_spec(3, 3))
    assert cfg["scene"] == "lab2"
    assert cfg["fixtures"]["cupboard"]["cupboard_1"]["shelf_heights"] == [0.508, 0.254]
    # every region_* has region_meta with a clearance + surface_z
    region_names = [n for n in cfg["regions"] if n.startswith("region_")]
    assert region_names
    for name in region_names:
        assert name in cfg["region_meta"]
        assert "cell_clearance" in cfg["region_meta"][name]
        assert "surface_z" in cfg["region_meta"][name]
    # tall-cell regions have the large clearance, short-cell the small one
    for name in region_names:
        clr = cfg["region_meta"][name]["cell_clearance"]
        assert clr in (G._TALL_CLEARANCE, G._SHORT_CLEARANCE)
    # goal_objects lists every goal cube/block and goal_state stays EMPTY (kinder checker)
    assert cfg["goal_state"] == []
    assert len(cfg["goal_objects"]) == 4 + 2  # r3 = 4 small + 2 tall goal objects


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
