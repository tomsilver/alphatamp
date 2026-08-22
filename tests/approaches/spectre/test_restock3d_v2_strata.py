"""Banding + budgets for the restock3d_v2 full collection (5 strata).

The v2 collection uses five strata, one more than ``compare.STRATUM_BAND = SPLIT_BAND // 4``
fits, so ``strata_v2`` owns a local ``//5`` band. These tests pin the collision fix: a fifth
stratum must not overflow into the next split's band (the ``train/s4 == val/s0`` bug).
"""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.envs.restock3d import generator as G
from alphatamp.approaches.spectre.envs.restock3d import strata_v2 as S


def test_five_strata_banding_injective_and_roundtrips() -> None:
    seen: dict[int, tuple] = {}
    for split in ("train", "val", "test"):
        for stratum in S.STRATA:
            # 0..80 comfortably covers 50/15/15 keepers + the resample cushion.
            for index in (0, 1, 50, 80):
                pid = S.problem_id(split, stratum, index)
                assert pid not in seen, (pid, seen.get(pid), (split, stratum, index))
                seen[pid] = (split, stratum, index)
                assert S.decode(pid) == (split, stratum, index)
                assert (
                    S.stratum_of(pid) == stratum
                )  # returns 0..4, no min(3, ...) clamp


def test_stratum_4_does_not_collide_with_next_split() -> None:
    # The exact bug the //5 band fixes: with compare's //4 band this pair is equal.
    assert S.problem_id("train", 4, 0) != S.problem_id("val", 0, 0)


def test_recipe_keys_committed_in_generator() -> None:
    assert set(S.STRATA) == {0, 1, 2, 3, 4}
    # 2x2/3x3/4x4 reuse the pilot keys; 3x4/4x3 are the new asymmetric keys.
    assert S.RECIPE_KEYS == {0: 11, 1: 12, 2: 14, 3: 15, 4: 13}
    for stratum in S.STRATA:
        key = S.recipe_key(stratum)
        assert key in G.STRATA, key
        n_small, n_tall = G.STRATA[key][0], G.STRATA[key][1]
        n_tall_cfg, n_short_cfg = S.CONFIGS[stratum]
        assert (n_tall, n_small) == (n_tall_cfg, n_short_cfg)


def test_budgets_cover_every_stratum() -> None:
    for stratum in S.STRATA:
        k_max, r_cap = S.budget(stratum)
        assert k_max > 0 and r_cap > 0
    # The crowded 7-8-object strata are capped at K_max=75 (Table A capture rates).
    assert S.budget(2)[0] == S.budget(3)[0] == S.budget(4)[0] == 75


def test_index_overflow_rejected() -> None:
    with pytest.raises(ValueError):
        S.problem_id("train", 0, S.V2_STRATUM_BAND)


def test_sizes_per_stratum_light_full_heavy_halved() -> None:
    for stratum in S.STRATA:
        sz = S.sizes(stratum)
        assert sz["train"] > 0 and sz["val"] > 0 and sz["test"] > 0
    # Light strata (2x2, 3x3) full; the three crowded strata halved to 25/10/10.
    assert S.sizes(0) == S.sizes(1) == {"train": 50, "val": 15, "test": 15}
    for stratum in (2, 3, 4):
        assert S.sizes(stratum) == {"train": 25, "val": 10, "test": 10}
    # Totals: 175 train / 60 val / 60 test = 295.
    totals = {
        sp: sum(S.sizes(s)[sp] for s in S.STRATA) for sp in ("train", "val", "test")
    }
    assert totals == {"train": 175, "val": 60, "test": 60}
    with pytest.raises(ValueError):
        S.sizes(99)


def test_per_worker_gb_cover_every_stratum_heavy_ge_light() -> None:
    for stratum in S.STRATA:
        assert S.per_worker_gb(stratum) > 0
    # Crowded strata peak higher than the light ones (drives fewer workers).
    assert min(S.per_worker_gb(s) for s in (2, 3, 4)) >= max(
        S.per_worker_gb(s) for s in (0, 1)
    )
    with pytest.raises(ValueError):
        S.per_worker_gb(99)


def test_sequential_order_is_permutation() -> None:
    assert len(S.SEQUENTIAL_ORDER) == len(S.STRATA)
    assert set(S.SEQUENTIAL_ORDER) == set(S.STRATA)


def test_sizer_keeps_peak_above_watchdog_floor() -> None:
    """The RAM-based worker sizer must never leave free-RAM-at-peak below the watchdog floor,
    at any free-RAM level (the floor guard from the 2026-08-19 sequential redesign)."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    path = repo / "experiments" / "spectre" / "restock3d_v2_collect.py"
    spec = importlib.util.spec_from_file_location("_rc_collect", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mem_floor = 6.0
    for avail in (12.0, 19.0, 40.0, 55.0):
        for stratum in S.STRATA:
            w = mod._sized_workers([stratum], 32, avail, mem_floor)
            assert w >= 1
            free_at_peak = avail - w * S.per_worker_gb(stratum)
            assert free_at_peak >= mem_floor, (stratum, avail, w, free_at_peak)
        # 2x2 at plentiful RAM is CPU-bound (0.85*32=27), not RAM-bound.
    assert mod._sized_workers([0], 32, 55.0, mem_floor) == 27
