"""Restock3D **v2** oracle certification tests (slow — real PyBullet motion planning).

Certifies the v2 oracle skeleton (continuous packing) and checks the two place operators
on a tall block by real collision: ``place_tall`` fits the bottom section,
``place_short`` collides the short section's ceiling board (F3) and never stores.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("stratum", [0, 2])
def test_oracle_v2_certifies(stratum: int) -> None:
    pytest.importorskip("kinder")
    from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import certify_stratum

    results = certify_stratum(stratum, 1)
    assert results[0].certified_feasible, results[0].note


@pytest.mark.slow
def test_place_short_of_tall_block_is_f3() -> None:
    """place_tall(block) fits the bottom section; place_short(block) is rejected
    (F3)."""
    pytest.importorskip("kinder")
    from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import (
        build_v2_bundle,
        refine_skeleton_v2,
    )

    bundle = build_v2_bundle(2)  # includes block_goal1

    x0, _ = bundle.sim.reset(seed=0)
    ok_tall, *_ = refine_skeleton_v2(bundle, x0, [("block_goal1", "section_0")], seed=0)

    x0, _ = bundle.sim.reset(seed=0)
    ok_short, *_ = refine_skeleton_v2(
        bundle, x0, [("block_goal1", "section_1")], seed=0
    )

    assert ok_tall, "place_tall(block) should fit the tall section"
    assert not ok_short, "place_short(block) should collide the short ceiling (F3)"
