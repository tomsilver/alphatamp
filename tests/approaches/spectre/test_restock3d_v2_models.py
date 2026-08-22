"""Tests for the **Restock3D v2** symbolic contract (continuous packing).

v2 drops ``InRegion`` and the ``?region`` argument: two place operators (``place_tall``
/ ``place_short``) with identical abstract effects (``add {HandEmpty, Stored}``), and a
purely geometric ``Stored`` (section membership). Building the sim spins up a headless
PyBullet client (fast, like the oracle skeleton test), so these are not marked slow.
"""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.envs.restock3d import models_v2 as M


def test_predicate_set_has_no_inregion() -> None:
    names = {p.name for p in (M.HandEmpty, M.Holding, M.OnFloor, M.Stored, M.OnBuffer)}
    assert names == {"HandEmpty", "Holding", "OnFloor", "Stored", "OnBuffer"}
    assert not hasattr(M, "InRegion")  # dropped in v2
    assert not hasattr(M, "RegionType")  # operators carry no region argument


def test_abstractor_at_reset_floor_only() -> None:
    pytest.importorskip("kinder")
    from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import build_v2_bundle

    bundle = build_v2_bundle(2)  # 3 cubes + 1 tall block
    x0, _ = bundle.sim.reset(seed=0)
    atoms = bundle.abstractor.state_abstractor(x0).atoms
    names = {a.predicate.name for a in atoms}
    # At reset every goal is on the floor, hand empty; nothing stored, no InRegion atoms exist.
    assert names == {"HandEmpty", "OnFloor"}
    on_floor = {a.objects[0].name for a in atoms if a.predicate.name == "OnFloor"}
    assert on_floor == set(bundle.goal_names)


def test_two_place_operators_identical_stored_effects() -> None:
    pytest.importorskip("kinder")
    import kinder

    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.restock3d.models_v2 import (
        create_restock3d_v2_models,
    )

    register_extra_envs()
    env = kinder.make(
        "spectre/Restock3D-r0-v0"
    )  # low-level env shared with v1; spaces only
    try:
        bundle = create_restock3d_v2_models(
            env.observation_space, env.action_space, stratum=0
        )
        ops = {op.name: op for op in bundle.models.operators}
        assert set(ops) == {"pick", "place_tall", "place_short", "place_buffer"}
        for nm in ("place_tall", "place_short"):
            op = ops[nm]
            assert [v.type.name for v in op.parameters] == [
                "Kinematic3DRobot",
                "Kinematic3DCuboid",
            ]  # (robot, target) only -- no region
            add = {a.predicate.name for a in op.add_effects}
            assert add == {
                "HandEmpty",
                "Stored",
            }  # identical effects; section is refinement-only
            assert "InRegion" not in add
            assert {a.predicate.name for a in op.preconditions} == {"Holding"}
    finally:
        env.close()
