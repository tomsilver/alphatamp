"""Gate-2 tests: v2 continuous-section attribution (``place_tall``/``place_short``).

Drives ``_probe_place_v2`` directly on hand-built micro-scenes: F3 (a tall block probed
onto the short section overflows the ceiling, culprit-free) and F2 (a place onto a
section that already has a resident names the resident). Marked ``slow`` -- needs
PyBullet.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow


def _scene(specs, poses):
    pytest.importorskip("kinder")
    from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
        make_recording_sampler,
    )
    from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
        ObjectCentricRestock3DEnv,
        Restock3DEnvConfig,
    )
    from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
        compute_section_infos,
    )

    cfg = Restock3DEnvConfig()
    sections = compute_section_infos(cfg)  # keys: section_0 (tall), section_1 (short)
    sim = ObjectCentricRestock3DEnv(
        specs, lambda s: poses, sections, config=cfg, allow_state_access=True
    )
    sampler = make_recording_sampler(
        controller_generator=lambda a: None,
        transition_function=lambda x, u: x,
        state_abstractor=lambda x: None,
        max_trajectory_steps=1,
        sim=sim,
        region_infos=sections,
    )
    return sim, sections, sampler


def _op(name, *obj_names, state):
    """A v2 place/pick op: parameters are (robot, target) -- no region arg."""
    params = [state.get_object_from_name(n) for n in obj_names]
    return SimpleNamespace(name=name, parameters=params)


def test_f3_tall_block_into_short_section_v2() -> None:
    specs = [("block_goal1", (0.025, 0.025, 0.12), (0.6, 0.2, 0.2, 1.0))]
    sim, _, sampler = _scene(specs, {"block_goal1": (-0.5, 0.7)})
    try:
        state, _ = sim.reset(seed=0)
        culprits, family = sampler._probe_place_v2(
            state, _op("place_short", "robot", "block_goal1", state=state)
        )
        assert family == "F3" and culprits == ()  # too tall for the short section
        # The SAME block into the tall section fits -> not F3.
        _, fam_tall = sampler._probe_place_v2(
            state, _op("place_tall", "robot", "block_goal1", state=state)
        )
        assert fam_tall != "F3"
    finally:
        sim.close()


def test_f2_place_onto_section_resident_v2() -> None:
    from pybullet_helpers.geometry import Pose, set_pose

    specs = [
        ("cube_goal1", (0.025, 0.025, 0.025), (0.1, 0.5, 0.1, 1.0)),
        ("cube_goal2", (0.025, 0.025, 0.025), (0.1, 0.5, 0.1, 1.0)),
    ]
    sim, sections, sampler = _scene(
        specs, {"cube_goal1": (-0.5, 0.7), "cube_goal2": (-0.5, 0.9)}
    )
    try:
        sim.reset(seed=0)
        # Seat cube_goal1 as a resident of the tall section, then probe placing cube_goal2.
        info = sections["section_0"]
        set_pose(
            sim._object_name_to_pybullet_id("cube_goal1"),
            Pose((info.center_xy[0], info.center_xy[1], info.surface_z + 0.025)),
            sim.physics_client_id,
        )
        state = sim._get_obs()
        culprits, family = sampler._probe_place_v2(
            state, _op("place_tall", "robot", "cube_goal2", state=state)
        )
        assert family == "F2"
        assert "cube_goal1" in culprits
    finally:
        sim.close()
