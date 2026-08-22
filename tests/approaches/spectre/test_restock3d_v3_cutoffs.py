"""Restock3D-v3 cutoffs-in-sim regression: pins the calibrated arm-insertion height cutoffs
(short <= 0.12 m, tall <= 0.17 m) at the re-balanced partition (0.27, 0.22) in the REAL kinematic
env, so a future gripper/controller change can't silently move them.

Slow (real motion planning, retried against BiRRT flakiness). A block at the cutoff must place in
its section; a block clearly above it must not (the arm can't insert it under the board, even
though the block itself fits under the board — the exact mismatch Phase 3's F3-parity attributes).
"""

from __future__ import annotations

import numpy as np
import pytest
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.inverse_kinematics import InverseKinematicsError

from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
    RestockFrontPickController,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller_v3 import (
    LeftToRightSectionPlaceController,
)
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    compute_section_infos,
)

_MAX_STEPS = 900
_SECTION_KEY = {"tall": "section_0", "short": "section_1"}


def _rollout(sim, controller, x):
    cur = x
    for _ in range(_MAX_STEPS):
        if controller.terminated():
            break
        u = controller.step()
        sim.set_state(cur)
        obs, _, _, _, _ = sim.step(u)
        cur = obs.copy()
        controller.observe(cur)
    return cur


def _resting_in_section(state, name, half_z, info) -> bool:
    x, _y, z = state.get_object_pose(name).position
    return (
        abs((z - half_z) - info.surface_z) < 0.05
        and abs(x - info.center_xy[0]) <= info.half_xy[0] + 0.06
    )


def _places(full_h: float, section: str, samples: int = 12) -> bool:
    cfg = Restock3DEnvConfig(section_clearances=F.SECTION_CLEARANCES)
    secs = compute_section_infos(cfg)
    info = secs[_SECTION_KEY[section]]
    name = "blk"
    specs = [(name, (0.025, 0.025, full_h / 2.0), (0.6, 0.2, 0.2, 1.0))]

    def pose_fn(seed):
        del seed
        return {name: (0.5, 0.12)}

    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, secs, config=cfg, allow_state_access=True
    )
    half_z = full_h / 2.0
    for a in range(samples):
        x0, _ = sim.reset(seed=0)
        robot = x0.get_object_from_name("robot")
        tgt = x0.get_object_from_name(name)
        rng = np.random.default_rng(a)
        pick = RestockFrontPickController([robot, tgt], sim)
        try:
            pick.reset(x0, pick.sample_parameters(x0, rng))
            cur = _rollout(sim, pick, x0)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if cur.grasped_object != name:
            continue
        place = LeftToRightSectionPlaceController([robot, tgt], sim, info)
        try:
            place.reset(cur, place.sample_parameters(cur, rng))
            cur = _rollout(sim, place, cur)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if _resting_in_section(cur, name, half_z, info):
            return True
    return False


@pytest.mark.slow
@pytest.mark.parametrize(
    "full_h,section,expect",
    [
        (0.12, "short", True),  # short cutoff — must place
        (0.18, "short", False),  # clearly above short cutoff — arm can't insert
        (0.17, "tall", True),  # tall cutoff — must place
        (0.23, "tall", False),  # clearly above tall cutoff — arm can't insert
    ],
)
def test_v3_cutoffs_in_sim(full_h, section, expect):
    assert _places(full_h, section) is expect
