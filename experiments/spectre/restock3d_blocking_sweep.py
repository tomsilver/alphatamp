"""Restock3D Gate-1 F1 blocking calibration.

Sweeps a single clutter block's placement (center-to-center gap x direction) next to a goal object
and measures, via the shared ``grasp_blockers`` probe, which recipes yield a CLEAN F1:

  * the goal's grasp is *reachable but obstructed* -- ``grasp_blockers(goal)`` NAMES the clutter
    (a collision-scan F1 with a named culprit, NOT an unreachable-IK F1 whose culprits are empty),
  * the clutter is independently pickable -- ``grasp_blockers(clutter)`` does NOT name the goal
    (no 2-cycle / deadlock),
  * (reported) the base can still primary-plan to the goal's pick standoff (else it falls back).

Coverage/waste (Gate 6) needs NAMED F1 culprits, so we want reachable-but-blocked, not IK-unreachable.

Run (repo root, venv active)::

    python experiments/spectre/restock3d_blocking_sweep.py
    python experiments/spectre/restock3d_blocking_sweep.py --gaps 0.05,0.07,0.09 --goals cube,block

Prints a per-(goal,dir,gap) table and a recommended (gap, direction) per goal type.
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). ----------
import glob
import os
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
pathlib.Path(_B).mkdir(parents=True, exist_ok=True)
for _a, (_sd, _pt) in {
    "liblapack.a": ("lapack", "liblapack.so.3*"),
    "libblas.a": ("blas", "libblas.so.3*"),
}.items():
    _lk = pathlib.Path(_B) / _a
    if not (_lk.exists() or _lk.is_symlink()):
        _cs = sorted(
            glob.glob(f"/usr/lib/x86_64-linux-gnu/{_sd}/{_pt}")
            + glob.glob(f"/usr/lib/x86_64-linux-gnu/{_pt}")
        )
        _r = next((c for c in _cs if os.path.isfile(c)), None)
        if _r:
            _lk.symlink_to(_r)

import argparse

import numpy as np
from kinder_models.kinematic3d.utils import get_target_robot_pose_from_parameters
from pybullet_helpers.geometry import Pose, set_pose

from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    grasp_blockers,
)
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
    RestockFrontPickController,
    RestockPickController,
    _base_nav_collision_ids,
    get_base_plan,
)
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
    RegionInfo,
    section_surfaces,
)

_GOAL = "goal1"
_CLUTTER = "clutter1"
#: A goal on the floor, reachable by the arm from a front (-y) standoff.
_GOAL_XY = (0.45, 0.25)
#: Directions to place clutter relative to the goal; -y (the base-approach side) is excluded.
_DIRS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0)}


def _scene(config: Restock3DEnvConfig, goal_type: str):
    goal_half = config.tall_half if goal_type == "block" else config.small_half
    specs = [
        (_GOAL, goal_half, (0.1, 0.55, 0.1, 1.0)),
        (_CLUTTER, config.clutter_half, (0.9, 0.2, 0.2, 1.0)),
    ]

    def pose_fn(seed: int):
        del seed
        return {
            _GOAL: _GOAL_XY,
            _CLUTTER: (0.9, -0.35),
        }  # clutter parked; moved per recipe

    surfaces = section_surfaces(config)
    sx = config.shelf_pose.position[0]
    front_y = config.shelf_pose.position[1] - config.region_front_offset
    half_xy = (config.region_half_x, config.region_half_y)
    regions = {
        "region_tall": RegionInfo(
            "region_tall", 0, (sx, front_y), half_xy, surfaces[0][1], surfaces[0][0]
        ),
        "region_short": RegionInfo(
            "region_short", 1, (sx, front_y), half_xy, surfaces[1][1], surfaces[1][0]
        ),
    }
    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, regions, config=config, allow_state_access=True
    )
    return sim


def _base_reaches(sim, state, target_name: str, use_front: bool) -> bool:
    """Does the base PRIMARY-plan (no fallback) to the target's pick standoff succeed with the
    clutter in the collision set? (False = boxed -> deployment falls back to a straight path.)
    """
    robot = state.get_object_from_name("robot")
    tgt = state.get_object_from_name(target_name)
    ctrl = (
        RestockFrontPickController([robot, tgt], sim)
        if use_front
        else RestockPickController([robot, tgt], sim)
    )
    params = ctrl.sample_parameters(state, np.random.default_rng(0))
    target_se2 = state.get_object_pose(target_name).to_se2()
    standoff = get_target_robot_pose_from_parameters(target_se2, params[0], params[1])
    sim.set_state(state)
    nav = _base_nav_collision_ids(sim, state, frozenset({target_name}))
    plan = get_base_plan(sim, standoff, nav, None, None, allow_fallback=False)
    sim.set_state(state)
    return plan is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaps", type=str, default="0.05,0.06,0.07,0.08,0.09,0.10,0.12"
    )
    parser.add_argument("--goals", type=str, default="cube,block")
    args = parser.parse_args()
    gaps = [float(g) for g in args.gaps.split(",")]
    goals = args.goals.split(",")

    config = Restock3DEnvConfig()
    print(
        f"goal@{_GOAL_XY}  clutter_half={config.clutter_half}  gaps={gaps}\n",
        flush=True,
    )

    recommendations = {}
    for goal_type in goals:
        use_front = goal_type == "block"
        sim = _scene(config, goal_type)
        x0, _ = sim.reset(seed=0)
        clutter_id = sim._object_name_to_pybullet_id(_CLUTTER)
        clutter_half_z = sim._get_half_extents(_CLUTTER)[2]
        gx, gy = _GOAL_XY

        # Baseline: goal grasp reachable with clutter parked far away?
        gb0, reach0 = grasp_blockers(sim, _GOAL, sim.get_state())
        print(
            f"== goal={goal_type} ({'front' if use_front else 'top-down'} grasp)  "
            f"baseline reachable={reach0} blockers={gb0} ==",
            flush=True,
        )
        clean = []
        for dname, (dx, dy) in _DIRS.items():
            for gap in gaps:
                set_pose(
                    clutter_id,
                    Pose((gx + dx * gap, gy + dy * gap, clutter_half_z)),
                    sim.physics_client_id,
                )
                st = sim.get_state()
                gb_goal, reach_goal = grasp_blockers(sim, _GOAL, st)
                gb_clutter, reach_clutter = grasp_blockers(sim, _CLUTTER, st)
                blocks = _CLUTTER in gb_goal
                cycle = _GOAL in gb_clutter
                is_clean = reach_goal and blocks and not cycle
                base_ok = _base_reaches(sim, st, _GOAL, use_front)
                tag = (
                    "CLEAN-F1"
                    if is_clean
                    else (
                        "ik-unreach"
                        if (blocks and not reach_goal)
                        else "cycle" if cycle else "no-block"
                    )
                )
                print(
                    f"  dir={dname} gap={gap:.2f}: goal_reach={int(reach_goal)} "
                    f"blocks={int(blocks)} clutter_pickable={int(not cycle)} "
                    f"base_primary_reach={int(base_ok)}  -> {tag}",
                    flush=True,
                )
                if is_clean:
                    clean.append((dname, gap, base_ok))
        # Prefer a mid-range gap (robust) that is clean; among clean, one where base still reaches.
        if clean:
            clean_base = [c for c in clean if c[2]] or clean
            # pick the median gap among clean-with-base to be robust to jitter
            chosen = sorted(clean_base, key=lambda c: c[1])[len(clean_base) // 2]
            recommendations[goal_type] = chosen
        print("", flush=True)

    print("==== GATE-1 RECOMMENDATION ====", flush=True)
    for goal_type in goals:
        rec = recommendations.get(goal_type)
        if rec:
            print(
                f"  {goal_type}: dir={rec[0]} gap={rec[1]:.2f} m "
                f"(base primary-reaches standoff={bool(rec[2])})",
                flush=True,
            )
        else:
            print(
                f"  {goal_type}: NO clean-F1 recipe in range "
                f"(expected for 'block': a front grasp is not obstructed by side/back clutter, and "
                f"close clutter is itself blocked by the tall block -> a cycle; F1 targets CUBES)",
                flush=True,
            )
    # F1 targets CUBE goals (small floor cubes are what get stored); a clean cube recipe is the gate.
    ok = "cube" in recommendations
    print(
        f"\n==== GATE-1: {'PASS (cube F1 recipe found)' if ok else 'FAIL'} ====",
        flush=True,
    )


if __name__ == "__main__":
    main()
