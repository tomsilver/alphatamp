"""Restock3D **v2** Stage-0 de-risk gate (continuous-packing variant).

Drives the REAL v2 front-grasp pick + **section place controllers** through the four core cases on the
single multi-section shelf and renders each to an mp4 for manual review. Unlike v1 (discrete regions),
v2 has two place operators — ``place_tall`` / ``place_short`` — realised by binding the place
controller to a shelf SECTION and sampling x uniformly across that section's continuous band:

  1. small cube  -> place_tall  (bottom section) : expect SUCCESS
  2. small cube  -> place_short (top    section) : expect SUCCESS
  3. tall  block -> place_tall  (bottom section) : expect SUCCESS
  4. tall  block -> place_short (top    section) : expect FAILURE (F3 - upright block collides ceiling)

Every case uses the unified front-grasp pick + translate-only place (cubes land upright, not tilted).
Feasibility is decided by real PyBullet collision, never a toy gate.

Run (from the repo root, venv active)::

    python experiments/spectre/restock3d_v2_stage0.py            # all four cases -> demos/stage0_v2/*.mp4
    python experiments/spectre/restock3d_v2_stage0.py --cases 3,4

Prints a PASS/FAIL line per case; the gate passes iff cases 1-3 succeed and case 4 fails.
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

import imageio.v2 as iio
import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from relational_structs import Object

from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
    RestockFrontPickController,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller_v2 import (
    SectionFrontPlaceController,
)
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import RegionInfo
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    compute_section_infos,
)

_OUT_DIR = pathlib.Path(
    "src/alphatamp/approaches/spectre/envs/restock3d/demos/stage0_v2"
)
_MAX_STEPS = 900
_ATTEMPTS = 6


def _micro_scene(config: Restock3DEnvConfig):
    """A 1-cube + 1-block floor scene with the two continuous section bands."""
    specs = [
        ("cube_goal1", config.small_half, (0.1, 0.55, 0.1, 1.0)),
        ("block_goal1", config.tall_half, (0.65, 0.2, 0.2, 1.0)),
    ]

    def pose_fn(seed: int) -> dict[str, tuple[float, float]]:
        del seed
        return {"cube_goal1": (0.45, 0.05), "block_goal1": (0.6, 0.12)}

    sections = compute_section_infos(config)
    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, sections, config=config, allow_state_access=True
    )
    return sim, sections


def _render(sim) -> np.ndarray:
    kw = sim.config.get_camera_kwargs()
    return capture_image(sim.physics_client_id, image_width=640, image_height=480, **kw)


def _transition(sim, x, u):
    sim.set_state(x)
    obs, _, _, _, _ = sim.step(u)
    return obs.copy()


def _rollout(sim, controller, x, frames: list) -> object:
    cur = x
    for _ in range(_MAX_STEPS):
        if controller.terminated():
            break
        u = controller.step()
        cur = _transition(sim, cur, u)
        controller.observe(cur)
        frames.append(_render(sim))
    return cur


def _get(state, name: str) -> Object:
    return state.get_object_from_name(name)


def _resting_in_section(state, name: str, half_z: float, info: RegionInfo) -> bool:
    pose = state.get_object_pose(name)
    x, y, z = pose.position
    rest_z = z - half_z
    return (
        abs(rest_z - info.surface_z) < 0.05
        and abs(x - info.center_xy[0]) <= info.half_xy[0] + 0.06
        and abs(y - info.center_xy[1]) <= info.half_xy[1] + 0.06
    )


def _ceiling_collision_slide(
    sim, info: RegionInfo, block_name: str, frames: list, n: int = 30
) -> bool:
    """Show WHY case 4 fails as a SMOOTH motion: slide the upright block from the cell
    opening into the short section at rest height; its top jams against the ceiling
    board before it can enter."""
    pcid = sim.physics_client_id
    bid = sim._object_name_to_pybullet_id(block_name)
    half_z = sim._get_half_extents(block_name)[2]
    cx, cy = info.center_xy
    z = info.surface_z + half_z + 0.003
    start_y = cy - 0.45
    hit = False
    for t in np.linspace(start_y, cy, n):
        set_pose(bid, Pose((cx, float(t), z), (0, 0, 0, 1)), pcid)
        frames.append(_render(sim))
        if any(
            check_body_collisions(bid, b, pcid, distance_threshold=1e-3)
            for b in sim.shelf_board_ids()
        ):
            hit = True
            for _ in range(15):
                frames.append(_render(sim))
            break
    return hit


def run_case(case: int, target: str, section: str, expect_success: bool):
    """Run one case; returns (frames, passed, note)."""
    config = Restock3DEnvConfig()
    sim, sections = _micro_scene(config)
    info = sections[section]
    frames: list = []
    note = ""

    for attempt in range(_ATTEMPTS):
        x0, _ = sim.reset(seed=0)
        frames = [_render(sim)]
        robot = _get(x0, "robot")
        tgt = _get(x0, target)
        half_z = sim._get_half_extents(target)[2]

        pick = RestockFrontPickController([robot, tgt], sim)
        rng = np.random.default_rng(attempt)
        try:
            pick.reset(x0, pick.sample_parameters(x0, rng))
            cur = _rollout(sim, pick, x0, frames)
        except TrajectorySamplingFailure as e:
            note = f"pick failed: {e}"
            continue
        if cur.grasped_object != target:
            note = "pick did not grasp"
            continue

        place = SectionFrontPlaceController([robot, tgt], sim, info)
        try:
            place.reset(cur, place.sample_parameters(cur, rng))
            cur = _rollout(sim, place, cur, frames)
        except TrajectorySamplingFailure as e:
            if not expect_success:  # case 4: infeasible place is the expected outcome
                overlap = _ceiling_collision_slide(sim, info, target, frames)
                return frames, overlap, f"place raised (F3); ceiling overlap={overlap}"
            note = f"place failed: {e}"
            continue

        placed = _resting_in_section(cur, target, half_z, info)
        if expect_success and placed:
            return frames, True, "placed on section"
        if not expect_success and not placed:
            _ceiling_collision_slide(sim, info, target, frames)
            return frames, True, "block did not rest in short section"
        note = f"placed={placed}, expected_success={expect_success}"
        if not expect_success:
            _ceiling_collision_slide(sim, info, target, frames)
            return (
                frames,
                False,
                "block WRONGLY fit the short section (F3 did not bite)",
            )

    return frames, False, note or "exhausted attempts"


_CASES = {
    1: ("cube_goal1", "section_0", True, "cube_place_tall"),
    2: ("cube_goal1", "section_1", True, "cube_place_short"),
    3: ("block_goal1", "section_0", True, "block_place_tall"),
    4: ("block_goal1", "section_1", False, "block_place_short_F3"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="1,2,3,4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    which = [int(c) for c in args.cases.split(",")]
    results = {}
    for c in which:
        target, section, expect, label = _CASES[c]
        print(f"[case {c}] {label}: running ...", flush=True)
        frames, passed, note = run_case(c, target, section, expect)
        path = _OUT_DIR / f"case{c}_{label}.mp4"
        iio.mimsave(path, frames, fps=args.fps, macro_block_size=16)
        exp = "SUCCESS" if expect else "FAILURE (F3)"
        got = "PASS" if passed else "FAIL"
        results[c] = passed
        print(
            f"[case {c}] {label}: expect={exp}  ->  {got}  ({note})  wrote {path}",
            flush=True,
        )

    ok = all(results.values())
    print("\n==== STAGE-0 v2 GATE:", "PASS" if ok else "FAIL", "====")
    for c in which:
        print(f"  case {c} ({_CASES[c][3]}): {'PASS' if results[c] else 'FAIL'}")


if __name__ == "__main__":
    main()
