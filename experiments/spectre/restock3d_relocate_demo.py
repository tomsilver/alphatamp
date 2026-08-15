"""Restock3D Gate-2 relocate->store demo / diagnostic.

Drives the REAL controllers through the F1 relocation sequence on a micro-scene (1 cube
goal blocked by 1 +y clutter, 1 short region): Pick(clutter, top-down) ->
PlaceBuffer(clutter) -> Pick(cube, top-down) -> Place(cube, region). Reports PASS/FAIL
per step so a broken BufferPlaceController is isolated, and renders the sequence to
demos/stage0/relocate_store.mp4.

python experiments/spectre/restock3d_relocate_demo.py
"""

from __future__ import annotations

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

import imageio.v2 as iio
import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.camera import capture_image
from relational_structs import Object

from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
    BufferPlaceController,
    RegionAdaptivePlaceController,
    RegionType,
    RestockAdaptivePickController,
    in_buffer_zone,
)
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
    RegionInfo,
    section_surfaces,
)

_OUT = pathlib.Path(
    "src/alphatamp/approaches/spectre/envs/restock3d/demos/stage0/relocate_store.mp4"
)
_MAX_STEPS = 1200
_ATTEMPTS = 12


def _scene(config):
    """1 cube goal + 1 clutter (+y, 0.07 gap) + one short region."""
    specs = [
        ("cube_goal1", config.small_half, (0.1, 0.55, 0.1, 1.0)),
        ("clutter1", config.clutter_half, (0.9, 0.2, 0.2, 1.0)),
    ]

    def pose_fn(seed):
        del seed
        return {"cube_goal1": (0.45, 0.20), "clutter1": (0.45, 0.27)}  # +y 0.07

    surfaces = section_surfaces(config)
    sx = config.shelf_pose.position[0]
    fy = config.shelf_pose.position[1] - config.region_front_offset
    hxy = (config.region_half_x, config.region_half_y)
    regions = {
        "region_short": RegionInfo(
            "region_short", 1, (sx, fy), hxy, surfaces[1][1], surfaces[1][0]
        ),
        "region_tall": RegionInfo(
            "region_tall", 0, (sx, fy + 0.12), hxy, surfaces[0][1], surfaces[0][0]
        ),
    }
    return (
        ObjectCentricRestock3DEnv(
            specs, pose_fn, regions, config=config, allow_state_access=True
        ),
        regions,
    )


def _render(sim):
    return capture_image(
        sim.physics_client_id,
        image_width=640,
        image_height=480,
        **sim.config.get_camera_kwargs(),
    )


def _drive(sim, controller, x, frames):
    cur = x
    for _ in range(_MAX_STEPS):
        if controller.terminated():
            break
        u = controller.step()
        sim.set_state(cur)
        obs, _, _, _, _ = sim.step(u)
        cur = obs.copy()
        controller.observe(cur)
        frames.append(_render(sim))
    return cur


_STEPS = ["pick-clutter", "place-buffer", "pick-cube", "place-cube"]


def _one_step(sim, ctrl, cur, frames, rng, label, expect_grasp=None):
    """Sample + drive one controller with the shared per-attempt rng.

    Returns (state, ok).
    """
    try:
        ctrl.reset(cur, ctrl.sample_parameters(cur, rng))
    except TrajectorySamplingFailure:
        return cur, False
    try:
        cur = _drive(sim, ctrl, cur, frames)
    except TrajectorySamplingFailure:
        return cur, False
    if expect_grasp is not None and cur.grasped_object != expect_grasp:
        return cur, False
    return cur, True


def _attempt(sim, regions, seed):
    """One full relocate->store attempt from a fresh reset with a single shared rng (so
    a pick standoff that dooms the place is re-rolled as a whole, exactly like the
    refiner backtracks)."""
    x0, _ = sim.reset(
        seed=0
    )  # the scene is deterministic; only the sampling seed varies
    robot = x0.get_object_from_name("robot")
    clutter = x0.get_object_from_name("clutter1")
    cube = x0.get_object_from_name("cube_goal1")
    reg = Object("region_short", RegionType)
    rng = np.random.default_rng(seed)
    frames = [_render(sim)]
    results = {s: False for s in _STEPS}

    cur, ok = _one_step(
        sim,
        RestockAdaptivePickController([robot, clutter], sim),
        x0,
        frames,
        rng,
        "pick-clutter",
        expect_grasp="clutter1",
    )
    results["pick-clutter"] = ok
    if ok:
        cur, ok = _one_step(
            sim,
            BufferPlaceController([robot, clutter], sim),
            cur,
            frames,
            rng,
            "place-buffer",
        )
        if ok:
            p = cur.get_object_pose("clutter1").position
            ok = in_buffer_zone(p[0], p[1])
        results["place-buffer"] = ok
    if ok:
        cur, ok = _one_step(
            sim,
            RestockAdaptivePickController([robot, cube], sim),
            cur,
            frames,
            rng,
            "pick-cube",
            expect_grasp="cube_goal1",
        )
        results["pick-cube"] = ok
    if ok:
        cur, ok = _one_step(
            sim,
            RegionAdaptivePlaceController([robot, cube, reg], sim, regions),
            cur,
            frames,
            rng,
            "place-cube",
        )
        if ok:
            ok = cur.get_object_pose("cube_goal1").position[2] > 0.2
        results["place-cube"] = ok
    return frames, results, all(results.values())


def main():
    config = Restock3DEnvConfig()
    sim, regions = _scene(config)
    best_frames, best_results = None, None
    for seed in range(_ATTEMPTS):
        frames, results, ok_all = _attempt(sim, regions, seed)
        print(
            f"seed {seed}: "
            + " ".join(f"{s}={'Y' if results[s] else 'n'}" for s in _STEPS),
            flush=True,
        )
        if best_results is None or sum(results.values()) > sum(best_results.values()):
            best_frames, best_results = frames, results
        if ok_all:
            break

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(_OUT, best_frames, fps=20, macro_block_size=16)
    print(f"wrote {_OUT}", flush=True)
    ok_all = all(best_results.values())
    print(f"\n==== RELOCATE->STORE: {'PASS' if ok_all else 'FAIL'} ====", flush=True)
    for s in _STEPS:
        print(f"  {s}: {'PASS' if best_results.get(s) else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
