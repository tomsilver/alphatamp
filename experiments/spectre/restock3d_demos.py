"""Render per-stratum Restock3D demos: the robot executes a successful plan (relocates
any grasp-blocking clutter, then stores every goal object into a feasible region).

For each (stratum, seed) it first clears the F1 clutter -- any movable whose body
obstructs a goal's grasp (via ``grasp_blockers``) is picked and relocated to a floor
buffer spot (``BufferPlaceController``) -- then assigns each goal a distinct region
(blocks -> tall section, cubes -> short section, spilling to tall when the short
regions run out) -- a feasible-by-construction plan -- and drives the real front-grasp
/ top-down controllers, retrying each pick/place from the current state (keeping
already-placed objects) and committing only a successful attempt's frames, so the video
shows one clean successful episode. Regions/buffers are metadata (never drawn). ``main``
searches seeds until a fully-successful episode and writes ONE
``demos/demo_r{stratum}.mp4`` per stratum (relocation makes r1 -- the one cluttered
stratum -- a genuine relocate->store episode).

Run from the repo root (venv active)::

    python experiments/spectre/restock3d_demos.py --strata 0,1,2,3 --seeds 0,1,2,3
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
from relational_structs import Object

from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    grasp_blockers,
)
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    stratum_env_args,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
    BufferPlaceController,
    RegionAdaptivePlaceController,
    RegionType,
    RestockAdaptivePickController,
    in_buffer_zone,
)
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import RegionInfo

_OUT = pathlib.Path("src/alphatamp/approaches/spectre/envs/restock3d/demos")
_MAX_STEPS = 900
_STORE_ATTEMPTS = 18  # placements are ~1/6 reliable (BiRRT flakiness); retry per store


def _render(sim) -> np.ndarray:
    return capture_image(
        sim.physics_client_id,
        image_width=640,
        image_height=480,
        **sim.config.get_camera_kwargs(),
    )


def _drive(sim, controller, x, frames=None):
    """Drive an already-reset controller to termination (correct order: step() THEN
    set_state(cur)+step).

    Renders into ``frames`` when given.
    """
    cur = x
    for _ in range(_MAX_STEPS):
        if controller.terminated():
            break
        u = controller.step()
        sim.set_state(cur)
        cur, *_ = sim.step(u)
        cur = cur.copy()
        controller.observe(cur)
        if frames is not None:
            frames.append(_render(sim))
    return cur


def _try_store(sim, x, obj, region, region_infos, seed, frames=None):
    """One store attempt at a fixed ``seed`` (pick then place, ONE rng advanced across
    both, as the Stage-0 gate does).

    Returns the state if it stored the object, else None. A fixed seed makes the attempt
    reproducible, so a successful seed can be re-run WITH rendering (``frames`` not
    None).
    """
    rng = np.random.default_rng(seed)
    robot = x.get_object_from_name("robot")
    tgt = x.get_object_from_name(obj)
    pick = RestockAdaptivePickController([robot, tgt], sim)
    pick.reset(x, pick.sample_parameters(x, rng))
    cur = _drive(sim, pick, x, frames)
    if cur.grasped_object != obj:
        return None
    place = RegionAdaptivePlaceController(
        [robot, tgt, Object(region, RegionType)], sim, region_infos
    )
    place.reset(cur, place.sample_parameters(cur, rng))
    cur = _drive(sim, place, cur, frames)
    return cur if _stored(cur, obj, region_infos[region]) else None


def _blocking_clutter(sim, x, goal_names) -> list[str]:
    """The clutter names whose bodies obstruct some goal's grasp (F1), via
    ``grasp_blockers``.

    These must be relocated to a buffer before the blocked goals can be picked. Sorted
    for a deterministic relocation order.
    """
    blocking: set[str] = set()
    for goal in goal_names:
        blk, _reachable = grasp_blockers(sim, goal, x)
        blocking.update(b for b in blk if b.startswith("clutter"))
    return sorted(blocking)


def _try_relocate(sim, x, clutter, seed, frames=None):
    """One relocate attempt at a fixed ``seed``: Pick(clutter, top-down) ->
    PlaceBuffer(clutter), ONE rng advanced across both (as the store attempt does).

    Returns the state if the clutter now rests in the buffer zone (released), else None
    -- so a successful seed can be re-run WITH rendering, matching ``_try_store``'s
    search-then-render.
    """
    rng = np.random.default_rng(seed)
    robot = x.get_object_from_name("robot")
    tgt = x.get_object_from_name(clutter)
    pick = RestockAdaptivePickController([robot, tgt], sim)
    pick.reset(x, pick.sample_parameters(x, rng))
    cur = _drive(sim, pick, x, frames)
    if cur.grasped_object != clutter:
        return None
    place = BufferPlaceController([robot, tgt], sim)
    place.reset(cur, place.sample_parameters(cur, rng))
    cur = _drive(sim, place, cur, frames)
    if cur.grasped_object == clutter:
        return None  # never released onto the buffer
    px, py, _ = cur.get_object_pose(clutter).position
    return cur if in_buffer_zone(px, py) else None


def _stored(state, name: str, info: RegionInfo) -> bool:
    pose = state.get_object_pose(name)
    x, y, z = pose.position
    half_z = 0.12 if name.startswith("block_goal") else 0.025
    return (
        abs((z - half_z) - info.surface_z) < 0.05
        and abs(x - info.center_xy[0]) <= info.half_xy[0] + 0.06
        and abs(y - info.center_xy[1]) <= info.half_xy[1] + 0.06
    )


def _central(region_infos, names, n, center_x):
    """The ``n`` regions from ``names`` closest to the shelf centre, returned left-to-
    right.

    The shelf *edges* are near the arm's reach limit and place unreliably, so a demo
    fills from the centre out.
    """
    ordered = sorted(names, key=lambda r: abs(region_infos[r].center_xy[0] - center_x))
    chosen = ordered[:n]
    return sorted(chosen, key=lambda r: region_infos[r].center_xy[0])


def _assignment(specs, region_infos, center_x):
    """Distinct, centre-first region per object: blocks -> tall (central); cubes ->
    short (central), overflow to tall.

    Central selection keeps placements inside the reliable reach zone.
    """
    tall = [n for n, i in region_infos.items() if i.shelf == 0]
    short = [n for n, i in region_infos.items() if i.shelf == 1]
    cubes = [n for n, _, _ in specs if n.startswith("cube_goal")]
    blocks = [n for n, _, _ in specs if n.startswith("block_goal")]
    pairs = []
    for b, reg in zip(blocks, _central(region_infos, tall, len(blocks), center_x)):
        pairs.append((b, reg))
    used = {r for _, r in pairs}
    n_short = min(len(cubes), len(short))
    for cube, reg in zip(
        cubes[:n_short], _central(region_infos, short, n_short, center_x)
    ):
        pairs.append((cube, reg))
        used.add(reg)
    overflow = _central(
        region_infos, [t for t in tall if t not in used], len(cubes) - n_short, center_x
    )
    for cube, reg in zip(cubes[n_short:], overflow):
        pairs.append((cube, reg))
    return pairs


def render_stratum(stratum: int, seed: int):
    """Render a full successful episode for ``(stratum, seed)``: relocate any grasp-
    blocking clutter to a buffer, then store every goal into a distinct feasible region.

    Returns ``(frames, relocated, n_block, stored, total)``. A fully-successful episode
    has ``relocated == n_block`` and ``stored == total``; a partial episode stops at the
    first unstorable/unrelocatable object (so the video always shows real, successful
    sub-plans only).
    """
    specs, pose_fn, region_infos, config = stratum_env_args(stratum)
    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, region_infos, config=config, allow_state_access=True
    )
    x, _ = sim.reset(seed=seed)
    frames = [_render(sim)]
    goal_names = [n for n, _, _ in specs if n.startswith(("cube_goal", "block_goal"))]

    # Relocation phase: clear the F1 clutter obstructing any goal's grasp (search a
    # working param seed WITHOUT rendering, then re-run WITH rendering, as store does).
    blockers = _blocking_clutter(sim, x, goal_names)
    relocated = 0
    for j, clutter in enumerate(blockers):
        succ_seed = None
        for attempt in range(_STORE_ATTEMPTS):
            aseed = seed * 100000 + 90000 + j * 1000 + attempt
            try:
                if _try_relocate(sim, x, clutter, aseed) is not None:
                    succ_seed = aseed
                    break
            except TrajectorySamplingFailure:
                continue
        if succ_seed is None:
            return frames, relocated, len(blockers), 0, len(goal_names)
        x = _try_relocate(sim, x, clutter, succ_seed, frames)
        relocated += 1
        print(f"    r{stratum} seed={seed}: relocated {clutter}", flush=True)

    # Store phase, SOUTH-TO-NORTH (nearest-first). In the fully-lateral layout the front grasp
    # reaches north over anything closer than the target, so a back object is blocked until the
    # nearer ones are cleared -- picking by ascending floor-y keeps every reach clear (mirrors the
    # oracle; decisions/07 2026-08-16).
    plan = _assignment(specs, region_infos, config.shelf_pose.position[0])
    plan.sort(key=lambda pr: x.get_object_pose(pr[0]).position[1])
    stored = 0
    for i, (obj_name, region_name) in enumerate(plan):
        # Search for a successful param seed WITHOUT rendering ...
        succ_seed = None
        for attempt in range(_STORE_ATTEMPTS):
            aseed = seed * 100000 + i * 1000 + attempt
            try:
                if (
                    _try_store(sim, x, obj_name, region_name, region_infos, aseed)
                    is not None
                ):
                    succ_seed = aseed
                    break
            except TrajectorySamplingFailure:
                continue
        if succ_seed is None:
            break  # cannot store this object -> stop (partial plan)
        # ... then re-run just that seed WITH rendering (reproducible).
        x = _try_store(sim, x, obj_name, region_name, region_infos, succ_seed, frames)
        stored += 1
        print(
            f"    r{stratum} seed={seed}: stored {obj_name} -> {region_name}",
            flush=True,
        )
    return frames, relocated, len(blockers), stored, len(plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    _OUT.mkdir(parents=True, exist_ok=True)
    for stratum in [int(s) for s in args.strata.split(",")]:
        out = _OUT / f"demo_r{stratum}.mp4"
        best = None  # (frames, relocated, n_block, stored, total), most-progress-first
        for seed in seeds:
            frames, rel, n_block, stored, total = render_stratum(stratum, seed)
            solved = rel == n_block and stored == total
            print(
                f"[restock3d_demos] r{stratum} seed={seed}: relocated {rel}/{n_block}"
                f" stored {stored}/{total} ({'SOLVED' if solved else 'partial'})",
                flush=True,
            )
            if best is None or (rel + stored) > (best[1] + best[3]):
                best = (frames, rel, n_block, stored, total)
            if solved:
                break
        assert best is not None
        iio.mimsave(out, best[0], fps=20, macro_block_size=16)
        solved = best[1] == best[2] and best[3] == best[4]
        print(
            f"[restock3d_demos] r{stratum}: wrote {out}"
            f" ({'SOLVED' if solved else 'partial(best)'}:"
            f" relocated {best[1]}/{best[2]} stored {best[3]}/{best[4]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
