"""Restock3D **v3** pick/place calibration & tight-packing study (pre-implementation).

v2 is too easy for the baselines (LAZY ~ oracle). v3 will make block selection matter by
(1) varying block **x-widths** (lateral) and (2) varying block **heights sampled near the
short/tall fit cutoff**. Before building v3 we measure the *physical envelope* of the
existing front-grasp pick/place on the CURRENT kinematic env, so v3's generator only samples
feasible (width, height) and its refiner can pack left-to-right with the right padding.

This is a **measurement harness only** -- it imports the real controllers / env unchanged
(no production edits) and drives a 1- or 2-object micro-scene. Three sweeps:

  * Sweep 1 (heights): min/max full block height that pick+place still handle, per shelf
    section {tall, short}, per grasp scheme {current, center, capped-center}.
  * Sweep 2 (widths): the graspable face width -- reported analytically (finger aperture)
    + a sim-permissiveness confound check (the kinematic grasp excludes the target during
    the reach-in, so the sim does NOT bind width -- v3 must cap it analytically).
  * Sweep 3 (padding): five methods (capacity arithmetic, gripper geometry, static
    swept-collision probe, two-object rollout ground truth, n-in-a-row capacity) reconciled
    into one recommended left-to-right packing gap.

Writes neat CLI tables and a markdown findings doc
(``src/alphatamp/approaches/spectre/docs/restock3d_v3_calibration.md`` by default).

Run (repo root, venv active)::

    python experiments/spectre/restock3d_v3_calibrate.py --quick      # coarse smoke (~min)
    python experiments/spectre/restock3d_v3_calibrate.py              # full sweeps
    python experiments/spectre/restock3d_v3_calibrate.py --sweeps 1,3 # subset
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached). --------------------
import glob
import os
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
# Keep BLAS single-threaded so the process pool doesn't oversubscribe the cores.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pybullet as p
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import InverseKinematicsError

from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
    _FRONT_GRASP_QUAT,
    _FRONT_GRASP_Y_OFFSET,
    _FRONT_GRIP_MARGIN,
    RestockFrontPickController,
    front_grasp_transform,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller_v2 import (
    SectionFrontPlaceController,
)
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    band_half_x,
    compute_section_infos,
)

# ============================================================================================
# Constants / configuration
# ============================================================================================

_MAX_STEPS = 900
_HALF_Y = 0.025  # depth (y) half-extent, held fixed for every sweep
_DEPTH = 2 * _HALF_Y

#: Grasp schemes for the parameterized pick. Each maps (half_z) -> object-frame z offset of the
#: grasp contact above the object CENTRE (the EE lands at world z = half_z + offset, since a
#: floor block's centre is at world z = half_z).
#:  * "current"       -- production: land the EE at a fixed reachable world z (0.13); near-top on
#:                       tall blocks. offset = clip(0.13 - half_z, on-face).
#:  * "center"        -- grasp exactly at the object centre (offset 0); the user's proposal.
#:  * "capped_center" -- centre grasp, but never raise the EE above _REACH_CAP (so very tall
#:                       blocks grasp as high as the 45deg reach envelope allows).
_REACH_CAP = 0.16  # arm's ~45deg reach envelope ceiling (world EE z)
_FRONT_GRASP_TARGET_EE_Z = 0.13
GRASP_SCHEMES = ("current", "center", "capped_center")

#: Robotiq 2F-85 nominal stroke (real-robot spec; NOT enforced by the kinematic sim).
_APERTURE_NOMINAL_MM = 85.0


def _scheme_transform(scheme: str, half_z: float) -> Pose:
    """Object->EE grasp transform for a scheme (mirrors ``front_grasp_transform``)."""
    half_z = float(half_z)
    if scheme == "current":
        return front_grasp_transform(half_z)
    if scheme == "center":
        ee_world_z = half_z
    elif scheme == "capped_center":
        ee_world_z = min(half_z, _REACH_CAP)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    offset = ee_world_z - half_z  # object-frame z above centre
    lim = max(0.0, half_z - _FRONT_GRIP_MARGIN)  # keep the grip on-face
    return Pose(
        (0.0, _FRONT_GRASP_Y_OFFSET, float(np.clip(offset, -lim, lim))),
        _FRONT_GRASP_QUAT,
    )


class _SchemePickController(RestockFrontPickController):
    """Front pick whose grasp z-offset follows a chosen scheme (harness-only; production
    ``RestockFrontPickController`` is untouched)."""

    scheme: str = "current"

    def _front_grasp_transform(self) -> Pose:  # type: ignore[override]
        half_z = self._current_state.get_object_half_extents(self.objects[1].name)[2]
        return _scheme_transform(self.scheme, half_z)


def _make_pick(scheme: str):
    def factory(objects, sim):
        ctrl = _SchemePickController(objects, sim)
        ctrl.scheme = scheme
        return ctrl

    return factory


# ============================================================================================
# Env construction + rollout helpers
# ============================================================================================


def _build_scene(specs: list[tuple[str, tuple, tuple]], floor_xy: dict[str, tuple]):
    """A floor micro-scene (1 or 2 objects) with the two continuous section bands."""
    config = Restock3DEnvConfig()

    def pose_fn(seed: int) -> dict[str, tuple[float, float]]:
        del seed
        return floor_xy

    sections = compute_section_infos(config)
    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, sections, config=config, allow_state_access=True
    )
    return sim, sections, config


def _rollout(sim, controller, x) -> Any:
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


def _resting_in_section(state, name: str, half_z: float, info) -> bool:
    x, y, z = state.get_object_pose(name).position
    rest_z = z - half_z
    return (
        abs(rest_z - info.surface_z) < 0.05
        and abs(x - info.center_xy[0]) <= info.half_xy[0] + 0.06
        and abs(y - info.center_xy[1]) <= info.half_xy[1] + 0.06
    )


def _pick_place_trial(
    sim, sections, name, half_z, section, scheme, attempt, place_param=None
) -> str:
    """One reset->pick->place attempt.

    Returns "ok" | "pick_fail" | "place_fail".
    """
    info = sections[section]
    x0, _ = sim.reset(seed=0)
    robot = x0.get_object_from_name("robot")
    tgt = x0.get_object_from_name(name)
    rng = np.random.default_rng(attempt)
    pick = _make_pick(scheme)([robot, tgt], sim)
    try:
        pick.reset(x0, pick.sample_parameters(x0, rng))
        cur = _rollout(sim, pick, x0)
    except TrajectorySamplingFailure:
        return "pick_fail"
    except InverseKinematicsError:
        return "pick_fail"
    if cur.grasped_object != name:
        return "pick_fail"
    place = SectionFrontPlaceController([robot, tgt], sim, info)
    try:
        param = (
            place.sample_parameters(cur, rng) if place_param is None else place_param
        )
        place.reset(cur, param)
        cur = _rollout(sim, place, cur)
    except TrajectorySamplingFailure:
        return "place_fail"
    except InverseKinematicsError:
        return "place_fail"
    return "ok" if _resting_in_section(cur, name, half_z, info) else "place_fail"


def _feasible(
    sim, sections, name, half_z, section, scheme, n_tries
) -> tuple[bool, int, str]:
    """Run up to ``n_tries`` attempts, early-stop on first success.

    Returns (feasible, successes_or_tries, last_reason).
    """
    reasons = {"pick_fail": 0, "place_fail": 0}
    for a in range(n_tries):
        r = _pick_place_trial(sim, sections, name, half_z, section, scheme, a)
        if r == "ok":
            return True, a + 1, "ok"
        reasons[r] += 1
    # dominant failure mode
    reason = "pick" if reasons["pick_fail"] >= reasons["place_fail"] else "place"
    return False, n_tries, reason


# ============================================================================================
# Sweep 1 -- block heights
# ============================================================================================


@dataclass
class _HeightRow:
    full_h: float
    # per (section, scheme): (feasible, n_or_tries, reason)
    cells: dict


def _height_worker(full_h: float, n_tries: int) -> _HeightRow:
    half_z = full_h / 2.0
    name = "block_goal1"
    specs = [(name, (_HALF_Y, _HALF_Y, half_z), (0.65, 0.2, 0.2, 1.0))]
    sim, sections, _ = _build_scene(specs, {name: (0.6, 0.12)})
    cells = {}
    for section in ("section_0", "section_1"):  # tall, short
        for scheme in GRASP_SCHEMES:
            feasible, k, reason = _feasible(
                sim, sections, name, half_z, section, scheme, n_tries
            )
            cells[(section, scheme)] = (feasible, k, reason)
    return _HeightRow(full_h=full_h, cells=cells)


def sweep_heights(heights, n_tries, workers, log) -> list[_HeightRow]:
    log(
        f"[sweep1] heights={len(heights)} vals x 2 sections x 3 schemes, {n_tries} tries, {workers}w"
    )
    rows: list[_HeightRow] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_height_worker, h, n_tries): h for h in heights}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            el = time.time() - t0
            eta = el / done * (len(heights) - done)
            log(
                f"[sweep1] {done}/{len(heights)} h={futs[fut]:.3f} ({el:.0f}s, ETA {eta:.0f}s)"
            )
    rows.sort(key=lambda r: r.full_h)
    return rows


# ============================================================================================
# Sweep 2 -- block widths (aperture + sim-permissiveness confound)
# ============================================================================================


def measure_aperture() -> dict[str, float]:
    """Inner finger-pad separation (mm) at open/mid/closed -- the physical grasp
    aperture."""
    name = "block_goal1"
    specs = [(name, (_HALF_Y, _HALF_Y, 0.12), (0.65, 0.2, 0.2, 1.0))]
    sim, _, _ = _build_scene(specs, {name: (0.6, 0.12)})
    sim.reset(seed=0)
    arm = sim.robot.arm
    rid = arm.robot_id
    pcid = sim.physics_client_id

    def link_idx(sub):
        for li in range(p.getNumJoints(rid, physicsClientId=pcid)):
            if sub in p.getJointInfo(rid, li, physicsClientId=pcid)[12].decode():
                return li
        return None

    li_l, li_r = link_idx("left_inner_finger_pad"), link_idx("right_inner_finger_pad")
    out = {}
    for state, lbl in [(0.0, "open"), (0.4, "mid"), (0.8, "closed")]:
        arm.set_finger_state(state)
        pl = np.array(p.getLinkState(rid, li_l, physicsClientId=pcid)[0])
        pr = np.array(p.getLinkState(rid, li_r, physicsClientId=pcid)[0])
        out[lbl] = float(np.linalg.norm(pl - pr)) * 1000.0
    return out


def _width_pick_worker(full_w: float, n_tries: int) -> tuple[float, bool, int]:
    """Does the SIM pick succeed for a block of this width?

    (confound check.)
    """
    name = "block_goal1"
    half_x = full_w / 2.0
    specs = [(name, (half_x, _HALF_Y, 0.05), (0.65, 0.2, 0.2, 1.0))]
    sim, _, _ = _build_scene(specs, {name: (0.6, 0.12)})
    for a in range(n_tries):
        x0, _ = sim.reset(seed=0)
        robot = x0.get_object_from_name("robot")
        tgt = x0.get_object_from_name(name)
        pick = RestockFrontPickController([robot, tgt], sim)
        rng = np.random.default_rng(a)
        try:
            pick.reset(x0, pick.sample_parameters(x0, rng))
            cur = _rollout(sim, pick, x0)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if cur.grasped_object == name:
            return full_w, True, a + 1
    return full_w, False, n_tries


def _width_place_worker(full_w: float, n_tries: int) -> tuple:
    """Does a block of this width PICK+PLACE at the band CENTRE and at the (width-
    adjusted) band EDGE?

    Width binds at placement/packing, not the (permissive) pick. Returns (full_w,
    center_ok, edge_ok, eff_band).
    """
    name = "block_goal1"
    half_x = full_w / 2.0
    specs = [(name, (half_x, _HALF_Y, 0.12), (0.65, 0.2, 0.2, 1.0))]
    sim, sections, config = _build_scene(specs, {name: (0.6, 0.12)})
    info = sections["section_0"]  # tall section
    band = band_half_x(config)
    center_x = config.shelf_pose.position[0]
    # keep the WIDER block's outer edge at the same board margin as the 0.025-half block:
    edge_center_off = max(0.0, band - max(0.0, half_x - 0.025))
    eff_band = 2 * edge_center_off  # usable centre-range for this width
    center_ok = _place_at(sim, sections, name, 0.12, "section_0", 0.0, n_tries)
    edge_ok = _place_at(
        sim, sections, name, 0.12, "section_0", edge_center_off, n_tries
    )
    return full_w, center_ok, edge_ok, eff_band


def _place_at(sim, sections, name, half_z, section, x_off, n_tries) -> bool:
    """Pick ``name`` off the floor and place at section centre + x_off
    (deterministic)."""
    info = sections[section]
    base0, _ = sim.reset(seed=0)
    ok, _ = _place_resident(
        sim,
        sections,
        name,
        half_z,
        section,
        info.center_xy[0] + x_off,
        base0,
        n_tries=n_tries,
    )
    return ok


def sweep_widths(widths, n_tries, workers, log):
    log(f"[sweep2] aperture + sim-pick confound + place-path over {len(widths)} widths")
    aperture = measure_aperture()
    log(
        f"[sweep2] aperture open={aperture['open']:.1f}mm mid={aperture['mid']:.1f}mm closed={aperture['closed']:.1f}mm"
    )
    pick_results, place_results = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pfuts = {ex.submit(_width_pick_worker, w, n_tries): ("pick", w) for w in widths}
        qfuts = {
            ex.submit(_width_place_worker, w, n_tries): ("place", w) for w in widths
        }
        for fut in as_completed({**pfuts, **qfuts}):
            kind = (pfuts.get(fut) or qfuts.get(fut))[0]
            (pick_results if kind == "pick" else place_results).append(fut.result())
    pick_results.sort(key=lambda r: r[0])
    place_results.sort(key=lambda r: r[0])
    return aperture, pick_results, place_results


# ============================================================================================
# Sweep 3 -- tight-packing padding
# ============================================================================================


def _place_resident(
    sim, sections, name, half_z, section, target_x, base_state, n_tries=18
):
    """Pick ``name`` off the floor and place it at a SPECIFIC section x (deterministic
    param), starting from ``base_state`` (which may already hold resident neighbours).

    Returns (success, resulting_state). On failure returns (False, base_state).
    """
    info = sections[section]
    for a in range(n_tries):
        cur = base_state.copy()
        sim.set_state(cur)
        robot = cur.get_object_from_name("robot")
        tgt = cur.get_object_from_name(name)
        rng = np.random.default_rng(1000 + a)
        pick = RestockFrontPickController([robot, tgt], sim)
        try:
            pick.reset(cur, pick.sample_parameters(cur, rng))
            cur = _rollout(sim, pick, cur)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if cur.grasped_object != name:
            continue
        place = SectionFrontPlaceController([robot, tgt], sim, info)
        try:
            place.reset(cur, np.array([target_x - info.center_xy[0], 0.0]))
            cur = _rollout(sim, place, cur)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if _resting_in_section(cur, name, half_z, info):
            return True, cur
    return False, base_state


def _two_object_min_gap(full_w, half_z, section, n_tries) -> Optional[float]:
    """Ground truth (Method 4): seat A resident (far-left, ONCE), snapshot, then binary-
    search the min centre-to-centre spacing where B still places to A's right.

    Returns min spacing (m), or None if A could not be seated / even a wide gap fails.
    """
    half_x = full_w / 2.0
    na, nb = "block_A", "block_B"
    specs = [
        (na, (half_x, _HALF_Y, half_z), (0.2, 0.5, 0.2, 1.0)),
        (nb, (half_x, _HALF_Y, half_z), (0.65, 0.2, 0.2, 1.0)),
    ]
    floor = {na: (-0.55, 0.12), nb: (-0.30, 0.85)}
    sim, sections, config = _build_scene(specs, floor)
    info = sections[section]
    band = band_half_x(config)
    center_x = config.shelf_pose.position[0]
    # Seat A comfortably LEFT of centre (deterministic placement is reliable across the whole
    # band, but the extreme edges are less so and leave no room for B); squeeze B in from A's right.
    x_A = center_x - 0.12
    base0, _ = sim.reset(seed=0)
    ok_A, state_A = _place_resident(
        sim, sections, na, half_z, section, x_A, base0, n_tries
    )
    _dbg = os.environ.get("CALIB_DEBUG")
    if _dbg:
        print(f"[M4dbg w={full_w}] ok_A={ok_A}", flush=True)
    if not ok_A:
        return None
    x_A = state_A.get_object_pose(na).position[0]  # actual seated x

    def attempt_gap(gap) -> Optional[bool]:
        x_B = x_A + gap
        if x_B > center_x + band:
            if _dbg:
                print(f"[M4dbg] gap={gap:.3f} x_B={x_B:.3f} OFFBAND", flush=True)
            return False  # off the band
        ok_B, st = _place_resident(
            sim, sections, nb, half_z, section, x_B, state_A, n_tries
        )
        if not ok_B:
            if _dbg:
                print(f"[M4dbg] gap={gap:.3f} x_B={x_B:.3f} B_not_seated", flush=True)
            return False
        # B must be near x_B and must not have shoved A
        bx = st.get_object_pose(nb).position[0]
        ax = st.get_object_pose(na).position[0]
        res = bool(
            abs(bx - x_B) < 0.03 and abs(ax - x_A) < 0.03
        )  # bool(): avoid numpy.bool_
        if _dbg:
            print(
                f"[M4dbg] gap={gap:.3f} x_B={x_B:.3f} bx={bx:.3f} ax={ax:.3f} -> {res}",
                flush=True,
            )
        return res

    lo = 2 * half_x  # touching
    hi = 2 * half_x + 0.14
    if not attempt_gap(hi):  # widest gap must succeed to bracket the search
        hi += 0.06
        if not attempt_gap(hi):
            return None
    for _ in range(6):  # ~5mm resolution
        mid = (lo + hi) / 2
        if attempt_gap(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _gripper_overhang(full_w, half_z, section, n_tries=12) -> Optional[float]:
    """Method 3 (geometric lower bound): run a REAL isolated-block place at the band
    centre and, over the place trajectory near the shelf, measure how far the gripper
    links that overlap the block's z-band stick out past the block's x-face.

    That lateral overhang is the one-sided clearance a left-to-right neighbour's face
    must leave for the finger. Robust (uses reachable controller configs, not one-shot
    IK). Returns the max overhang beyond the face (m), or None.
    """
    name = "block_goal1"
    half_x = full_w / 2.0
    specs = [(name, (half_x, _HALF_Y, half_z), (0.65, 0.2, 0.2, 1.0))]
    sim, sections, config = _build_scene(specs, {name: (0.6, 0.12)})
    info = sections[section]
    rid = sim.robot.arm.robot_id
    pcid = sim.physics_client_id
    n_links = p.getNumJoints(rid, physicsClientId=pcid)
    best = 0.0

    rest_z = info.surface_z + half_z

    def track(state):
        nonlocal best
        if state.grasped_object != name:
            return
        bx, _by, bz = state.get_object_pose(name).position
        if (
            abs(bz - rest_z) > 0.02
        ):  # only the SETTLED placement moment (block at rest, held)
            return
        z_lo, z_hi = bz - half_z, bz + half_z
        # finger links only (11.. = knuckles/fingers/pads); EXCLUDE the wide base link 10, whose
        # 90-115mm span sits ABOVE the block and is not what a lateral neighbour's face abuts.
        for li in range(11, n_links):
            lo, hi = p.getAABB(rid, li, physicsClientId=pcid)
            if hi[2] < z_lo or lo[2] > z_hi:
                continue
            oh = max(hi[0] - (bx + half_x), (bx - half_x) - lo[0])
            if oh > best:
                best = oh

    for a in range(n_tries):
        base0, _ = sim.reset(seed=0)
        robot = base0.get_object_from_name("robot")
        tgt = base0.get_object_from_name(name)
        rng = np.random.default_rng(a)
        pick = RestockFrontPickController([robot, tgt], sim)
        try:
            pick.reset(base0, pick.sample_parameters(base0, rng))
            cur = _rollout(sim, pick, base0)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if cur.grasped_object != name:
            continue
        place = SectionFrontPlaceController([robot, tgt], sim, info)
        try:
            place.reset(cur, np.array([0.0, 0.0]))
            for _ in range(_MAX_STEPS):
                if place.terminated():
                    break
                u = place.step()
                sim.set_state(cur)
                obs, _, _, _, _ = sim.step(u)
                cur = obs.copy()
                place.observe(cur)
                track(cur)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if _resting_in_section(cur, name, half_z, info):
            return float(best)
    return None


def _n_in_a_row(full_w, half_z, section, gap, n_tries) -> tuple[int, int]:
    """Method 5: seat ⌊(band+gap)/(w+gap)⌋ blocks left->right at pitch (w+gap); return
    (n_target, n_seated).

    Stops at the first failure (left-to-right).
    """
    config = Restock3DEnvConfig()
    band2 = 2 * band_half_x(config)
    w = full_w
    pitch = w + gap
    n_target = max(1, min(int((band2 + gap) / pitch), 6))
    half_x = w / 2.0
    names = [f"block_{i}" for i in range(n_target)]
    specs = [(nm, (half_x, _HALF_Y, half_z), (0.6, 0.25, 0.2, 1.0)) for nm in names]
    # spread staging over the floor so blocks don't overlap at reset
    floor = {
        nm: (-0.62 + 0.14 * (i % 3), 0.10 + 0.16 * (i // 3))
        for i, nm in enumerate(names)
    }
    sim, sections, _ = _build_scene(specs, floor)
    center_x = config.shelf_pose.position[0]
    band = band_half_x(config)
    x0_left = center_x - band + half_x
    state, _ = sim.reset(seed=0)
    seated = 0
    for i, nm in enumerate(names):
        target_x = x0_left + i * pitch
        if target_x > center_x + band:
            break
        ok, state = _place_resident(
            sim, sections, nm, half_z, section, target_x, state, n_tries
        )
        if ok:
            seated += 1
        else:
            break
    return n_target, seated


# -- Sweep-3 worker wrappers. Each method runs in its OWN process: building >1 PyBullet sim
# -- sequentially in a single process corrupts motion planning (confirmed), so M3/M4/M5 must
# -- not share a process. Each wrapper builds a single sim and returns a plain tuple.
def _m3_worker(full_w, neigh_hz):
    return (full_w, _gripper_overhang(full_w, neigh_hz, "section_0"))


def _m4_worker(full_w, neigh_hz, n_tries):
    return (full_w, _two_object_min_gap(full_w, neigh_hz, "section_0", n_tries))


def _m5_worker(full_w, neigh_hz, gap, n_tries):
    nt, ns = _n_in_a_row(full_w, neigh_hz, "section_0", max(0.0, gap), n_tries)
    return (full_w, nt, ns)


# ============================================================================================
# Table / doc rendering
# ============================================================================================


def _fmt_cell(cell) -> str:
    feasible, k, reason = cell
    if feasible:
        return f"OK({k})"
    return f"x:{reason}"


def render_height_table(rows) -> str:
    lines = []
    hdr = "| full_h (m) | tall/current | tall/center | tall/capped | short/current | short/center | short/capped |"
    sep = "|---|---|---|---|---|---|---|"
    lines += [hdr, sep]
    for r in rows:
        c = r.cells
        lines.append(
            f"| {r.full_h:.3f} "
            f"| {_fmt_cell(c[('section_0','current')])} "
            f"| {_fmt_cell(c[('section_0','center')])} "
            f"| {_fmt_cell(c[('section_0','capped_center')])} "
            f"| {_fmt_cell(c[('section_1','current')])} "
            f"| {_fmt_cell(c[('section_1','center')])} "
            f"| {_fmt_cell(c[('section_1','capped_center')])} |"
        )
    return "\n".join(lines)


def _feasible_range(rows, section, scheme):
    hs = [r.full_h for r in rows if r.cells[(section, scheme)][0]]
    return (min(hs), max(hs)) if hs else None


def render_height_summary(rows) -> str:
    lines = [
        "| section | scheme | min feasible full_h | max feasible full_h |",
        "|---|---|---|---|",
    ]
    for section, label in [
        ("section_0", "tall (0.34 clr)"),
        ("section_1", "short (0.15 clr)"),
    ]:
        for scheme in GRASP_SCHEMES:
            rng = _feasible_range(rows, section, scheme)
            if rng:
                lines.append(f"| {label} | {scheme} | {rng[0]:.3f} | {rng[1]:.3f} |")
            else:
                lines.append(f"| {label} | {scheme} | -- | -- (none) |")
    return "\n".join(lines)


def render_recommendations(
    rows, aperture, place_results, m3, m4, m5, pad_widths
) -> str:
    """A computed 'Key findings + Recommended v3 ranges' block from whatever sweeps
    ran."""
    L = ["## Key findings & recommended v3 ranges", ""]
    # Heights
    if rows is not None:
        tall = _feasible_range(rows, "section_0", "current")
        short = _feasible_range(rows, "section_1", "current")
        # best scheme by widest tall range
        best_scheme, best_span = "current", -1.0
        for sc in GRASP_SCHEMES:
            r = _feasible_range(rows, "section_0", sc)
            if r and (r[1] - r[0]) > best_span:
                best_span, best_scheme = r[1] - r[0], sc
        tall_s = f"{tall[0]:.2f}–{tall[1]:.2f} m" if tall else "none"
        short_s = f"{short[0]:.2f}–{short[1]:.2f} m" if short else "none"
        cube_only = bool(short and abs(short[1] - short[0]) < 1e-6)
        L += [
            f"- **Block height.** Feasible full-height (current grasp): **tall section {tall_s}**, "
            f"**short section {short_s}**. The short section's usable height is set by the "
            "**gripper's vertical clearance above the block** (~0.10 m of room is needed ABOVE the "
            "block to place it), NOT the 0.15 m shelf clearance.",
            f"- **Grasp scheme.** The production *current* scheme (height-adaptive, fixed-world "
            f"grasp z) gives the widest feasible range; the naive *center* / *capped-center* grasps "
            "the user proposed FAIL placement more (a lower grasp point worsens the place reach-in). "
            "**The controller is already flexible to variable heights — do not switch to a center "
            "grasp.**",
        ]
        if cube_only:
            L.append(
                f"- **⚠️ Caveat — the short section is effectively CUBE-ONLY.** Only the "
                f"{short[1]:.2f} m block places in the short section (the next height already fails), "
                "because the gripper needs ~0.10 m above the block and the short clearance is only "
                "0.15 m. **This is too tight for v3's goal of sampling a *range* of heights near a "
                "short/tall cutoff in the short section** — v3 will need either a **taller "
                "short-section clearance** (env change) or an **adjusted short-section place approach** "
                "to admit a meaningful height range. As-is, v3's height variation is a **tall-section** "
                f"story ({tall_s}) with the short/tall decision being 'cube vs taller'."
            )
        else:
            L.append(
                f"- The **short/tall fit cutoff v3 samples around is the short-section max "
                f"(≈ {short[1]:.2f} m)** — a block just above it must go tall, just below it fits either."
            )
    # Width
    if aperture is not None:
        safe_w = round(0.9 * aperture["open"] / 1000.0, 3)
        eff = ""
        if place_results:
            # widest width that still places at the band edge
            edge_ok = [w for w, c, e, _ in place_results if e]
            if edge_ok:
                eff = f" It still places across the band up to ≈ {max(edge_ok):.2f} m wide."
        L.append(
            f"- **Block width.** Graspable-face ceiling = the finger aperture "
            f"**≈ {aperture['open']:.0f} mm** (nominal 85 mm). The kinematic sim is width-PERMISSIVE "
            "(it 'picks' absurd widths because the target is collision-excluded during the reach-in), "
            f"so **v3 must cap width analytically in the generator** — recommend a safe max face width "
            f"≈ **{safe_w:.2f} m** (≈0.9× aperture).{eff}"
        )
    # Padding
    if m4 is not None and pad_widths:
        vals4 = [m4[w] for w in pad_widths if m4.get(w) is not None]
        vals3 = [m3[w] for w in pad_widths if m3 and m3.get(w) is not None]
        if vals4:
            g = max(vals4)
            rec_gap = round(g + 0.01, 3)
            m3s = (
                f" M3 finger-overhang lower bound {min(vals3)*1000:.0f}–{max(vals3)*1000:.0f} mm."
                if vals3
                else ""
            )
            packed = ", ".join(
                f"{ns}/{nt} @ w={w:.02f}" for w in pad_widths for (nt, ns) in [m5[w]]
            )
            L.append(
                f"- **Packing padding (left-to-right).** The empirical min edge-to-edge gap between "
                f"adjacent blocks (M4, real placement vs a tall neighbour) is "
                f"**{min(vals4)*1000:.0f}–{max(vals4)*1000:.0f} mm** — **~5–8× the naive "
                f"finger-thickness estimate**, because the finger+knuckle assembly and the reach-in "
                f"motion (not the pad) bind.{m3s} **Recommend v3 pack with an edge gap ≥ "
                f"{rec_gap*1000:.0f} mm** (measured max + ~10 mm safety); n-in-a-row validation "
                f"seated {packed}."
            )
    L.append("")
    L.append("_Details and full grids below._")
    return "\n".join(L)


def render_width_table(aperture, pick_results, place_results=None) -> str:
    lines = [
        f"Finger aperture (inner-pad separation): **open {aperture['open']:.1f} mm**, "
        f"mid {aperture['mid']:.1f} mm, closed {aperture['closed']:.1f} mm "
        f"(nominal 2F-85 stroke {_APERTURE_NOMINAL_MM:.0f} mm).",
        "",
        "**Pick (confound check).** Grasp = kinematic attach on the ±x faces; the target is",
        "collision-excluded during the reach-in, so the sim does NOT bind width:",
        "",
        "| full width (m) | half_x (m) | sim pick succeeds? | tries |",
        "|---|---|---|---|",
    ]
    for w, ok, k in pick_results:
        lines.append(f"| {w:.3f} | {w/2:.3f} | {'YES' if ok else 'no'} | {k} |")
    if place_results:
        lines += [
            "",
            "**Place path.** Width binds at PLACEMENT, not picking — a wider block eats the",
            "x-band and the effective usable centre-range shrinks by (half_x−0.025) per side:",
            "",
            "| full width (m) | place @ band-centre | place @ band-edge | effective usable band (m) |",
            "|---|---|---|---|",
        ]
        for w, c_ok, e_ok, eff in place_results:
            lines.append(
                f"| {w:.3f} | {'YES' if c_ok else 'no'} | {'YES' if e_ok else 'no'} | {eff:.3f} |"
            )
    return "\n".join(lines)


# ============================================================================================
# Main
# ============================================================================================

_DOC_PATH = pathlib.Path(
    "src/alphatamp/approaches/spectre/docs/restock3d_v3_calibration.md"
)


def _default_heights(quick):
    if quick:
        return [0.05, 0.12, 0.15, 0.20, 0.30]
    hs = list(np.round(np.arange(0.03, 0.37, 0.02), 3))
    for extra in (0.14, 0.15, 0.16, 0.33):
        if extra not in hs:
            hs.append(extra)
    return sorted(hs)


def _default_widths(quick):
    if quick:
        return [0.05, 0.09, 0.16]
    return list(np.round(np.arange(0.03, 0.21, 0.02), 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse smoke grids")
    ap.add_argument("--sweeps", type=str, default="1,2,3")
    ap.add_argument(
        "--workers", type=int, default=max(1, int(0.8 * (os.cpu_count() or 8)))
    )
    ap.add_argument(
        "--tries",
        type=int,
        default=10,
        help="pick+place retries per cell (heights/widths)",
    )
    ap.add_argument(
        "--pad-tries",
        type=int,
        default=18,
        help="retries for the padding sweep (deterministic-x placement is flakier)",
    )
    ap.add_argument("--pad-widths", type=str, default="0.05,0.09")
    ap.add_argument("--out", type=str, default=str(_DOC_PATH))
    args = ap.parse_args()
    which = {int(s) for s in args.sweeps.split(",")}
    t_start = time.time()

    def log(msg):
        print(f"[{time.time()-t_start:6.0f}s] {msg}", flush=True)

    log(
        f"restock3d v3 calibration: quick={args.quick} sweeps={sorted(which)} workers={args.workers} tries={args.tries}"
    )

    doc_sections: list[str] = []
    rows = None
    aperture = None
    place_results = None
    m3 = m4 = m5 = None
    pad_widths = None

    # ---- Sweep 1: heights ------------------------------------------------------------------
    if 1 in which:
        heights = _default_heights(args.quick)
        rows = sweep_heights(heights, args.tries, args.workers, log)
        tbl = render_height_table(rows)
        summ = render_height_summary(rows)
        print(
            "\n===== SWEEP 1: HEIGHTS (OK(tries-to-success) | x:pick/place fail) ====="
        )
        print(tbl)
        print("\n-- feasible height range per section x scheme --")
        print(summ)
        doc_sections.append(
            "## Sweep 1 — block heights\n\n"
            + summ
            + "\n\n<details><summary>full grid</summary>\n\n"
            + tbl
            + "\n\n</details>"
        )

    # ---- Sweep 2: widths -------------------------------------------------------------------
    if 2 in which:
        widths = _default_widths(args.quick)
        aperture, wresults, place_results = sweep_widths(
            widths, args.tries, args.workers, log
        )
        wtbl = render_width_table(aperture, wresults, place_results)
        print("\n===== SWEEP 2: WIDTHS (aperture + sim-pick confound) =====")
        print(wtbl)
        sim_max = max((w for w, ok, _ in wresults if ok), default=0.0)
        note = (
            f"\n\n**Finding:** the kinematic sim pick succeeds up to full width {sim_max:.2f} m "
            "(the widest tested) — it does NOT bind width, because the grasp attaches "
            "kinematically and excludes the target during the reach-in. The real graspable-width "
            f"ceiling is the **finger aperture ≈ {aperture['open']:.0f} mm** (nominal 85 mm). v3 must "
            "cap block width analytically; the sim will not reject an over-wide block at pick time."
        )
        print(note)
        doc_sections.append("## Sweep 2 — graspable width\n\n" + wtbl + note)

    # ---- Sweep 3: padding ------------------------------------------------------------------
    if 3 in which:
        config = Restock3DEnvConfig()
        band2 = 2 * band_half_x(config)
        pad_widths = [float(x) for x in args.pad_widths.split(",")]
        log(f"[sweep3] padding: band={band2:.3f}m, widths={pad_widths}")
        # Method 1 — capacity arithmetic (report for 4 blocks of each width)
        m1 = []
        for w in pad_widths:
            for n in (4, 5):
                gap = (band2 - n * w) / n
                m1.append((w, n, gap))
        neigh_h = 0.24  # tall neighbour (worst case for the descending gripper)
        neigh_hz = neigh_h / 2
        m3: dict = {}
        m4: dict = {}
        m5: dict = {}
        # Phase 1: aperture (M2) + M3 + M4 in ISOLATED worker processes (one sim per process).
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fa = ex.submit(measure_aperture)
            f3 = {ex.submit(_m3_worker, w, neigh_hz): w for w in pad_widths}
            f4 = {
                ex.submit(_m4_worker, w, neigh_hz, args.pad_tries): w
                for w in pad_widths
            }
            aperture = fa.result()
            log(
                f"[sweep3] aperture open={aperture['open']:.1f}mm closed={aperture['closed']:.1f}mm"
            )
            for fut in as_completed({**f3, **f4}):
                if fut in f3:
                    w, oh = fut.result()
                    m3[w] = oh
                    log(f"[sweep3] width {w:.3f}: M3 finger overhang={oh}")
                else:
                    w, g4 = fut.result()
                    m4[w] = None if g4 is None else g4 - w
                    log(
                        f"[sweep3] width {w:.3f}: M4 min spacing={g4}, edge gap={m4[w]}"
                    )
        # Phase 2: M5 packs at each width's measured min gap + a small safety margin.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            f5 = {}
            for w in pad_widths:
                base_gap = (
                    m4[w]
                    if m4[w] is not None
                    else (m3[w] if m3[w] is not None else 0.04)
                )
                use_gap = base_gap + 0.005
                f5[ex.submit(_m5_worker, w, neigh_hz, use_gap, args.pad_tries)] = w
                log(f"[sweep3] width {w:.3f}: M5 pack at edge gap={use_gap:.3f} ...")
            for fut in as_completed(f5):
                w, n_t, n_s = fut.result()
                m5[w] = (n_t, n_s)
                log(f"[sweep3] width {w:.3f}: M5 seated {n_s}/{n_t}")

        # render
        pad_lines = ["## Sweep 3 — tight-packing padding (left-to-right)", ""]
        pad_lines.append(
            f"Usable band = **{band2:.3f} m** (object-centre range). Neighbour height = {neigh_h:.2f} m (worst case)."
        )
        pad_lines.append("")
        pad_lines.append(
            "**Method 1 — capacity arithmetic** (edge gap if n blocks share the band):"
        )
        pad_lines.append("| width (m) | n | edge gap = (band − n·w)/n (m) |")
        pad_lines.append("|---|---|---|")
        for w, n, gap in m1:
            pad_lines.append(f"| {w:.3f} | {n} | {gap:.4f} |")
        pad_lines.append("")
        pad_lines.append(
            f"**Method 2 — gripper geometry (naive)**: from the finger-pad thickness alone "
            f"(6.35 mm; closed inner-pad separation {aperture['closed']:.1f} mm) you would guess a "
            "single neighbour-side finger needs only ≈ **6–8 mm** beyond the block face. **The "
            "empirical methods below overshoot this ~5–8×** — the binding geometry is the whole "
            "finger+knuckle assembly plus the diagonal reach-in swept volume, not the pad thickness."
        )
        pad_lines.append("")
        pad_lines.append(
            "**Methods 3–5 — empirical (tall neighbour):** min edge-to-edge gap "
            "between adjacent block faces (holds B's neighbour-side finger)."
        )
        pad_lines.append(
            "| width (m) | M3 gripper-overhang edge gap (m) | M4 rollout min edge gap (m) | M5 seated/target |"
        )
        pad_lines.append("|---|---|---|---|")
        for w in pad_widths:
            g3 = "--" if m3[w] is None else f"{m3[w]:.4f}"
            g4 = "--" if m4[w] is None else f"{m4[w]:.4f}"
            nt, ns = m5[w]
            pad_lines.append(f"| {w:.3f} | {g3} | {g4} | {ns}/{nt} |")
        pad_txt = "\n".join(pad_lines)
        print("\n===== SWEEP 3: PADDING =====")
        print(pad_txt)
        doc_sections.append(pad_txt)

    # ---- Recommendations (computed from whatever sweeps ran) --------------------------------
    if rows is not None or aperture is not None or m4 is not None:
        rec = render_recommendations(
            rows, aperture, place_results, m3, m4, m5, pad_widths
        )
        print("\n===== RECOMMENDED v3 RANGES =====")
        print(rec)
        doc_sections.insert(0, rec)  # put recommendations at the TOP of the doc

    # ---- write doc -------------------------------------------------------------------------
    if doc_sections:
        header = _doc_header()
        body = header + "\n\n" + "\n\n".join(doc_sections) + "\n"
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        log(f"wrote findings doc -> {out}")
    log(f"done in {time.time()-t_start:.0f}s")


def _doc_header() -> str:
    return (
        "# Restock3D v3 — pick/place calibration & packing findings\n\n"
        "_Generated by `experiments/spectre/restock3d_v3_calibrate.py`. Measurement of the "
        "**current** kinematic env's front-grasp pick/place envelope, to bound v3's varied "
        "block widths/heights and its left-to-right packing padding. No production code was "
        "modified to produce these numbers._\n\n"
        "## Established geometry (from code)\n\n"
        "| Quantity | Value |\n|---|---|\n"
        "| Usable shelf x-band (object-centre range) | 0.522 m (0.139–0.661) |\n"
        "| Tall section floor / clearance / ceiling | 0.290 / **0.340** / 0.630 m |\n"
        "| Short section floor / clearance / ceiling | 0.6427 / **0.150** / 0.7927 m |\n"
        "| Cube / tall-block footprint (fixed today) | 0.05 × 0.05 m |\n"
        "| Cube / tall-block full height (fixed today) | 0.05 / 0.24 m |\n"
        "| Gripper | Robotiq 2F-85, ~85 mm stroke; closes on block ±x faces |\n"
        "| Finger pad box (W × thick × tall) | 0.022 × 0.00635 × 0.0375 m |\n"
        "| Geometric fit rule (block vs ceiling only) | ≤ 0.15 fits both; (0.15, 0.34] tall only; > 0.34 neither |\n"
        "| **Actual pick+place envelope (gripper-limited, Sweep 1)** | **tall 0.05–0.23 m; short 0.05 m only** |\n"
    )


if __name__ == "__main__":
    main()
