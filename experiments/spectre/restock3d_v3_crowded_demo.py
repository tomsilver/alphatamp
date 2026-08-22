"""Restock3D **v3** crowded-feasibility demo (uniform footprints).

Confirms the *most crowded* v3 scene is physically solvable with the re-balanced shelf and an
analytic left-to-right placement refiner. Concretely:

1. **Re-balanced partition** — move the divider DOWN 0.07 m: ``section_clearances=(0.27, 0.22)``
   (was (0.34, 0.15)), same total shelf height. Tall section now 0.27 m, short section 0.22 m
   (each leaves ~0.10 m gripper headroom for its max block).
2. **Left-to-right analytic refiner** — ``LeftToRightSectionPlaceController`` computes each block's
   EXACT slot x (leftmost centre + i·(w + gap), gap 0.06 m) instead of uniform x sampling, with a
   small ±0.01 m jitter and only ``--place-samples`` (default 5) retries.
3. **10 blocks, uniform 0.05×0.05 footprint** — 5 short (0.12 m tall) in a FRONT row, 5 tall
   (0.17 m tall) in a BACK row (the per-section height limits from the calibration study).
4. **Oracle plan** — pick short blocks closest-first (right→left), place them left-to-right into
   the short (top) section; then the tall blocks the same way into the tall (bottom) section.
5. **Execute with the REAL controllers** (real MP + PyBullet collision) and render an mp4 of the
   whole episode.

Runs a **single-block feasibility gate first** (a 0.12 m block into the short section, a 0.17 m
block into the tall section); if either fails the crowded demo is impossible and we say so before
spending time on the full run.

Run (repo root, venv active)::

    python experiments/spectre/restock3d_v3_crowded_demo.py --check-only   # just the gate
    python experiments/spectre/restock3d_v3_crowded_demo.py                # gate + full demo + mp4
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached). --------------------
import glob
import os
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
from typing import Optional

import imageio.v2 as iio
import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.camera import capture_image
from pybullet_helpers.inverse_kinematics import InverseKinematicsError

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
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    band_half_x,
    compute_section_infos,
)

# ---- v3 design under test -------------------------------------------------------------------
_SECTION_CLEARANCES = (
    0.27,
    0.22,
)  # (tall, short) — divider moved down 0.07 m; same total H
_GAP = 0.06  # analytic left-to-right edge gap (m)
_JITTER = 0.01  # placement jitter (m)
_HALF_XY = 0.025  # 0.05 x 0.05 footprint (unchanged)
_SHORT_FULL_H = 0.12  # short-block height (max for the 0.22 m short section)
_TALL_FULL_H = 0.17  # tall-block height (max for the 0.27 m tall section)
_N_PER = 5  # 5 short + 5 tall
_MAX_STEPS = 900

_TALL_SECTION, _SHORT_SECTION = "section_0", "section_1"
_OUT_DIR = pathlib.Path(
    "src/alphatamp/approaches/spectre/envs/restock3d/demos/v3_crowded"
)


def _config() -> Restock3DEnvConfig:
    return Restock3DEnvConfig(section_clearances=_SECTION_CLEARANCES)


class LeftToRightSectionPlaceController(SectionFrontPlaceController):
    """The modified refiner: place at a pre-computed slot x (analytic left-to-right
    packing) with a small ±jitter, instead of uniform-across-band sampling."""

    def __init__(self, objects, sim, section_info, target_x, jitter=_JITTER):
        super().__init__(objects, sim, section_info)
        self._target_x = float(target_x)
        self._jitter = float(jitter)

    def sample_parameters(self, x, rng):
        cx = self._section_info.center_xy[0]
        return np.array(
            [
                self._target_x - cx + rng.uniform(-self._jitter, self._jitter),
                rng.uniform(-self._jitter, self._jitter),
            ]
        )


def _slot_centers(config, n=_N_PER, half_x=_HALF_XY, gap=_GAP) -> list[float]:
    """Analytic left-to-right slot centre x's: leftmost block's LEFT EDGE at the board margin,
    then pitch = width + gap. (Matches the calibration's validated n-in-a-row convention.)
    """
    cx = config.shelf_pose.position[0]
    band = band_half_x(config)
    x0 = cx - band + half_x  # leftmost centre
    pitch = 2 * half_x + gap
    return [x0 + i * pitch for i in range(n)]


def _render(sim) -> np.ndarray:
    kw = sim.config.get_camera_kwargs()
    return capture_image(sim.physics_client_id, image_width=640, image_height=480, **kw)


def _rollout(sim, controller, x, frames=None, skip=2) -> object:
    cur = x
    i = 0
    for _ in range(_MAX_STEPS):
        if controller.terminated():
            break
        u = controller.step()
        sim.set_state(cur)
        obs, _, _, _, _ = sim.step(u)
        cur = obs.copy()
        controller.observe(cur)
        if frames is not None and (i % skip == 0):
            frames.append(_render(sim))
        i += 1
    return cur


def _resting_in_section(state, name, half_z, info) -> bool:
    x, y, z = state.get_object_pose(name).position
    return (
        abs((z - half_z) - info.surface_z) < 0.05
        and abs(x - info.center_xy[0]) <= info.half_xy[0] + 0.06
        and abs(y - info.center_xy[1]) <= info.half_xy[1] + 0.06
    )


def _spec(name, full_h, rgba):
    return (name, (_HALF_XY, _HALF_XY, full_h / 2.0), rgba)


# ---------------------------------------------------------------------------------------------
# Single-block feasibility gate
# ---------------------------------------------------------------------------------------------
def feasibility_gate(samples=8) -> tuple[bool, str]:
    """A 0.12 m block must place in the short section and a 0.17 m block in the tall
    section."""
    config = _config()
    sections = compute_section_infos(config)
    cx = config.shelf_pose.position[0]
    cases = [
        ("short_0.12_into_short", _SHORT_FULL_H, _SHORT_SECTION, True),
        ("tall_0.17_into_tall", _TALL_FULL_H, _TALL_SECTION, True),
        ("tall_0.17_into_short_expect_F3", _TALL_FULL_H, _SHORT_SECTION, False),
    ]
    msgs = []
    ok_all = True
    for label, full_h, section, expect in cases:
        name = "blk"
        specs = [_spec(name, full_h, (0.6, 0.2, 0.2, 1.0))]
        placed = _one_block_place(specs, name, full_h, section, cx, samples)
        good = placed == expect
        ok_all = ok_all and good
        msgs.append(
            f"  {label}: placed={placed} expect={expect} -> {'OK' if good else 'FAIL'}"
        )
    return ok_all, "\n".join(msgs)


def _one_block_place(specs, name, full_h, section, target_x, samples) -> bool:
    config = _config()
    sections = compute_section_infos(config)
    info = sections[section]

    def pose_fn(seed):
        del seed
        return {name: (0.6, 0.12)}

    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, sections, config=config, allow_state_access=True
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
        place = LeftToRightSectionPlaceController(
            [robot, tgt], sim, info, target_x=info.center_xy[0]
        )
        try:
            place.reset(cur, place.sample_parameters(cur, rng))
            cur = _rollout(sim, place, cur)
        except (TrajectorySamplingFailure, InverseKinematicsError):
            continue
        if _resting_in_section(cur, name, half_z, info):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Full crowded oracle demo
# ---------------------------------------------------------------------------------------------
def _build_crowded_scene(config):
    """5 short blocks (front row) + 5 tall blocks (back row), uniform footprint.

    Returns
    (sim, sections, short_names, tall_names, floor_x_by_name).
    """
    x_cols = [-0.73, -0.60, -0.47, -0.34, -0.21]  # 5 columns, west of the shelf
    short_y, tall_y = 0.70, 1.05
    short_names = [f"short_{i}" for i in range(_N_PER)]
    tall_names = [f"tall_{i}" for i in range(_N_PER)]
    specs, floor = [], {}
    for i, x in enumerate(x_cols):
        specs.append(
            _spec(short_names[i], _SHORT_FULL_H, (0.15, 0.5, 0.9, 1.0))
        )  # blue-ish
        floor[short_names[i]] = (x, short_y)
        specs.append(
            _spec(tall_names[i], _TALL_FULL_H, (0.8, 0.3, 0.15, 1.0))
        )  # orange-ish
        floor[tall_names[i]] = (x, tall_y)

    def pose_fn(seed):
        del seed
        return floor

    sections = compute_section_infos(config)
    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, sections, config=config, allow_state_access=True
    )
    floor_x = {n: floor[n][0] for n in floor}
    return sim, sections, short_names, tall_names, floor_x


def run_demo(
    place_samples=5, pick_samples=14, frame_skip=2, fps=30
) -> tuple[bool, str, list]:
    """Execute the oracle plan with the real controllers, capturing frames.

    Returns
    (all_placed, report, frames).
    """
    config = _config()
    sim, sections, short_names, tall_names, floor_x = _build_crowded_scene(config)
    slots = _slot_centers(config)
    frames: list = []
    log_lines = []

    x0, _ = sim.reset(seed=0)
    frames.append(_render(sim))
    cur = x0

    # groups: (block names, section, full_h) — short (top) first, then tall (bottom)
    groups = [
        (short_names, _SHORT_SECTION, _SHORT_FULL_H, "short"),
        (tall_names, _TALL_SECTION, _TALL_FULL_H, "tall"),
    ]
    for names, section, full_h, tag in groups:
        info = sections[section]
        half_z = full_h / 2.0
        # pick order: closest-first == right-to-left == descending floor x
        order = sorted(names, key=lambda n: floor_x[n], reverse=True)
        for slot_idx, name in enumerate(order):
            target_x = slots[slot_idx]
            # --- pick ---
            picked = False
            for a in range(pick_samples):
                sim.set_state(cur)
                robot = cur.get_object_from_name("robot")
                tgt = cur.get_object_from_name(name)
                rng = np.random.default_rng(100 + a)
                pick = RestockFrontPickController([robot, tgt], sim)
                try:
                    pick.reset(cur, pick.sample_parameters(cur, rng))
                    nxt = _rollout(sim, pick, cur, frames, frame_skip)
                except (TrajectorySamplingFailure, InverseKinematicsError):
                    continue
                if nxt.grasped_object == name:
                    cur = nxt
                    picked = True
                    break
            if not picked:
                return (
                    False,
                    f"PICK FAILED: {name} ({tag}) after {pick_samples} tries\n"
                    + "\n".join(log_lines),
                    frames,
                )
            # --- place (analytic left-to-right slot) ---
            placed = False
            attempts_used = 0
            for a in range(place_samples):
                sim.set_state(cur)
                robot = cur.get_object_from_name("robot")
                tgt = cur.get_object_from_name(name)
                rng = np.random.default_rng(500 + a)
                place = LeftToRightSectionPlaceController(
                    [robot, tgt], sim, info, target_x=target_x
                )
                try:
                    place.reset(cur, place.sample_parameters(cur, rng))
                    nxt = _rollout(sim, place, cur, frames, frame_skip)
                except (TrajectorySamplingFailure, InverseKinematicsError):
                    continue
                if _resting_in_section(nxt, name, half_z, info):
                    cur = nxt
                    placed = True
                    attempts_used = a + 1
                    break
            if not placed:
                return (
                    False,
                    f"PLACE FAILED: {name} ({tag}) -> slot {slot_idx} (x={target_x:.3f}) "
                    f"after {place_samples} samples\n" + "\n".join(log_lines),
                    frames,
                )
            log_lines.append(
                f"  {tag} {name} -> slot {slot_idx} x={target_x:.3f}  ({attempts_used} place-samples)"
            )
    # hold the final frame
    for _ in range(15):
        frames.append(_render(sim))
    return True, "\n".join(log_lines), frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="just the single-block feasibility gate",
    )
    ap.add_argument("--place-samples", type=int, default=5)
    ap.add_argument("--pick-samples", type=int, default=14)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    log(
        f"v3 crowded demo: clearances={_SECTION_CLEARANCES} gap={_GAP} "
        f"short={_SHORT_FULL_H} tall={_TALL_FULL_H} place_samples={args.place_samples}"
    )
    log(f"slot centres = {[round(x,3) for x in _slot_centers(_config())]}")

    log("=== feasibility gate (single block per section) ===")
    ok, msg = feasibility_gate()
    print(msg, flush=True)
    if not ok:
        log("GATE FAILED — the crowded demo is NOT possible as specified. Stopping.")
        return
    log("gate PASSED")
    if args.check_only:
        return

    log("=== full crowded oracle demo (10 blocks) ===")
    placed, report, frames = run_demo(
        place_samples=args.place_samples,
        pick_samples=args.pick_samples,
        frame_skip=args.frame_skip,
        fps=args.fps,
    )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / "crowded_10block_oracle.mp4"
    iio.mimsave(out, frames, fps=args.fps, macro_block_size=16)
    print(report, flush=True)
    if placed:
        log(
            f"SUCCESS — all 10 blocks placed. Wrote {out} ({len(frames)} frames) in {time.time()-t0:.0f}s"
        )
    else:
        log(
            f"INCOMPLETE — see failure above. Partial video: {out} ({len(frames)} frames)"
        )


if __name__ == "__main__":
    main()
