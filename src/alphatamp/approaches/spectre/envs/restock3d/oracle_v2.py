"""Privileged oracle + certification for **Restock3D v2** (continuous packing).

Builds a correct v2 skeleton directly — tall blocks to the tall section (F3 forces it),
cubes balanced across the two sections, stored SOUTH-TO-NORTH (nearest-first, so the
front-grasp reach path over farther objects is always clear) — and certifies it by a
**manual multi-object rollout with per-step resampling**: each pick / place is retried
on ``TrajectorySamplingFailure`` with a fresh sample, which is exactly the continuous-
packing backtracking (a place whose sampled x collides a resident resamples to a free x;
a full section exhausts its retries). This is the milestone certification path — it does
NOT need the collection pipeline / real BacktrackingRefiner (deferred).

Section choice is validated by real PyBullet collision, never a toy gate:
``place_short`` of a tall block collides the short section's ceiling board (F3) and
never certifies there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from relational_structs import GroundAtom

from .kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from .models_v2 import RestockAbstractorV2, Stored
from .place_controller import RestockFrontPickController
from .place_controller_v2 import SectionFrontPlaceController
from .region_geometry import RegionInfo
from .section_geometry import compute_section_infos

_TALL_PREFIX = "block_goal"
_CUBE_PREFIX = "cube_goal"
_MAX_STEPS = 900
# Placements are ~1/6 reliable per sample (BiRRT flakiness; each attempt resamples x across the band
# + the pick standoff/rot), so certification retries generously per step — matching v1's demo budget.
_ATTEMPTS_PER_STEP = 18


@dataclass(frozen=True)
class OracleResultV2:
    """Per-instance v2 oracle outcome (feasibility certificate)."""

    stratum: int
    problem_id: int
    certified_feasible: bool
    plan_len: int
    note: str


@dataclass
class V2Bundle:
    """A stratum's built sim + geometry + abstractor (reused across problems)."""

    sim: ObjectCentricRestock3DEnv
    section_infos: dict[str, RegionInfo]
    goal_names: list[str]
    abstractor: RestockAbstractorV2


def build_v2_bundle(
    stratum: int, config: Optional[Restock3DEnvConfig] = None
) -> V2Bundle:
    """Build the v2 sim + section bands + abstractor for a stratum (no obs/act spaces
    needed)."""
    from .models_v2 import stratum_env_args_v2  # local: avoid a cycle at import time

    object_specs, pose_fn, section_infos, config = stratum_env_args_v2(stratum, config)
    sim = ObjectCentricRestock3DEnv(
        object_specs, pose_fn, section_infos, config=config, allow_state_access=True
    )
    goal_names = [
        n for n, _, _ in object_specs if n.startswith((_CUBE_PREFIX, _TALL_PREFIX))
    ]
    return V2Bundle(
        sim, section_infos, goal_names, RestockAbstractorV2(section_infos, goal_names)
    )


def solve_assignment_v2(
    section_infos: dict[str, RegionInfo], goal_object_names: list[str]
) -> list[tuple[str, str]]:
    """Assign each goal object to a section key (``section_0`` tall / ``section_1``
    short).

    Tall blocks go to the tall section (a short placement is F3). Cubes are **load-
    balanced** across the two sections (fewer current objects wins; ties -> short) so no
    single band is crowded — a cube fits either section, and spreading them keeps the
    continuous packing loose. Returns ``(obj_name, section_key)`` pairs, talls first.
    """
    talls = sorted(n for n in goal_object_names if n.startswith(_TALL_PREFIX))
    cubes = sorted(n for n in goal_object_names if n.startswith(_CUBE_PREFIX))
    load = {"section_0": 0, "section_1": 0}
    pairs: list[tuple[str, str]] = []
    for t in talls:
        pairs.append((t, "section_0"))
        load["section_0"] += 1
    for c in cubes:
        sec = "section_1" if load["section_1"] <= load["section_0"] else "section_0"
        pairs.append((c, sec))
        load[sec] += 1
    return pairs


def build_skeleton_v2(x0, assignment: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Order the ``(obj, section)`` assignment SOUTH-TO-NORTH by floor y (nearest-
    first).

    Nearest-first clears each object's front-grasp reach path (all closer objects are
    stored before a farther one is picked); picking a far object over a nearer one on
    the floor makes the reach-over collide — a real refinement failure the naive order
    hits and the oracle avoids.
    """
    return sorted(assignment, key=lambda pr: x0.get_object_pose(pr[0]).position[1])


def _mix(seed: int, step: int, phase: int, attempt: int) -> int:
    return seed * 100003 + step * 997 + phase * 31 + attempt


def _transition(sim, x, u):
    sim.set_state(x)
    obs, _, _, _, _ = sim.step(u)
    return obs.copy()


def _stored(abstractor: RestockAbstractorV2, state, obj_name: str) -> bool:
    obj = state.get_object_from_name(obj_name)
    return GroundAtom(Stored, [obj]) in abstractor.state_abstractor(state).atoms


def _attempt(sim, controller, cur, success_fn, rng, render):
    """Roll one controller out from ``cur``; return ``(ok, new_state, frames)``.

    A ``TrajectorySamplingFailure`` (at sample or any step — e.g. F3 planning, or an F2 collision)
    is a failed attempt: return ``(False, cur, [])`` so the caller resamples.
    """
    frames: list = []
    try:
        controller.reset(cur, controller.sample_parameters(cur, rng))
    except TrajectorySamplingFailure:
        return False, cur, frames
    nxt = cur
    for _ in range(_MAX_STEPS):
        if controller.terminated():
            break
        try:
            u = controller.step()
        except TrajectorySamplingFailure:
            return False, cur, frames
        nxt = _transition(sim, nxt, u)
        controller.observe(nxt)
        if render is not None:
            frames.append(render(sim))
    return success_fn(nxt), nxt, frames


def refine_skeleton_v2(
    bundle: V2Bundle,
    x0,
    store_order: list[tuple[str, str]],
    seed: int,
    attempts_per_step: int = _ATTEMPTS_PER_STEP,
    render: Optional[Callable] = None,
    max_seconds: Optional[float] = None,
) -> tuple[bool, object, list, str]:
    """Manual multi-object rollout with per-step resampling.

    Returns ``(ok, final_state, frames, note)``.     ``render`` (optional) is a ``sim ->
    frame`` callable; when given, frames from the SUCCESSFUL     attempts are
    concatenated (failed attempts' frames are discarded) for a clean demo.
    ``max_seconds`` (optional) caps the total wall-clock: checked before each pick/place
    attempt, it aborts with ``note="timeout"`` (a capped refinement counts as unsolved).
    Default ``None`` keeps the uncapped behavior (oracle certifier + tests unaffected).
    """
    sim, section_infos, abstractor = bundle.sim, bundle.section_infos, bundle.abstractor
    cur = x0
    all_frames: list = []
    start = time.perf_counter()
    if render is not None:
        all_frames.append(render(sim))

    def _timed_out() -> bool:
        return max_seconds is not None and time.perf_counter() - start >= max_seconds

    for step_idx, (obj_name, sec_key) in enumerate(store_order):
        # --- Pick (front grasp) ---
        ok = False
        for a in range(attempts_per_step):
            if _timed_out():
                return False, cur, all_frames, "timeout"
            robot = cur.get_object_from_name("robot")
            obj = cur.get_object_from_name(obj_name)
            rng = np.random.default_rng(_mix(seed, step_idx, 0, a))
            ctrl = RestockFrontPickController([robot, obj], sim)
            ok, nxt, frames = _attempt(
                sim, ctrl, cur, lambda s: s.grasped_object == obj_name, rng, render
            )
            if ok:
                cur, all_frames = nxt, all_frames + frames
                break
        if not ok:
            return False, cur, all_frames, f"pick failed: {obj_name}"

        # --- Place onto the assigned section (continuous x; F2/F3 by real collision) ---
        sec_info = section_infos[sec_key]
        ok = False
        for a in range(attempts_per_step):
            if _timed_out():
                return False, cur, all_frames, "timeout"
            robot = cur.get_object_from_name("robot")
            obj = cur.get_object_from_name(obj_name)
            rng = np.random.default_rng(_mix(seed, step_idx, 1, a))
            ctrl = SectionFrontPlaceController([robot, obj], sim, sec_info)
            ok, nxt, frames = _attempt(
                sim, ctrl, cur, lambda s: _stored(abstractor, s, obj_name), rng, render
            )
            if ok:
                cur, all_frames = nxt, all_frames + frames
                break
        if not ok:
            return False, cur, all_frames, f"place failed: {obj_name} -> {sec_key}"

    return True, cur, all_frames, "ok"


def certify_problem(
    bundle: V2Bundle,
    problem_id: int,
    attempts_per_step: int = _ATTEMPTS_PER_STEP,
    render: Optional[Callable] = None,
) -> tuple[OracleResultV2, list]:
    """Reset the bundle's sim to ``problem_id`` and certify the v2 oracle skeleton on
    it."""
    x0, _ = bundle.sim.reset(seed=problem_id)
    assignment = solve_assignment_v2(bundle.section_infos, bundle.goal_names)
    store_order = build_skeleton_v2(x0, assignment)
    ok, _final, frames, note = refine_skeleton_v2(
        bundle,
        x0,
        store_order,
        seed=problem_id,
        attempts_per_step=attempts_per_step,
        render=render,
    )
    return (
        OracleResultV2(
            stratum=_stratum_of_bundle(bundle),
            problem_id=problem_id,
            certified_feasible=ok,
            plan_len=len(store_order),
            note=note,
        ),
        frames,
    )


def _stratum_of_bundle(bundle: V2Bundle) -> int:
    """Recover the stratum from the goal-object counts (small + tall) — bundles don't
    store it."""
    n_small = sum(1 for n in bundle.goal_names if n.startswith(_CUBE_PREFIX))
    n_tall = sum(1 for n in bundle.goal_names if n.startswith(_TALL_PREFIX))
    from .generator import STRATA

    for s, (ns, nt, _, _) in STRATA.items():
        if (ns, nt) == (n_small, n_tall):
            return s
    return -1


def certify_stratum(
    stratum: int, n_problems: int, attempts_per_step: int = _ATTEMPTS_PER_STEP
) -> list[OracleResultV2]:
    """Certify ``n_problems`` sampled problems of ``stratum`` (seeds ``0..n-1``), one
    sim reused."""
    bundle = build_v2_bundle(stratum)
    return [
        certify_problem(bundle, pid, attempts_per_step)[0] for pid in range(n_problems)
    ]
