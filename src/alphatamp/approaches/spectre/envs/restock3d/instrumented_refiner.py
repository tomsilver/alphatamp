"""Kinematic Restock3D refinement with real-collision failure recording.

Unlike the MuJoCo build (which gated feasibility with a hand-written geometric ``place_gate``), the
kinematic refiner **does not gate**: it runs the real pick/place controllers, which fail motion
planning when no collision-free solution exists (the base env also reverts colliding moves). A
candidate genuinely fails by real PyBullet collision. On each rejection the recorder runs a **real
collision probe** (``getClosestPoints`` / ``check_body_collisions``) purely to *attribute* the
failure — it never changes the accept/reject decision (observation-only):

* **F1 grasp obstruction** (pick) — the arm at the grasp IK collides with a floor neighbour. Retired
  under the unified front grasp (a neighbour never touches the arm at the front-grasp config), but the
  probe is kept wired; those blockers are class-1 *pre-existing* culprits.
* **F4 reach-over** (pick) — the grasp is reachable at the final config but a nearer object blocks the
  diagonal approach path (invisible to the F1 final-config probe); attributed geometrically by
  ``reach_over_culprits`` to the un-cleared south blockers — class-1, actionable, feeding coverage with
  the correct polarity (decisions/07 2026-08-17).
* **F2 crowding** (place) — the held block, at the region's resting pose, collides with a resident
  the plan already placed there; those residents are class-1 *self-inflicted* culprits.
* **F3 height mismatch** (place) — the held block, lifted just clear of the surface, collides only
  with the shelf board above (no movable); culprit-free + exhausted, so it ``proves_failure()``.

``failure_metadata`` emits the canonical ``refiner_metadata["failures"]`` payload the env-agnostic
SPECTRE downstream consumes (schema shared with ``envs/shelf3d/instrumented_refiner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pybullet as p
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractState, TransitionFailure
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)
from relational_structs import GroundAtom, GroundOperator, ObjectCentricState

from .place_controller import (
    _FRONT_GRASP_MIN_HALF_Z,
    front_grasp_transform,
)
from .region_geometry import RegionInfo

_PROBE_LIFT = (
    0.006  # lift the held object this far clear of the surface before the F3 probe
)
_COLLISION_MARGIN = 1e-3
# v2 continuous packing: an already-stored object is a "resident" of a section iff its
# underside sits within this tolerance of that section's placement surface. Distinguishes
# section residents (bottom ~ surface_z, ~0.29+) from floor objects (bottom ~ 0) and from
# residents of the *other* section (a different surface_z).
_RESIDENT_Z_TOL = 0.05

# Reach-over corridor (fully-lateral layout, decisions/07 2026-08-17). The front grasp reaches NORTH
# over anything nearer than the target, so object A obstructs B's front-pick when A is SOUTH of B in a
# lateral corridor with a tall block involved (a cube-over-cube reach clears; MP-calibrated). These
# constants + ``_blocks_reach`` are the single source shared by the eager ``reach_blockers`` table and
# the refiner's reach-over culprit attribution (``reach_over_culprits`` / ``_probe_pick``).
REACH_LATERAL = 0.12  # |A.x - B.x| below which a south object is in B's reach corridor
REACH_Y_MARGIN = (
    0.03  # A must be at least this far SOUTH of B to count (else side-by-side)
)
_FLOOR_Z_MAX = (
    0.2  # an object with centre-z below this is still on the floor (not shelf-stored)
)


def _blocks_reach(a_pos, a_tall: bool, b_pos, b_tall: bool) -> bool:
    """Whether object A (world ``a_pos``, tall iff ``a_tall``) obstructs object B's
    (``b_pos``, ``b_tall``) front-pick reach -- A south of B in the lateral corridor, a
    tall block involved."""
    return (
        a_pos[1] < b_pos[1] - REACH_Y_MARGIN
        and abs(a_pos[0] - b_pos[0]) < REACH_LATERAL
        and (a_tall or b_tall)
    )


def grasp_blockers(
    sim, obj_name: str, state: ObjectCentricState
) -> tuple[tuple[str, ...], bool]:
    """Names of movables whose bodies the arm collides with at ``obj_name``'s grasp IK
    -- the class-1 F1 culprits obstructing THIS object's grasp.

    The grasp matches what the controller would actually use -- the **unified front
    grasp** for every object (cube or tall block; the front grip height adapts per
    object height inside ``front_grasp_transform``), so the blockers named are the ones
    that really obstruct the grasp. Returns ``(blockers, reachable)``: ``reachable`` is
    False when the grasp IK itself fails (the object is unreachable even ignoring
    clutter -> blockers unknown, empty).

    This is the single source of truth shared by the refiner's F1 probe
    (:class:`_probe_pick`), the eager-heuristic blockers table
    (``eager_tables.build_tables``) and the generator's blocking graph, so all three
    agree on exactly what blocks a grasp.
    """
    sim.set_state(state.copy())
    pcid = sim.physics_client_id
    target_pose = state.get_object_pose(obj_name)
    half_z = float(state.get(state.get_object_from_name(obj_name), "half_extent_z"))
    grasp_tf = front_grasp_transform(half_z)
    ee_pose = multiply_poses(target_pose, grasp_tf)
    try:
        joints = inverse_kinematics(
            sim.robot.arm, ee_pose, validate=False, set_joints=True
        )
        sim.robot.arm.set_joints(joints)
    except InverseKinematicsError:
        return (), False  # unreachable grasp; blockers unknown but pick-side
    p.performCollisionDetection(physicsClientId=pcid)
    target_id = sim._object_name_to_pybullet_id(obj_name)
    blockers = sorted(
        name
        for name in sim.movable_names()
        if name != obj_name
        and sim._object_name_to_pybullet_id(name) != target_id
        and check_body_collisions(
            sim.robot.arm.robot_id,
            sim._object_name_to_pybullet_id(name),
            pcid,
            distance_threshold=_COLLISION_MARGIN,
            perform_collision_detection=False,
        )
    )
    return tuple(blockers), True


def reach_over_culprits(
    sim, obj_name: str, state: ObjectCentricState
) -> tuple[str, ...]:
    """Movables still on the floor that obstruct ``obj_name``'s front-pick **reach** --
    the class-1 culprits of a reach-over pick failure (what a south-to-north order must
    clear first).

    The front grasp reaches north over anything nearer than the target, so a grasp can be feasible at
    the final config (``grasp_blockers`` empty) yet fail motion planning because a nearer object is in
    the diagonal approach path. That approach-path collision is invisible to the final-config
    ``grasp_blockers`` probe, so it is attributed here **geometrically** -- the same corridor rule the
    eager ``reach_blockers`` table uses (``_blocks_reach``), evaluated on the CURRENT (failure) state so
    only un-cleared floor blockers are named. These culprits are movable and actionable (each is a goal
    that gets stored, or a relocatable clutter), so they feed the class-1 coverage channel with the
    correct polarity: a candidate that stores/relocates the blocker before re-picking the target covers
    it (decisions/07 2026-08-17).
    """
    b = state.get_object_from_name(obj_name)
    b_pos = state.get_object_pose(obj_name).position
    b_tall = float(state.get(b, "half_extent_z")) >= _FRONT_GRASP_MIN_HALF_Z
    out: list[str] = []
    for name in sim.movable_names():
        if name == obj_name:
            continue
        pos = state.get_object_pose(name).position
        if pos[2] >= _FLOOR_Z_MAX:  # already shelf-stored -> not a floor blocker
            continue
        a_tall = (
            float(state.get(state.get_object_from_name(name), "half_extent_z"))
            >= _FRONT_GRASP_MIN_HALF_Z
        )
        if _blocks_reach(pos, a_tall, b_pos, b_tall):
            out.append(name)
    return tuple(sorted(out))


@dataclass(frozen=True)
class _Rejection:
    """One rejected sample: the step, its class-2 deviation, any class-1 culprits, the
    family."""

    step: GroundOperator
    expected: frozenset[GroundAtom]
    achieved: Optional[frozenset[GroundAtom]]
    culprits: tuple[str, ...]
    family: str  # "F1" | "F2" | "F3" | "C2" — diagnostic, not serialized


class RestockRecordingSampler(ParameterizedControllerTrajectorySampler):
    """Real-collision trajectory sampler that records every rejection with its blamed
    objects."""

    def __init__(
        self,
        *args: object,
        sim,
        region_infos: dict[str, RegionInfo],
        robot_name: str = "robot",
        section_height_cutoffs: Optional[dict[str, float]] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._sim = sim
        self._region_infos = region_infos
        self._robot_name = robot_name
        # v3 arm-insertion cutoffs {section_key: max full-height}; None -> v2 behaviour unchanged.
        self._section_cutoffs = section_height_cutoffs
        self.rejections: list[_Rejection] = []

    def clear(self) -> None:
        self.rejections.clear()

    # -- the real refinement rollout --------------------------------------
    def __call__(  # type: ignore[override]
        self,
        x: ObjectCentricState,
        s: RelationalAbstractState,
        a: GroundOperator,
        ns: RelationalAbstractState,
        bpg: BilevelPlanningGraph,
        rng: np.random.Generator,
    ) -> tuple[list, list]:
        controller = self._controller_generator(a)
        params = controller.sample_parameters(x, rng)
        try:
            controller.reset(x, params)
        except (
            BaseException
        ):  # noqa: BLE001  (TrajectorySamplingFailure is BaseException)
            self._record(x, a, ns, None)
            raise TrajectorySamplingFailure()  # pylint: disable=raise-missing-from

        x_traj: list = [x]
        u_traj: list = []
        cur = x
        for _ in range(self._max_trajectory_steps):
            if controller.terminated():
                break
            try:
                u = controller.step()
            except BaseException:  # noqa: BLE001
                self._record(cur, a, ns, None)
                raise TrajectorySamplingFailure()  # pylint: disable=raise-missing-from
            try:
                nx = self._transition_function(cur, u)
            except TransitionFailure:
                break
            controller.observe(nx)
            x_traj.append(nx)
            u_traj.append(u)
            bpg.add_state_node(nx)
            bpg.add_action_edge(cur, u, nx)
            cur = nx

        final_state = x_traj[-1]
        achieved = self._state_abstractor(final_state)
        bpg.add_abstract_state_node(achieved)
        bpg.add_state_abstractor_edge(final_state, achieved)
        if achieved == ns:
            return x_traj, u_traj
        # Rolled out but did not reach the predicted abstract state.
        self._record(final_state, a, ns, achieved)
        raise TrajectorySamplingFailure()

    # -- rejection recording + real-collision attribution -----------------
    def _record(
        self,
        state: ObjectCentricState,
        a: GroundOperator,
        ns: RelationalAbstractState,
        achieved: Optional[RelationalAbstractState],
    ) -> None:
        culprits, family = self._probe(state, a)
        self.rejections.append(
            _Rejection(
                a,
                frozenset(ns.atoms),
                None if achieved is None else frozenset(achieved.atoms),
                culprits,
                family,
            )
        )

    def _probe(
        self, state: ObjectCentricState, a: GroundOperator
    ) -> tuple[tuple[str, ...], str]:
        try:
            if a.name == "place":  # v1 discrete-region place (3-arg)
                return self._probe_place(state, a)
            if a.name in (
                "place_tall",
                "place_short",
            ):  # v2 continuous-section place (2-arg)
                return self._probe_place_v2(state, a)
            if a.name == "pick":
                return self._probe_pick(state, a)
        except BaseException:  # noqa: BLE001  (a probe must never change the decision)
            return (), "C2"
        return (), "C2"

    def _movable_ids(self) -> dict[str, int]:
        return {
            n: self._sim._object_name_to_pybullet_id(n)
            for n in self._sim.movable_names()
        }

    def _probe_place(
        self, state: ObjectCentricState, a: GroundOperator
    ) -> tuple[tuple[str, ...], str]:
        _, obj, region = a.parameters
        info = self._region_infos.get(region.name)
        if info is None:
            return (), "C2"
        self._sim.set_state(state.copy())
        pcid = self._sim.physics_client_id
        held_id = self._sim._object_name_to_pybullet_id(obj.name)
        half_z = float(state.get(state.get_object_from_name(obj.name), "half_extent_z"))
        saved = get_pose(held_id, pcid)
        # Lift the object just clear of the surface at the region centre: bottom above the board,
        # top penetrating the ceiling iff it is too tall (F3); overlapping a resident iff F2.
        probe_pose = Pose(
            (
                info.center_xy[0],
                info.center_xy[1],
                info.surface_z + half_z + _PROBE_LIFT,
            )
        )
        set_pose(held_id, probe_pose, pcid)
        p.performCollisionDetection(physicsClientId=pcid)
        culprits = [
            name
            for name, mid in self._movable_ids().items()
            if name != obj.name
            and check_body_collisions(
                held_id,
                mid,
                pcid,
                distance_threshold=_COLLISION_MARGIN,
                perform_collision_detection=False,
            )
        ]
        # F3: the upright held object at the region rest pose collides the shelf STRUCTURE (a board
        # above a too-short cell, or a wall) and no movable — culprit-free, so it proves_failure().
        shelf_hit = any(
            check_body_collisions(
                held_id,
                sid,
                pcid,
                distance_threshold=_COLLISION_MARGIN,
                perform_collision_detection=False,
            )
            for sid in self._sim.shelf_structure_ids()
        )
        set_pose(held_id, saved, pcid)
        if culprits:
            return tuple(sorted(culprits)), "F2"
        if shelf_hit:
            return (), "F3"
        return (), "C2"

    def _probe_place_v2(
        self, state: ObjectCentricState, a: GroundOperator
    ) -> tuple[tuple[str, ...], str]:
        """V2 continuous-section place attribution (``place_tall``/``place_short``).

        F3 (height) is unchanged from v1 -- the section's ceiling board spans the whole
        band, so a single centre probe still detects a too-tall block. F2 (crowding) is
        redesigned: continuous packing spreads residents across the wide band, so the v1
        single-centre-point overlap misses them. Instead, once the object is confirmed to
        FIT this section height-wise (no shelf hit), a failed place is "no free x remains",
        whose class-1 culprits are the section's **residents** -- the objects the prefix
        already stored on this band, which collectively crowded the placement out.
        """
        _, obj = a.parameters  # v2 place ops are (robot, target); no region arg
        sec_key = "section_0" if a.name == "place_tall" else "section_1"
        info = self._region_infos.get(sec_key)
        if info is None:
            return (), "C2"
        # v3 arm-insertion F3 (Phase 3, decisions/07 2026-08-20): the section's board clearance sits
        # ~0.10 m ABOVE the arm-insertion cutoff, so a block in (cutoff, clearance] FITS under the
        # board (the block-vs-board test below would miss it) yet the arm cannot thread it in and the
        # real rollout fails MP. Attribute it here as a provable, culprit-free F3 -- matching the
        # analytic ``feasibility_v3`` classifier so collection labels and real-refiner records agree
        # (Gate G1). Guarded by ``_section_cutoffs``; None (v2) leaves the old behaviour byte-identical.
        if self._section_cutoffs is not None:
            full_h = 2.0 * float(
                state.get(state.get_object_from_name(obj.name), "half_extent_z")
            )
            cutoff = self._section_cutoffs.get(sec_key)
            if cutoff is not None and full_h > cutoff + 1e-9:
                return (), "F3"
        self._sim.set_state(state.copy())
        pcid = self._sim.physics_client_id
        held_id = self._sim._object_name_to_pybullet_id(obj.name)
        half_z = float(state.get(state.get_object_from_name(obj.name), "half_extent_z"))
        saved = get_pose(held_id, pcid)
        # Centre-of-band probe: bottom on the board, top penetrating the ceiling iff the
        # block is too tall for this section (F3).
        probe_pose = Pose(
            (
                info.center_xy[0],
                info.center_xy[1],
                info.surface_z + half_z + _PROBE_LIFT,
            )
        )
        set_pose(held_id, probe_pose, pcid)
        p.performCollisionDetection(physicsClientId=pcid)
        shelf_hit = any(
            check_body_collisions(
                held_id,
                sid,
                pcid,
                distance_threshold=_COLLISION_MARGIN,
                perform_collision_detection=False,
            )
            for sid in self._sim.shelf_structure_ids()
        )
        set_pose(held_id, saved, pcid)
        if shelf_hit:
            return (
                (),
                "F3",
            )  # too tall for this section -- culprit-free, proves_failure()
        residents = self._section_residents(state, info, exclude=obj.name)
        if residents:
            return tuple(sorted(residents)), "F2"
        return (), "C2"

    def _section_residents(
        self, state: ObjectCentricState, info: RegionInfo, exclude: str
    ) -> list[str]:
        """Movables already resting on ``info``'s placement surface (excluding
        ``exclude``)."""
        out: list[str] = []
        for name in self._sim.movable_names():
            if name == exclude:
                continue
            pose = state.get_object_pose(name)
            half_z = float(state.get(state.get_object_from_name(name), "half_extent_z"))
            bottom = float(pose.position[2]) - half_z
            if abs(bottom - info.surface_z) < _RESIDENT_Z_TOL:
                out.append(name)
        return out

    def _probe_pick(
        self, state: ObjectCentricState, a: GroundOperator
    ) -> tuple[tuple[str, ...], str]:
        # F1 grasp obstruction: a movable collides the arm at the final grasp config (``grasp_blockers``,
        # the single source of truth shared with the eager table + generator). Retired under the front
        # grasp (a floor neighbour never touches the arm there) but kept wired.
        target = a.parameters[1].name
        blockers, _reachable = grasp_blockers(self._sim, target, state)
        if blockers:
            return blockers, "F1"
        # F4 reach-over: the grasp is reachable at the final config but a nearer object blocks the
        # diagonal approach path (invisible to grasp_blockers). Attribute it to the geometric reach
        # blockers so the failure carries class-1 culprits and feeds coverage (decisions/07 2026-08-17).
        reach = reach_over_culprits(self._sim, target, state)
        return reach, "F4"


def _deepest_rejection(
    rejections: Sequence[_Rejection], action_plan: Sequence[GroundOperator]
) -> Optional[tuple[int, _Rejection]]:
    """The rejection at the furthest step the refiner reached (backtracking retries
    shallow)."""
    best: Optional[tuple[int, _Rejection]] = None
    for rej in rejections:
        index = next((j for j, op in enumerate(action_plan) if op == rej.step), None)
        if index is None:
            continue
        if best is None or index > best[0]:  # pylint: disable=unsubscriptable-object
            best = (index, rej)
    return best


def _atom_pairs(atoms: frozenset[GroundAtom]) -> list[list]:
    """``[[predicate, [arg, ...]], ...]`` — picklable, canonicalisable, sorted."""
    return sorted(
        ([atom.predicate.name, [o.name for o in atom.objects]] for atom in atoms),
        key=repr,
    )


def failure_metadata(
    sampler: RestockRecordingSampler,
    action_plan: Sequence[GroundOperator],
    num_sampling_attempts_per_step: int,
    budget_exhausted: bool,
) -> list[dict]:
    """The ``refiner_metadata["failures"]`` payload for one failed candidate.

    A class-1 record (F1 blockers / F2 residents) carries ``culprits`` and no deviation;
    a class-2 record carries the ``dev_added``/``dev_deleted`` deviation; an F3 record
    is culprit-free with an empty deviation, and when the step exhausted without a
    budget cut it ``proves_failure()``. The reduction keeps the deepest reached step.
    """
    deepest = _deepest_rejection(sampler.rejections, action_plan)
    if deepest is None:
        return []
    index, rej = deepest
    n_step = sum(1 for r in sampler.rejections if r.step == rej.step)
    is_class_1 = bool(rej.culprits)
    added = frozenset() if rej.achieved is None else (rej.achieved - rej.expected)
    deleted = frozenset() if rej.achieved is None else (rej.expected - rej.achieved)
    return [
        {
            "step_index": int(index),
            "schema": str(rej.step.name),
            "args": [pp.name for pp in rej.step.parameters],
            "culprits": list(rej.culprits),
            "n_step": int(n_step),
            "exhausted": bool(n_step >= num_sampling_attempts_per_step),
            "budget_exhausted": bool(budget_exhausted),
            "dev_added": None if is_class_1 else _atom_pairs(added),
            "dev_deleted": None if is_class_1 else _atom_pairs(deleted),
        }
    ]


def make_recording_sampler(
    controller_generator: Callable,
    transition_function: Callable,
    state_abstractor: Callable,
    max_trajectory_steps: int,
    sim,
    region_infos: dict[str, RegionInfo],
    robot_name: str = "robot",
    section_height_cutoffs: Optional[dict[str, float]] = None,
) -> RestockRecordingSampler:
    """Construct the real-collision recording sampler.

    ``section_height_cutoffs`` (v3 only) enables the arm-insertion F3 attribution in
    ``_probe_place_v2``; leave None for v2 to keep the block-vs-board behaviour unchanged.
    """
    return RestockRecordingSampler(
        controller_generator=controller_generator,
        transition_function=transition_function,
        state_abstractor=state_abstractor,
        max_trajectory_steps=max_trajectory_steps,
        sim=sim,
        region_infos=region_infos,
        robot_name=robot_name,
        section_height_cutoffs=section_height_cutoffs,
    )
