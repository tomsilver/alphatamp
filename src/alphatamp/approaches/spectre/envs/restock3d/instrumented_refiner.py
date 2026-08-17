"""Kinematic Restock3D refinement with real-collision failure recording.

Unlike the MuJoCo build (which gated feasibility with a hand-written geometric ``place_gate``), the
kinematic refiner **does not gate**: it runs the real pick/place controllers, which fail motion
planning when no collision-free solution exists (the base env also reverts colliding moves). A
candidate genuinely fails by real PyBullet collision. On each rejection the recorder runs a **real
collision probe** (``getClosestPoints`` / ``check_body_collisions``) purely to *attribute* the
failure — it never changes the accept/reject decision (observation-only):

* **F1 grasp obstruction** (pick) — the arm at the grasp IK collides with adjacent floor clutter;
  those blockers are class-1 *pre-existing* culprits.
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
    front_grasp_transform,
)
from .region_geometry import RegionInfo

_PROBE_LIFT = (
    0.006  # lift the held object this far clear of the surface before the F3 probe
)
_COLLISION_MARGIN = 1e-3


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
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._sim = sim
        self._region_infos = region_infos
        self._robot_name = robot_name
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
            if a.name == "place":
                return self._probe_place(state, a)
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

    def _probe_pick(
        self, state: ObjectCentricState, a: GroundOperator
    ) -> tuple[tuple[str, ...], str]:
        # Single source of truth for F1 blockers (see module-level ``grasp_blockers``), so the
        # refiner probe, the eager blockers table and the generator blocking graph all agree.
        blockers, _reachable = grasp_blockers(self._sim, a.parameters[1].name, state)
        return blockers, "F1"


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
) -> RestockRecordingSampler:
    """Construct the real-collision recording sampler."""
    return RestockRecordingSampler(
        controller_generator=controller_generator,
        transition_function=transition_function,
        state_abstractor=state_abstractor,
        max_trajectory_steps=max_trajectory_steps,
        sim=sim,
        region_infos=region_infos,
        robot_name=robot_name,
    )
