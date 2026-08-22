"""Refinement that reports *why* it failed, for ShelfObstruct3D — the class-1 evidence.

The unified evidence construction (``unified_evidence.py``) needs, per failed candidate, the
failed step and either (class 1) the objects a validity check named as the reason, or (class 2)
the deviation between the abstract state the candidate predicted and the one the sample actually
produced. Upstream's :class:`BacktrackingRefiner` returns ``Plan | None`` and its sampler raises
a payload-free :class:`TrajectorySamplingFailure`; this module keeps what they throw away.

**ShelfObstruct3D can afford class 1**, unlike StickButton2D. When a ``place(cube, region)``
step fails, we run a geometric obstruction check — the same idea as DD2D's ``grasp_blocker`` —
over the destination region's footprint: any *other* cube physically overlapping where the held
cube must go is named as the **culprit** (a class-1 record, ``deviation=None``). This is a
validity check *we* run and name on, so it is legitimately class 1: the object that blocked the
placement is exactly what a failure-conditioned re-ranker should learn to route around. When no
obstruction is found (a reachability/MP failure with the destination clear), the record falls
back to the class-2 deviation between predicted and achieved, mirroring SB2D.

**Caveat — on ShelfObstruct3D the class-1 obstruction is physically INERT** (measured; ADR /
notebook 2026-08-13). The M2 certifying generator confirmed it: the shelf holds only cubes
≤ 0.07 m wide, so a cube far enough from a region centre to leave it ``Clear`` (offset >
``models._AT_XY_TOL``) overlaps a placed cube by ≤ ~0.03 m, which the placement physics squeezes
past rather than failing (a certified obstructed candidate refined to SUCCESS). This check is
still correct and both channels are verified — it will fire on any env whose geometry *does*
afford a blocking-but-not-occupying obstruction — but ShelfObstruct3D leans class-2 like SB2D, so
class-1 records do not arise here in practice.

**Observation-only.** The recorded labels never change what the refiner decides — the recording
sampler subclasses the same trajectory sampler and only *keeps* the rejections it would otherwise
discard; the accept/reject decision, the transitions, and the abstractions are identical to an
uninstrumented run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractState, TransitionFailure
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from relational_structs import GroundAtom, GroundOperator, ObjectCentricState

_ON_SHELF_Z = 0.3
_OBSTRUCT_MARGIN = 0.01  # slack added to the physical footprint-overlap test
_SAME_SHELF_DZ = (
    0.12  # a culprit must be within this z of the placement (same shelf, not below)
)


@dataclass(frozen=True)
class _Rejection:
    """One rejected sample: the step, how it deviated (class 2), and any named culprit."""

    step: GroundOperator
    expected: frozenset[GroundAtom]
    achieved: Optional[frozenset[GroundAtom]]  # None if reset failed before any state
    culprits: tuple[str, ...]  # class-1: objects the obstruction check named


class RecordingObstructionSampler(ParameterizedControllerTrajectorySampler):
    """Stock trajectory sampler that keeps every rejection and names place obstructions.

    Accumulates into :attr:`rejections` across calls (the refiner backtracks, so one candidate
    yields many); :func:`failure_metadata` reduces them to the deepest-step record.
    """

    def __init__(
        self,
        *args: object,
        region_centers: dict[str, tuple[float, float]],
        shelf_surface_z: float = 0.55,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._region_centers = region_centers
        self._shelf_surface_z = shelf_surface_z
        self.rejections: list[_Rejection] = []

    def clear(self) -> None:
        """Drop accumulated rejections — call between candidates."""
        self.rejections.clear()

    def _place_culprits(
        self, state: ObjectCentricState, a: GroundOperator
    ) -> tuple[str, ...]:
        """Cubes physically obstructing a ``place(cube, region)`` destination.

        The destination region is symbolically ``Clear`` (the planner required it), so any cube
        whose footprint would overlap the placed cube at the region centre is a neighbour
        intruding into the footprint — the class-1 reason the placement cannot be refined. The
        overlap test uses the actual cube half-widths (plus a small margin), so it fires exactly
        when the two cubes physically cannot both occupy their positions.
        """
        if a.name != "place":
            return ()
        placed = a.parameters[1].name
        region = a.parameters[2].name
        if region not in self._region_centers:
            return ()
        dx, dy = self._region_centers[region]
        placed_obj = state.get_object_from_name(placed)
        placed_half = 0.5 * float(state.get(placed_obj, "bb_x"))
        place_z = (
            self._shelf_surface_z + placed_half
        )  # where the placed cube's centre lands
        culprits: list[str] = []
        for cube in state.get_objects(_movable_type(state)):
            if cube.name == placed or state.get(cube, "z") <= _ON_SHELF_Z:
                continue
            # A culprit must be on the *same shelf* as the placement -- a cube that dropped to the
            # shelf below overlaps in xy but not in reality.
            if abs(state.get(cube, "z") - place_z) > _SAME_SHELF_DZ:
                continue
            dist = float(np.hypot(state.get(cube, "x") - dx, state.get(cube, "y") - dy))
            collide = (
                placed_half + 0.5 * float(state.get(cube, "bb_x")) + _OBSTRUCT_MARGIN
            )
            if dist < collide:
                culprits.append(cube.name)
        return tuple(sorted(culprits))

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
        except BaseException:  # pylint: disable=broad-exception-caught
            # Reset failed (e.g. motion planning) before any successor state existed. Name a
            # geometric culprit from the *initial* state if the destination is obstructed.
            self.rejections.append(
                _Rejection(
                    step=a,
                    expected=frozenset(ns.atoms),
                    achieved=None,
                    culprits=self._place_culprits(x, a),
                )
            )
            raise TrajectorySamplingFailure()  # pylint: disable=raise-missing-from

        x_traj: list = [x]
        u_traj: list = []
        cur = x
        for _ in range(self._max_trajectory_steps):
            if controller.terminated():
                break
            u = controller.step()
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

        self.rejections.append(
            _Rejection(
                step=a,
                expected=frozenset(ns.atoms),
                achieved=frozenset(achieved.atoms),
                culprits=self._place_culprits(final_state, a),
            )
        )
        raise TrajectorySamplingFailure()


def _movable_type(state: ObjectCentricState):
    from kinder.envs.dynamic3d.object_types import MujocoMovableObjectType

    del state
    return MujocoMovableObjectType


def _deepest_rejection(
    rejections: Sequence[_Rejection], action_plan: Sequence[GroundOperator]
) -> Optional[tuple[int, _Rejection]]:
    """The rejection at the furthest step the refiner reached (backtracking retries shallow)."""
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
    sampler: RecordingObstructionSampler,
    action_plan: Sequence[GroundOperator],
    num_sampling_attempts_per_step: int,
    budget_exhausted: bool,
) -> list[dict]:
    """The ``refiner_metadata["failures"]`` payload for one failed candidate.

    A class-1 record (obstruction named) carries ``culprits`` and no deviation; a class-2 record
    carries the ``dev_added``/``dev_deleted`` deviation. The reduction is the deepest reached
    step, so the serialized record describes the point the candidate actually got stuck at.
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
            "args": [p.name for p in rej.step.parameters],
            "culprits": list(rej.culprits),
            "n_step": int(n_step),
            "exhausted": bool(n_step >= num_sampling_attempts_per_step),
            "budget_exhausted": bool(budget_exhausted),
            # Class 1 (obstruction named) stores no deviation; class 2 stores it. Emitting one
            # or the other keeps each record in a single class for the consumer.
            "dev_added": None if is_class_1 else _atom_pairs(added),
            "dev_deleted": None if is_class_1 else _atom_pairs(deleted),
        }
    ]


def make_recording_sampler(
    controller_generator: Callable,
    transition_function: Callable,
    state_abstractor: Callable,
    max_trajectory_steps: int,
    region_centers: dict[str, tuple[float, float]],
    shelf_surface_z: float = 0.55,
) -> RecordingObstructionSampler:
    """Construct the recording sampler with the model's region centres for culprit naming."""
    return RecordingObstructionSampler(
        controller_generator=controller_generator,
        transition_function=transition_function,
        state_abstractor=state_abstractor,
        max_trajectory_steps=max_trajectory_steps,
        region_centers=region_centers,
        shelf_surface_z=shelf_surface_z,
    )
