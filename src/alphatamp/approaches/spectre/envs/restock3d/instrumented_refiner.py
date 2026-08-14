"""Restock3D refinement with a geometric feasibility gate + failure recording.

The base sampler enforces a *geometric feasibility gate* before rolling out any placement — this
is the env's feasibility model, and it is what makes an infeasible candidate genuinely fail
(rather than letting MuJoCo physics squeeze past, the ShelfObstruct3D failure). Two families in
v1 (F1 grasp obstruction is deferred):

* **F2 over-assignment (self-inflicted, class 1).** ``Place(obj, region)`` where the plan already
  assigned ``region`` its capacity of residents (read from the *predicted* abstract state ``s`` —
  the plan's own intent) is rejected, naming those residents as culprits. Self-inflicted: the
  residents are there because earlier ``Place`` steps of *this* skeleton put them there.
* **F3 height mismatch (culprit-free, exhausted).** ``Place(tall, short_cell)`` where the object
  is taller than the cell clearance has no valid sample — rejected culprit-free, so the record
  ``proves_failure()`` (a clean sampler-exhaustible infeasibility).

The gate is params-independent, so every attempt at a doomed step rejects → the step exhausts →
the candidate fails there. **Observation-only:** the gate lives in the base sampler (the env's
feasibility), and the recording subclass only *keeps* the rejections it would otherwise discard,
so the accept/reject decisions are identical to an uninstrumented gated run. ``failure_metadata``
emits the canonical ``refiner_metadata["failures"]`` payload the env-agnostic SPECTRE downstream
consumes (schema shared with ``envs/shelf3d/instrumented_refiner.py``).
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

from .geometry import place_gate
from .region_geometry import RegionInfo

_HEIGHT_MARGIN = 0.02  # vertical slack the hand needs above a held object (F3 gate)


@dataclass(frozen=True)
class _Rejection:
    """One rejected sample: the step, its class-2 deviation, any class-1 culprits, the
    family."""

    step: GroundOperator
    expected: frozenset[GroundAtom]
    achieved: Optional[
        frozenset[GroundAtom]
    ]  # None if rejected before any successor state
    culprits: tuple[
        str, ...
    ]  # class-1: objects the gate named (F2); empty for F3/class-2
    family: str  # "F2" | "F3" | "C2" (physics deviation) — diagnostic, not serialized


class RestockRecordingSampler(ParameterizedControllerTrajectorySampler):
    """Gated trajectory sampler that records every rejection with its blamed objects.

    Accumulates into :attr:`rejections` across calls (the refiner backtracks);
    :func:`clear` resets between candidates and :func:`failure_metadata` reduces to the
    deepest-step record.
    """

    def __init__(
        self,
        *args: object,
        region_infos: dict[str, RegionInfo],
        robot_name: str,
        height_margin: float = _HEIGHT_MARGIN,
        geometric_place: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._region_infos = region_infos
        self._robot_name = robot_name
        self._height_margin = height_margin
        # Data refiner uses a deterministic geometric place (physics place is flaky as the shelf
        # fills — DD-6). The demo constructs the sampler with geometric_place=False for real
        # physics execution.
        self._geometric_place = geometric_place
        self.rejections: list[_Rejection] = []

    def clear(self) -> None:
        """Drop accumulated rejections — call between candidates."""
        self.rejections.clear()

    # -- the geometric feasibility gate -----------------------------------
    def _gate(
        self,
        x: ObjectCentricState,
        s: RelationalAbstractState,
        a: GroundOperator,
        ns: RelationalAbstractState,
    ) -> Optional[_Rejection]:
        """Reject a placement that violates height (F3) or region capacity (F2)."""
        if a.name != "place":
            return None  # F1 (pick-side grasp obstruction) is deferred
        _, obj, region = a.parameters
        info = self._region_infos.get(region.name)
        if info is None:
            return None
        obj_obj = x.get_object_from_name(obj.name)
        residents = tuple(
            sorted(
                atom.objects[0].name
                for atom in s.atoms
                if atom.predicate.name == "InRegion"
                and atom.objects[1].name == region.name
                and atom.objects[0].name != obj.name
            )
        )
        family, culprits = place_gate(
            info,
            float(x.get(obj_obj, "bb_z")),
            0.5 * float(x.get(obj_obj, "bb_x")),
            residents,
            self._height_margin,
        )
        if family is None:
            return None
        return _Rejection(a, frozenset(ns.atoms), None, culprits, family)

    def _geometric_place_transition(
        self,
        x: ObjectCentricState,
        a: GroundOperator,
        ns: RelationalAbstractState,
        bpg: BilevelPlanningGraph,
    ) -> tuple[list, list]:
        """Deterministic geometric place: teleport the held cube to the region slot,
        open gripper.

        Feasibility is already decided by the gate, so this realizes the successful
        placement without the flaky shelf-insertion motion plan (DD-6). The abstractor
        then reads InRegion + Stored + HandEmpty, matching ``ns``.
        """
        _, obj, region = a.parameters
        info = self._region_infos.get(region.name)
        nx = x.copy()
        obj_o = nx.get_object_from_name(obj.name)
        if info is not None:
            cx, cy = info.center_xy
            nx.set(obj_o, "x", cx)
            nx.set(obj_o, "y", cy)
            nx.set(obj_o, "z", info.surface_z + 0.5 * float(x.get(obj_o, "bb_z")))
        nx.set(nx.get_object_from_name(self._robot_name), "pos_gripper", 0.0)
        u = np.zeros(11, dtype=np.float32)
        bpg.add_state_node(nx)
        bpg.add_action_edge(x, u, nx)
        achieved = self._state_abstractor(nx)
        bpg.add_abstract_state_node(achieved)
        bpg.add_state_abstractor_edge(nx, achieved)
        if achieved == ns:
            return [x, nx], [u]
        self.rejections.append(
            _Rejection(a, frozenset(ns.atoms), frozenset(achieved.atoms), (), "C2")
        )
        raise TrajectorySamplingFailure()

    def __call__(  # type: ignore[override]
        self,
        x: ObjectCentricState,
        s: RelationalAbstractState,
        a: GroundOperator,
        ns: RelationalAbstractState,
        bpg: BilevelPlanningGraph,
        rng: np.random.Generator,
    ) -> tuple[list, list]:
        rejection = self._gate(x, s, a, ns)
        if rejection is not None:
            self.rejections.append(rejection)
            raise TrajectorySamplingFailure()

        if a.name == "place" and self._geometric_place:
            return self._geometric_place_transition(x, a, ns, bpg)

        controller = self._controller_generator(a)
        params = controller.sample_parameters(x, rng)
        try:
            controller.reset(x, params)
        except BaseException:  # pylint: disable=broad-exception-caught
            # Physics/motion-planning failure before any successor state (class 2, no culprit).
            self.rejections.append(_Rejection(a, frozenset(ns.atoms), None, (), "C2"))
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

        # Passed the gate but physics did not reach the predicted state -> class-2 deviation.
        self.rejections.append(
            _Rejection(a, frozenset(ns.atoms), frozenset(achieved.atoms), (), "C2")
        )
        raise TrajectorySamplingFailure()


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

    A class-1 record (F2, culprits named) carries ``culprits`` and no deviation; a
    class-2 record (physics deviation) carries the ``dev_added``/``dev_deleted``
    deviation; an F3 record is culprit-free with an empty deviation (a means failure) —
    and, when the step exhausted without a budget cut, it ``proves_failure()``
    downstream. The reduction is the deepest reached step.
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
            "dev_added": None if is_class_1 else _atom_pairs(added),
            "dev_deleted": None if is_class_1 else _atom_pairs(deleted),
        }
    ]


def make_recording_sampler(
    controller_generator: Callable,
    transition_function: Callable,
    state_abstractor: Callable,
    max_trajectory_steps: int,
    region_infos: dict[str, RegionInfo],
    robot_name: str,
    height_margin: float = _HEIGHT_MARGIN,
    geometric_place: bool = True,
) -> RestockRecordingSampler:
    """Construct the gated recording sampler with the model's region geometry.

    ``geometric_place=True`` (data collection) uses the deterministic geometric place;
    pass ``False`` (demo) for a full physics place rollout.
    """
    return RestockRecordingSampler(
        controller_generator=controller_generator,
        transition_function=transition_function,
        state_abstractor=state_abstractor,
        max_trajectory_steps=max_trajectory_steps,
        region_infos=region_infos,
        robot_name=robot_name,
        height_margin=height_margin,
        geometric_place=geometric_place,
    )
