"""Eager-validity tables + action-cost penalty for kinematic Restock3D (no-clutter v1).

The abstract-plan search (A*+hff) is geometry-blind by design: ``Place`` has no ``Clear``
precondition, so hff cannot tell an over-assigning / tall-into-short skeleton from a
feasible one and ranks by plan length alone. This module supplies a small, observable
**eager-validity** signal -- the same facts the refiner will check, evaluated at the
initial state -- folded into state-dependent action costs so the informed A*
(:class:`eager_search.EagerValidityPlanGenerator`) surfaces feasible skeletons early. It
reads nothing beyond region geometry (section = tall/short) and the goal object names;
it is a model component, not an oracle, and it never prunes (penalties are
large-but-finite, so tall->short skeletons stay in the pool as F3 evidence).

No-clutter v1 specialisation (guide ``docs/restock3d_eager_heuristic_guide.md`` §4):
regions are single-object (``slots[R] = 1``) and there is no grasp clutter
(``blockers[o] = {}``), so the whole signal is a **Place-only** penalty -- T1 tall->short
(F3), T2 region-occupied (F2), T3 cube squats a still-needed tall-section region -- and
``fits(o, R) = (o is a cube) or (R is a tall-section region)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundOperator

from .region_geometry import RegionInfo

_TALL_SECTION = 0  # RegionInfo.shelf value for the tall (bottom) section
_TALL_OBJECT_PREFIX = "block_goal"  # goal tall blocks; cubes are "cube_goal"

# Reach-over corridor (fully-lateral layout, decisions/07 2026-08-17). The front grasp reaches NORTH
# over anything nearer than the target, so a goal A obstructs another goal B's front-pick when A is
# SOUTH of B within a lateral corridor -- and (calibrated by MP sweep) only when a TALL block is
# involved: a cube-over-cube reach clears (the low cube grasp passes over a short south cube), while a
# tall block directly in-line blocks a cube target, and any nearer object blocks a tall-block target
# (its high 45deg reach sweeps the corridor). B must then be picked AFTER A is cleared (south-to-north).
_REACH_LATERAL = 0.12  # |A.x - B.x| below which a south object is in B's reach corridor
_REACH_Y_MARGIN = (
    0.03  # A must be at least this far SOUTH of B to count (else side-by-side)
)


@dataclass(frozen=True)
class EagerTables:
    """Static, per-problem validity facts consumed by the penalty and the feasibility
    classifier."""

    tall_regions: frozenset[str]
    short_regions: frozenset[str]
    tall_goal: frozenset[
        str
    ]  # goal objects that only fit the tall section (the tall blocks)
    footprint: dict[str, float] = field(
        default_factory=dict
    )  # reserved for T4 (λ_o); unused at o=0
    # goal object name -> the movable clutter that obstructs its top-down grasp (F1). Empty under the
    # unified front grasp (retired). Computed from ``grasp_blockers`` on the sim (build_tables sim=...).
    blockers: dict[str, frozenset[str]] = field(default_factory=dict)
    # goal object name -> the OTHER goals south of it in the reach corridor (reach-over). B is
    # front-pickable only after every reach_blockers[B] is cleared (picked). Computed geometrically
    # from the initial object positions (build_tables state=...).
    reach_blockers: dict[str, frozenset[str]] = field(default_factory=dict)

    def fits(self, obj_name: str, region_name: str) -> bool:
        """A cube fits any region; a tall block fits only a tall-section region."""
        if obj_name in self.tall_goal:
            return region_name in self.tall_regions
        return True


@dataclass(frozen=True)
class EagerWeights:
    """Coarse penalty weights (guide §5).

    Deliberately not tuned; only ``h`` / K may move.
    """

    h: float = (
        50.0  # T1 height: tall→short, provable dead end (must dominate plan length)
    )
    c: float = 8.0  # T2 capacity: region already occupied (single-object regions)
    r: float = 8.0  # T3 reservation: cube squats a still-needed tall-section region
    o: float = 0.0  # T4 crowding (soft); off — single-object regions make it degenerate
    b: float = (
        8.0  # T5 blockers: pick a goal whose F1 clutter is not yet relocated (> 2-step
    )
    #                relocation cost, so the eager order prefers relocating over eating the F1)


def build_tables(
    region_infos: dict[str, RegionInfo],
    goal_object_names: Iterable[str],
    sim=None,
    state=None,
) -> EagerTables:
    """Derive the eager tables from region geometry (section) + the goal object names.

    When both ``sim`` and ``state`` (the problem's initial state) are given, the F1
    ``blockers`` map is computed via ``grasp_blockers`` -- the SAME probe the refiner
    uses -- so the eager penalty / feasibility exactly match what fails refinement.
    Omitting them keeps the no-clutter behaviour (``blockers={}``) byte-identical.
    ``grasp_blockers`` sets ``sim`` to ``state`` itself, so the models' scratch sim need
    not be pre-positioned.
    """
    goal_names = list(goal_object_names)
    tall_regions = frozenset(
        name for name, info in region_infos.items() if info.shelf == _TALL_SECTION
    )
    short_regions = frozenset(
        name for name, info in region_infos.items() if info.shelf != _TALL_SECTION
    )
    tall_goal = frozenset(n for n in goal_names if n.startswith(_TALL_OBJECT_PREFIX))
    blockers: dict[str, frozenset[str]] = {}
    if sim is not None and state is not None:
        from .instrumented_refiner import (  # local import: avoid import cycle
            grasp_blockers,
        )

        names_present = set(state.get_object_names())
        for goal in goal_names:
            if goal not in names_present:
                continue
            gb, _reach = grasp_blockers(sim, goal, state)
            if gb:
                blockers[goal] = frozenset(gb)

    # Reach-over: geometric, from the initial object positions (needs ``state``, not ``sim``). A goal
    # A obstructs B's front-pick reach when A is south of B in the lateral corridor and a tall block is
    # involved (a cube-over-cube reach clears). See the corridor constants above.
    reach_blockers: dict[str, frozenset[str]] = {}
    if state is not None:
        pos = {
            n: state.get_object_pose(n).position
            for n in goal_names
            if n in set(state.get_object_names())
        }
        for b in goal_names:
            if b not in pos:
                continue
            bx, by = pos[b][0], pos[b][1]
            b_tall = b in tall_goal
            rb = frozenset(
                a
                for a in goal_names
                if a != b
                and a in pos
                and pos[a][1] < by - _REACH_Y_MARGIN
                and abs(pos[a][0] - bx) < _REACH_LATERAL
                and (b_tall or a in tall_goal)
            )
            if rb:
                reach_blockers[b] = rb
    return EagerTables(
        tall_regions,
        short_regions,
        tall_goal,
        blockers=blockers,
        reach_blockers=reach_blockers,
    )


def _regions_occupied(state: RelationalAbstractState) -> set[str]:
    """Region names holding an object in ``state`` (from ``InRegion(obj, region)``
    atoms)."""
    return {
        atom.objects[1].name
        for atom in state.atoms
        if atom.predicate.name == "InRegion"
    }


def _objects_stored(state: RelationalAbstractState) -> set[str]:
    """Object names already ``Stored`` in ``state``."""
    return {
        atom.objects[0].name for atom in state.atoms if atom.predicate.name == "Stored"
    }


def make_penalty(
    tables: EagerTables, weights: EagerWeights
) -> Callable[[GroundOperator, RelationalAbstractState], float]:
    """Return ``penalty(action, pre_state)`` for the eager A* g-cost.

    Keyed on the **pre-state** the action is applied in (so "region already occupied"
    and "tall demand remaining" read the state before this Place adds its own effects).
    ``Pick`` is free (no clutter in v1). See the module docstring / guide §5 for the
    term semantics.
    """

    def penalty(action: GroundOperator, pre_state: RelationalAbstractState) -> float:
        # T5 — pick a goal whose F1 clutter is still OnFloor (grasp obstructed). Penalising the
        # direct pick makes the eager order prefer relocating the clutter first (Pick+PlaceBuffer,
        # ~2 steps) over eating the F1 -- b > 2 * step-cost. Inert where blockers is empty (r0/r2).
        if action.name == "pick":
            # T5 (F1 clutter, retired/empty) + T6 (reach-over): both penalise picking a goal while a
            # blocker of it is still OnFloor -- the F1 clutter must be relocated first, a reach-over
            # goal must be stored first; either way pick the blocker before this goal (south-to-north).
            name = action.parameters[1].name
            blk = tables.blockers.get(name, frozenset()) | tables.reach_blockers.get(
                name, frozenset()
            )
            if not blk:
                return 0.0
            on_floor = {
                atom.objects[0].name
                for atom in pre_state.atoms
                if atom.predicate.name == "OnFloor"
            }
            return weights.b * sum(1 for c in blk if c in on_floor)
        if action.name != "place":
            return 0.0
        obj_name = action.parameters[1].name
        region_name = action.parameters[2].name
        p = 0.0
        # T1 — tall block into a short-section region: provable dead end (F3).
        if not tables.fits(obj_name, region_name):
            p += weights.h
        # T2 — region already holds an object (single-object capacity, F2).
        if region_name in _regions_occupied(pre_state):
            p += weights.c
        # T3 — a cube taking a tall-section region while tall blocks still need one.
        if obj_name not in tables.tall_goal and region_name in tables.tall_regions:
            stored = _objects_stored(pre_state)
            demand = sum(1 for t in tables.tall_goal if t not in stored)
            occupied = _regions_occupied(pre_state)
            free_tall = sum(1 for r in tables.tall_regions if r not in occupied)
            if free_tall - 1 < demand:
                p += weights.r
        # T4 (λ_o) — soft crowding; off in v1 (single-object regions).
        if weights.o and tables.footprint:
            fp = tables.footprint.get(obj_name, 1.0)
            fp_max = max(tables.footprint.values())
            load = 1.0 if region_name in _regions_occupied(pre_state) else 0.0
            p += weights.o * load * fp / fp_max
        return p

    return penalty


def is_feasible_skeleton(
    action_plan: Iterable[GroundOperator], tables: EagerTables
) -> bool:
    """Classify feasible **without refining** and **in order**: no tall->short (F3), no
    region reused (F2), and every goal picked only after its blockers (F1 clutter +
    reach-over goals) are cleared off the floor.

    F2/F3 are **exact** (real collisions at the region rest pose). The **reach-over**
    gate is a *conservative geometric proxy* (``reach_blockers`` from ``build_tables``):
    the true blocking is cumulative/depth-dependent, so requiring every south-corridor
    object cleared before a target may over-forbid a lone-blocker case (a possible false
    negative) -- but it is safe (the south-to-north order always satisfies it) and
    captures the real multi-object reach-over that fails refinement. A blocker is
    "cleared" once the object is picked (Holding, off its floor spot), so tracking picks
    in order makes a far-before-near skeleton infeasible and a nearest-first skeleton
    feasible.
    """
    occupied: set[str] = set()
    cleared: set[str] = set()  # objects already picked (off the floor)
    for action in action_plan:
        if action.name == "pick":
            obj_name = action.parameters[1].name
            # F1 clutter (relocate first) + reach-over (store the nearer goal first). A pick is
            # infeasible if any blocker of this goal has not yet been picked off the floor.
            blk = tables.blockers.get(
                obj_name, frozenset()
            ) | tables.reach_blockers.get(obj_name, frozenset())
            if not blk.issubset(cleared):
                return False
            cleared.add(obj_name)
            continue
        if action.name != "place":  # place_buffer / anything else: no region constraint
            continue
        obj_name = action.parameters[1].name
        region_name = action.parameters[2].name
        if not tables.fits(obj_name, region_name):  # tall→short (F3)
            return False
        if region_name in occupied:  # over-assignment (F2)
            return False
        occupied.add(region_name)
    return True
