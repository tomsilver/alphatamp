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


def build_tables(
    region_infos: dict[str, RegionInfo], goal_object_names: Iterable[str]
) -> EagerTables:
    """Derive the eager tables from region geometry (section) + the goal object
    names."""
    tall_regions = frozenset(
        name for name, info in region_infos.items() if info.shelf == _TALL_SECTION
    )
    short_regions = frozenset(
        name for name, info in region_infos.items() if info.shelf != _TALL_SECTION
    )
    tall_goal = frozenset(
        n for n in goal_object_names if n.startswith(_TALL_OBJECT_PREFIX)
    )
    return EagerTables(tall_regions, short_regions, tall_goal)


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
    """Classify feasible **without refining**: no tall->short, no region reused.

    Exact for the no-clutter v1 hardness sources (regions single-object, feasibility =
    real collision at the region rest pose); residual refinement failures are
    continuous-level sampler noise, not abstract-level infeasibility.
    """
    occupied: set[str] = set()
    for action in action_plan:
        if action.name != "place":
            continue
        obj_name = action.parameters[1].name
        region_name = action.parameters[2].name
        if not tables.fits(obj_name, region_name):  # tall→short (F3)
            return False
        if region_name in occupied:  # over-assignment (F2)
            return False
        occupied.add(region_name)
    return True
