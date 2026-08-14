"""Symbolic feasibility of a Restock3D skeleton — the data feasibility + evidence source (DD-7).

A skeleton's refinement feasibility is fully determined by its assignment (which object goes to
which region, in what order) plus object heights and region cells — none of which needs physics.
:func:`evaluate_skeleton` walks the skeleton, applies the shared :func:`geometry.place_gate` at
every place against the *predicted* abstract state, and returns whether it refines and, if not,
the ``refiner_metadata["failures"]``-shaped record for the first doomed step.

This is deterministic and exact (``exhausted=True``, never a budget cut), so F3's
``proves_failure()`` holds and F2 names the self-placed residents. The physics sampler in
``instrumented_refiner`` is used only for the demo and as a physical spot-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundOperator, ObjectCentricState

from .geometry import place_gate
from .region_geometry import RegionInfo

_NUM_ATTEMPTS = 5  # the per-step sampling budget an exhausted query is charged against


@dataclass(frozen=True)
class SkeletonVerdict:
    """Result of evaluating one skeleton."""

    feasible: bool
    failure: Optional[dict]  # a refiner_metadata["failures"] record, or None if feasible
    family: Optional[str]  # "F2" | "F3" | None (diagnostic)


def object_dims(state: ObjectCentricState, cube_type) -> dict[str, tuple[float, float]]:
    """``{cube_name: (bb_x, bb_z)}`` — static object footprint width and height."""
    return {
        c.name: (float(state.get(c, "bb_x")), float(state.get(c, "bb_z")))
        for c in state.get_objects(cube_type)
    }


def _residents(
    state: RelationalAbstractState, region_name: str, placed_name: str
) -> tuple[str, ...]:
    """Objects predicted ``InRegion(region_name)`` before this place (excluding the placed one)."""
    return tuple(
        sorted(
            atom.objects[0].name
            for atom in state.atoms
            if atom.predicate.name == "InRegion"
            and atom.objects[1].name == region_name
            and atom.objects[0].name != placed_name
        )
    )


def evaluate_skeleton(
    state_plan: Sequence[RelationalAbstractState],
    action_plan: Sequence[GroundOperator],
    region_infos: dict[str, RegionInfo],
    obj_dims: dict[str, tuple[float, float]],
    num_attempts: int = _NUM_ATTEMPTS,
) -> SkeletonVerdict:
    """Symbolic gate walk. Returns feasibility + the first doomed step's failure record."""
    for k, a in enumerate(action_plan):
        if a.name != "place":
            continue
        _, obj, region = a.parameters
        info = region_infos.get(region.name)
        if info is None:
            continue
        bb_x, bb_z = obj_dims.get(obj.name, (0.04, 0.04))
        residents = _residents(state_plan[k], region.name, obj.name)
        family, culprits = place_gate(info, bb_z, 0.5 * bb_x, residents)
        if family is not None:
            failure = {
                "step_index": int(k),
                "schema": "place",
                "args": [p.name for p in a.parameters],
                "culprits": list(culprits),
                "n_step": int(num_attempts),
                "exhausted": True,
                "budget_exhausted": False,
                "dev_added": None,
                "dev_deleted": None,
            }
            return SkeletonVerdict(False, failure, family)
    return SkeletonVerdict(True, None, None)
