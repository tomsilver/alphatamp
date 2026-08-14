"""Geometric feasibility primitives for Restock3D's sampler-level gate.

These decide refinement feasibility *by construction* (the DD2D lesson) rather than leaving it to
MuJoCo physics — which is what avoided ShelfObstruct3D's inert-obstruction failure. The gate that
calls them lives in ``instrumented_refiner``; physics only transitions samples the gate accepts.

v1 covers F2 (region over-assignment) + F3 (height mismatch). F1 (top-down grasp obstruction) and
its ``grasp_cfree_3d`` swept-volume check are deferred (see the ADRs); their hook is noted below.
"""

from __future__ import annotations

from typing import Optional

from .region_geometry import RegionInfo


def height_ok(obj_height: float, cell_clearance: float, margin: float = 0.02) -> bool:
    """Does an object of height ``obj_height`` fit under a cell of clearance ``cell_clearance``?

    Written against the object's own height (proposal §3.2 constraint 3), so the F3 infeasibility
    is robust to how faithfully any simulator models the gripper. ``margin`` is the small vertical
    slack the hand needs above the object during a horizontal insertion.
    """
    return obj_height + margin <= cell_clearance


def region_capacity(
    region: RegionInfo, obj_half_x: float, slot_margin: float = 0.025
) -> int:
    """Number of single-object slots along a region's front strip.

    ``floor(strip_width / (obj_width + 2*slot_margin))``, at least 1. v1 authors narrow strips so
    this is 1 (single-object regions, DD-2); a wider strip becomes the multi-slot capacity knob.
    """
    strip_width = 2.0 * region.half_xy[0]
    obj_width = 2.0 * obj_half_x
    return max(1, int(strip_width // (obj_width + 2.0 * slot_margin)))


def place_gate(
    region: RegionInfo,
    obj_height: float,
    obj_half_x: float,
    residents: tuple[str, ...],
    height_margin: float = 0.02,
) -> tuple[Optional[str], tuple[str, ...]]:
    """The per-place feasibility gate. Returns ``(family, culprits)``.

    * ``("F3", ())`` — the object is too tall for the cell (culprit-free means failure).
    * ``("F2", residents)`` — the region is already at capacity with self-placed residents.
    * ``(None, ())`` — the placement is feasible.

    Shared by the symbolic feasibility walk (``refine.evaluate_skeleton``) and the physics
    sampler (``instrumented_refiner``) so both agree by construction.
    """
    if not height_ok(obj_height, region.cell_clearance, height_margin):
        return "F3", ()
    if len(residents) >= region_capacity(region, obj_half_x):
        return "F2", residents
    return None, ()
