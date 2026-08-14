"""Region geometry for Restock3D.

Regions (``region_<shelf>_<idx>``) are 6-value local boxes on a cupboard shelf declared in the
task JSON; they are NOT objects in the env state. The abstractor (``InRegion``) and the
place-to-region controller need each region's world centre, and the geometric feasibility gate
(F2 slot occupancy, F3 height) needs each region's footprint and its *cell clearance* (the
vertical gap above the region's shelf surface). This module reconstructs all of that from the
task config plus the cupboard's pose in the current state.

Adapted from ``envs/shelf3d/region_geometry.py`` (same cupboard yaw transform), broadened to a
``region_`` name filter and enriched with footprint half-extents + cell clearance. The cupboard
transform (yaw ``theta`` about +z): local ``(lx, ly)`` -> world
``(cx + cos*lx - sin*ly, cy + sin*lx + cos*ly)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from spatialmath import UnitQuaternion

_CUPBOARD_NAME = "cupboard_1"
_REGION_PREFIX = "region_"


@dataclass(frozen=True)
class RegionInfo:
    """A shelf region: world centre, footprint, the shelf it sits on, and its cell
    clearance."""

    name: str
    shelf: int
    center_xy: tuple[float, float]  # world (x, y) of the region box centre
    half_xy: tuple[float, float]  # world-aligned footprint half-extents (x, y)
    cell_clearance: (
        float  # vertical gap above the region's shelf surface (for F3 height gate)
    )
    surface_z: float  # world z of the region's shelf surface (place height reference)


def _cupboard_yaw(state, cupboard) -> float:
    q = UnitQuaternion(
        s=state.get(cupboard, "qw"),
        v=(
            state.get(cupboard, "qx"),
            state.get(cupboard, "qy"),
            state.get(cupboard, "qz"),
        ),
    )
    return float(q.rpy()[2])


def load_region_infos(task_json_path: str, state) -> dict[str, RegionInfo]:
    """World geometry of every ``region_*`` cupboard shelf region in the task JSON.

    ``cell_clearance`` / ``surface_z`` come from the task JSON's ``region_meta`` block (written by
    the generator: per-region cell clearance and shelf surface z), defaulting to a large clearance
    and the region box's own z-floor when absent (so an un-annotated scene still loads).
    """
    cupboard = state.get_object_from_name(_CUPBOARD_NAME)
    cx, cy = state.get(cupboard, "x"), state.get(cupboard, "y")
    theta = _cupboard_yaw(state, cupboard)
    ct, st = np.cos(theta), np.sin(theta)

    with open(task_json_path, encoding="utf-8") as f:
        cfg = json.load(f)
    meta = cfg.get("region_meta", {})

    infos: dict[str, RegionInfo] = {}
    for name, r in cfg.get("regions", {}).items():
        if r.get("target") != _CUPBOARD_NAME or "shelf" not in r:
            continue
        if not name.startswith(_REGION_PREFIX):
            continue  # *_init_region are spawn-only, not planning objects
        box = r["ranges"][0]  # [lx0, ly0, z0, lx1, ly1, z1]
        lx = 0.5 * (box[0] + box[3])
        ly = 0.5 * (box[1] + box[4])
        hlx = 0.5 * abs(box[3] - box[0])
        hly = 0.5 * abs(box[4] - box[1])
        wx = cx + ct * lx - st * ly
        wy = cy + st * lx + ct * ly
        # yaw multiples of 90 deg swap x/y half-extents; use |cos|/|sin| to map generally.
        half_x = abs(ct) * hlx + abs(st) * hly
        half_y = abs(st) * hlx + abs(ct) * hly
        rmeta = meta.get(name, {})
        infos[name] = RegionInfo(
            name=name,
            shelf=int(r["shelf"]),
            center_xy=(float(wx), float(wy)),
            half_xy=(float(half_x), float(half_y)),
            cell_clearance=float(rmeta.get("cell_clearance", 1.0)),
            surface_z=float(rmeta.get("surface_z", box[2])),
        )
    return infos


def shelf_surface_z_from_cube(state, movable_on_shelf) -> float:
    """Shelf surface height read off a cube currently resting on that shelf
    (fallback)."""
    return float(
        state.get(movable_on_shelf, "z") - state.get(movable_on_shelf, "bb_z") / 2
    )
