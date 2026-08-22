"""Region geometry for the ShelfObstruct3D obstruction variant.

Regions (``target_region_*`` / ``free_region_*``) are defined in the task JSON as 6-value local
boxes on a cupboard shelf; they are NOT objects in the env state. The abstractor (``At`` /
``Clear``) and the place-to-region controller both need each region's world centre, so this
module reconstructs them from the task config plus the cupboard's pose in the current state.

Cupboard transform (yaw θ about +z): a local point ``(lx, ly)`` maps to world
``(cx + cosθ·lx - sinθ·ly, cy + sinθ·lx + cosθ·ly)``. For the ShelfObstruct3D cupboard at yaw
90° this is ``world = (cx - ly, cy + lx)`` -- verified against the blocker's spawn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from spatialmath import UnitQuaternion

_CUPBOARD_NAME = "cupboard_1"


@dataclass(frozen=True)
class RegionInfo:
    """A shelf region's world centre and the shelf it sits on."""

    name: str
    shelf: int
    center_xy: tuple[float, float]  # world (x, y) of the region's box centre
    is_target: (
        bool  # target_region_* (goal destinations) vs free_region_* (relocation spots)
    )


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
    """World centre of every cupboard shelf region defined in the task JSON."""
    cupboard = state.get_object_from_name(_CUPBOARD_NAME)
    cx, cy = state.get(cupboard, "x"), state.get(cupboard, "y")
    theta = _cupboard_yaw(state, cupboard)
    ct, stheta = np.cos(theta), np.sin(theta)

    with open(task_json_path, encoding="utf-8") as f:
        cfg = json.load(f)

    infos: dict[str, RegionInfo] = {}
    for name, r in cfg.get("regions", {}).items():
        if r.get("target") != _CUPBOARD_NAME or "shelf" not in r:
            continue
        # Only placement regions are planning objects; *_init_region are spawn-only.
        if not (name.startswith("target_region") or name.startswith("free_region")):
            continue
        box = r["ranges"][0]  # [lx0, ly0, z0, lx1, ly1, z1]
        lx = 0.5 * (box[0] + box[3])
        ly = 0.5 * (box[1] + box[4])
        wx = cx + ct * lx - stheta * ly
        wy = cy + stheta * lx + ct * ly
        infos[name] = RegionInfo(
            name=name,
            shelf=int(r["shelf"]),
            center_xy=(float(wx), float(wy)),
            is_target=name.startswith("target_region"),
        )
    return infos


def shelf_surface_z(state, movable_on_shelf) -> float:
    """The shelf surface height, read off a cube currently resting on that shelf.

    All ShelfObstruct3D regions are on the same shelf as the blocker, so the blocker's rest
    height gives the place height directly (surface = cube_z - bb_z/2). This avoids depending on
    the cupboard's internal z origin.
    """
    return float(
        state.get(movable_on_shelf, "z") - state.get(movable_on_shelf, "bb_z") / 2
    )
