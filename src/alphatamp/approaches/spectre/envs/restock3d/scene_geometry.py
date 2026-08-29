"""Ground-truth object-centric scene geometry for **Restock3D v2** (3D point clouds).

``EpisodeRecord.scene_geometry`` is optional in the schema and ``None`` for every kinder
collection to date -- a silent trap, not a gap: ``train._trainable`` drops every
geometry-less episode, so ``n_train`` reaches 0 and the run finishes with exit 0 and no
``best.pt`` (see ``envs/stickbutton2d/scene_geometry.py``). A Restock3D collection without
this module produces a training run that fails successfully.

**Everything here is read from the state's ground-truth features, not sensed.** Restock3D
objects are axis-aligned cuboids, so each object's point cloud is an *analytic* box surface
sampled from its half-extents (``state.get_object_half_extents``) at its world pose
(``state.get_object_pose``). Crucially, a cube (half_z 0.025) and a tall block (half_z
0.12) share a 2D footprint and differ *only* in z -- so this module emits **both** the 2D
footprint ring (for 2D consumers: PIGINet/VLMPlan/legacy SPECTRE) **and** the 3D
``point_cloud`` + ``pose_z`` + ``height`` (for the 3D SPECTRE encoder), the latter carrying
the F3-critical height distinction the footprint loses. See docs/decisions 2026-08-18.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from alphatamp.approaches.spectre.schema import (
    ContainerGeometry,
    ObjectGeometry,
    SceneGeometry,
)

from .kinematic_env import Restock3DEnvConfig

#: Points per object point cloud. Mirrors ``encoders.N_BOUNDARY_POINTS`` (32) -- the
# tensorizer resamples to that anyway, so matching it here avoids a resample. Defined
# locally so this env module carries no torch dependency.
POINT_CLOUD_SIZE = 32

#: Movable object name prefixes. v2 builds ``cube_goal*``/``block_goal*``/``clutter*``; v3 builds
# per-object-dims ``obj_goal*`` (family derived from height, not name). Everything else in the state
# (the robot, shelf boards) is handled explicitly or skipped.
_MOVABLE_PREFIXES = ("cube_goal", "block_goal", "clutter", "obj_goal")

#: v3 obj_goal family threshold (full height): > this is tall-only (matches ``feasibility_v3``'s
# short cutoff). Metadata only.
_V3_TALL_HEIGHT_THRESHOLD = 0.12

#: Nominal robot base footprint (a TidyBot mobile base). The robot is in the object
# registry (it is a ``pick``/``place`` argument), so I5 requires it to have geometry, but
# its exact extent is immaterial -- it is never the F3 signal and is identical every
# episode. A fixed nominal box keeps the record self-consistent.
_ROBOT_HALF = (0.15, 0.15, 0.30)


def _unit_box_points() -> np.ndarray:
    """A fixed 32-point sampling of the surface of the unit box ``[-1, 1]^3``.

    8 corners (a faithful extent -- especially in z, the F3 signal) + 6 faces x 4
    quadrant-centre points. Deterministic and order-free (the point-set encoder pools
    symmetrically), so the exact scheme is immaterial as long as the extent is honest.
    """
    pts: list[tuple[float, float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append((sx, sy, sz))
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1.0, 1.0):
            for u in (-0.5, 0.5):
                for v in (-0.5, 0.5):
                    p = [0.0, 0.0, 0.0]
                    p[axis] = sign
                    p[others[0]] = u
                    p[others[1]] = v
                    pts.append((p[0], p[1], p[2]))
    return np.asarray(pts, dtype=np.float32)


_UNIT_BOX = _unit_box_points()


def object_point_cloud(
    half_extents: tuple[float, float, float], n_points: int = POINT_CLOUD_SIZE
) -> np.ndarray:
    """Analytic ``(n_points, 3)`` surface point cloud for an axis-aligned box.

    Item frame, centroid at the origin. The z-extent scales with ``half_extents[2]``, so a
    cube and a tall block -- identical in x, y -- produce clouds that differ in z. That is
    the whole point: it is the F3 signal a 2D footprint cannot carry.
    """
    unit = _UNIT_BOX
    if n_points != unit.shape[0]:
        idx = np.arange(n_points) % unit.shape[0]
        unit = unit[idx]
    return (unit * np.asarray(half_extents, dtype=np.float32)).astype(np.float32)


def _yaw_from_quat(q: tuple[float, float, float, float]) -> float:
    """Yaw (rotation about +z) from a ``(qx, qy, qz, qw)`` quaternion."""
    qx, qy, qz, qw = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _rect_ring(hx: float, hy: float) -> tuple[tuple[float, float], ...]:
    """The 2D footprint ring (centred, item frame) for a box half-width/depth."""
    return ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))


def _family_of(name: str, height: Optional[float] = None) -> str:
    if name.startswith("block_goal"):
        return "tall"
    if name.startswith("cube_goal"):
        return "cube"
    if name.startswith("obj_goal"):  # v3: family from height, not name
        return (
            "tall"
            if (height is not None and height > _V3_TALL_HEIGHT_THRESHOLD)
            else "cube"
        )
    return "clutter"


def _movable_geometry(name: str, state: Any) -> ObjectGeometry:
    pose = state.get_object_pose(name)
    x, y, z = (float(c) for c in pose.position)
    yaw = _yaw_from_quat(tuple(float(c) for c in pose.orientation))  # type: ignore[arg-type]
    hx, hy, hz = (float(h) for h in state.get_object_half_extents(name))
    pc = object_point_cloud((hx, hy, hz))
    return ObjectGeometry(
        name=name,
        pose=(x, y, yaw),
        boundary=_rect_ring(hx, hy),
        family=_family_of(name, 2.0 * hz),
        area=4.0 * hx * hy,
        concave=False,
        point_cloud=tuple((float(a), float(b), float(c)) for a, b, c in pc),
        pose_z=z,
        height=2.0 * hz,
    )


def _robot_geometry(state: Any) -> ObjectGeometry:
    robot = state.get_object_from_name("robot")
    rx = float(state.get(robot, "pos_base_x"))
    ry = float(state.get(robot, "pos_base_y"))
    rrot = float(state.get(robot, "pos_base_rot"))
    hx, hy, hz = _ROBOT_HALF
    pc = object_point_cloud((hx, hy, hz))
    return ObjectGeometry(
        name="robot",
        pose=(rx, ry, rrot),
        boundary=_rect_ring(hx, hy),
        family="robot",
        area=4.0 * hx * hy,
        concave=False,
        point_cloud=tuple((float(a), float(b), float(c)) for a, b, c in pc),
        pose_z=hz,
        height=2.0 * hz,
    )


def build_scene_geometry(
    state: Any, config: Optional[Restock3DEnvConfig] = None
) -> SceneGeometry:
    """Ground-truth 3D geometry for every registry object in a Restock3D initial state.

    Emits geometry for the robot and every movable (``cube_goal*``/``block_goal*``/
    ``clutter*``); shelf-board fixtures are recorded as containers, not objects. Covers
    the whole abstract universe I5 requires (the object registry is robot + movables,
    with no region objects in v2). ``is_target`` is ``False`` throughout: Restock3D
    stores *every* goal object, so there is no single distinguished target.
    """
    cfg = config if config is not None else Restock3DEnvConfig()
    objects: list[ObjectGeometry] = []
    for obj in state:
        name = str(obj.name)
        if name == "robot":
            objects.append(_robot_geometry(state))
        elif name.startswith(_MOVABLE_PREFIXES):
            objects.append(_movable_geometry(name, state))
        # else: shelf-board fixtures -- not registry objects; recorded as a container below.

    sx = float(cfg.shelf_pose.position[0])
    sy = float(cfg.shelf_pose.position[1])
    hw, hd = cfg.shelf_width / 2.0, cfg.shelf_depth / 2.0
    containers = (
        ContainerGeometry(kind="shelf", bounds=(sx - hw, sy - hd, sx + hw, sy + hd)),
    )
    return SceneGeometry(
        objects=tuple(sorted(objects, key=lambda o: o.name)),
        containers=containers,
        units="m",
        # Normalization frame: a fixed, config-derived workspace box (stable across seeds,
        # so pose normalization is consistent). ``dataset.build_example`` needs
        # ``frame_w``/``frame_d``; ``frame_h`` carries the z extent for the 3D path.
        frame={
            "frame_w": float(cfg.shelf_width),
            "frame_d": sy + float(cfg.shelf_depth),
            "frame_h": float(cfg.bottom_surface_z) + float(sum(cfg.section_clearances)),
        },
    )
