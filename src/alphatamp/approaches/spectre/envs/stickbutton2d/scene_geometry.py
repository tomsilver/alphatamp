"""Ground-truth object-centric scene geometry for StickButton2D.

``EpisodeRecord.scene_geometry`` is optional in the schema and ``None`` for every kinder
collection to date, which is a silent trap rather than a gap: ``train._trainable``
filters every geometry-less episode, so ``n_train`` reaches 0, ``deployed_val_fp``
returns ``inf``, ``improved = inf < inf`` is never true, and the run finishes with exit
code 0 and no ``best.pt``. A collection without this module produces a training run that
fails successfully.

**Everything here is read from kinder, not re-derived.**
:func:`kinder.envs.utils.object_to_multibody2d` is the same function the renderer and
the collision checker consume, so the geometry recorded cannot drift from the geometry
the refiner actually enforced. The only judgement this module makes is how to project a
multibody onto the schema's one-ring-per-object shape, and there is exactly one object
where that bites -- see :func:`_robot_geometry`.

Consumers beyond v3: the stored rings and poses are what a PIGINet-style low-level
predictor and the VLMPlan scene renderer read (``vlmplan/dd2d_adapter.py`` uses exactly
these fields), which is why containers and the world frame are recorded even though the
v3 tensorizer ignores them. Getting them at collection time is free; back-filling them
later is not, because ``decisions.md`` 2026-07-19 forbids regenerating geometry.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig
from kinder.envs.utils import object_to_multibody2d
from tomsgeoms2d.structs import Circle, Rectangle

from alphatamp.approaches.spectre.schema import (
    ContainerGeometry,
    ObjectGeometry,
    SceneGeometry,
)

#: Points used to polygonalize a disc. ``dataset.resample_ring`` re-samples every
# ring : to ``N_BOUNDARY_POINTS`` (32) by arc length anyway, so this only has to be fine
# enough : that the arc-length parameterisation is not visibly faceted.
_CIRCLE_POINTS = 32


def _ring_area(ring: Sequence[tuple[float, float]]) -> float:
    """Shoelace area of a closed ring, sign-independent."""
    total = 0.0
    for i, (x0, y0) in enumerate(ring):
        x1, y1 = ring[(i + 1) % len(ring)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _circle_ring(radius: float) -> tuple[tuple[float, float], ...]:
    """A disc of ``radius`` as a centred polygon ring."""
    return tuple(
        (
            radius * math.cos(2.0 * math.pi * i / _CIRCLE_POINTS),
            radius * math.sin(2.0 * math.pi * i / _CIRCLE_POINTS),
        )
        for i in range(_CIRCLE_POINTS)
    )


def _centred(
    ring: Iterable[tuple[float, float]], centre: tuple[float, float]
) -> tuple[tuple[float, float], ...]:
    return tuple((float(x) - centre[0], float(y) - centre[1]) for x, y in ring)


def _robot_geometry(obj: Any, state: Any) -> ObjectGeometry:
    """The robot's footprint: its **base disc**, not the union of its bodies.

    ``crv_robot_to_multibody2d`` returns three bodies -- ``base`` (a disc,
    ``ZOrder.ALL``) plus ``arm`` and ``gripper`` (thin rectangles, ``ZOrder.SURFACE``).
    Only the base participates in the collisions that constrain where the robot can be:
    the arm and gripper sweep *over* the table rather than into it, which is the whole
    reason a far-side button needs the stick (``geometry.robot_reach_max_y``).

    The arm is also *configuration*, not shape -- ``arm_joint`` and ``theta`` change it
    within an episode, while ``scene_geometry`` records one static footprint. Storing a
    swept union would encode an arbitrary instant of a moving part as if it were the
    object's extent. The base disc is the part that is genuinely fixed and genuinely
    blocking, so that is what is recorded.
    """
    base = object_to_multibody2d(obj, state, {}).get_body("base")
    geom = base.geom
    assert isinstance(geom, Circle), "crv_robot base is expected to be a disc"
    radius = float(geom.radius)
    return ObjectGeometry(
        name=str(obj.name),
        pose=(float(geom.x), float(geom.y), float(state.get(obj, "theta"))),
        boundary=_circle_ring(radius),
        family="circle",
        area=math.pi * radius * radius,
        concave=False,
    )


def _body_geometry(obj: Any, state: Any) -> ObjectGeometry:
    """One single-body object (a button disc or the stick rectangle)."""
    geom = object_to_multibody2d(obj, state, {}).get_body("root").geom
    theta = float(state.get(obj, "theta"))
    if isinstance(geom, Circle):
        radius = float(geom.radius)
        return ObjectGeometry(
            name=str(obj.name),
            pose=(float(geom.x), float(geom.y), theta),
            boundary=_circle_ring(radius),
            family="circle",
            area=math.pi * radius * radius,
            concave=False,
        )
    assert isinstance(geom, Rectangle), f"unhandled geom {type(geom).__name__}"
    # `Rectangle(x, y, w, h, theta)` takes the **lower-left corner** -- upstream has a
    # separate `from_center` for the other convention -- so `state.get(obj, "x")` is not
    # the pose the schema wants. Reading it as a centroid would displace the stick by
    # half its 1.25 length. `.center` and `.vertices` are upstream's own, so the two
    # cannot disagree.
    centre = (float(geom.center[0]), float(geom.center[1]))
    ring = _centred(geom.vertices, centre)
    return ObjectGeometry(
        name=str(obj.name),
        pose=(centre[0], centre[1], theta),
        boundary=ring,
        family="rectangle",
        area=_ring_area(ring),
        concave=False,
    )


def build_scene_geometry(
    state: Any, config: StickButton2DEnvConfig | None = None
) -> SceneGeometry:
    """Ground-truth geometry for every object in a StickButton2D initial state.

    Covers the whole abstract universe -- ``robot``, ``stick`` and every ``buttonN`` --
    which is what invariant I5 requires (``EpisodeRecord.validate`` asserts that every
    ``object_registry`` key has geometry, and the registry is built from exactly those).

    ``is_target`` stays ``False`` on every object: DD2D's target is the one item being
    retrieved, and StickButton2D has no analogue. The consequence is visible rather than
    hidden -- ``dataset`` falls back to ``(0, 0)`` for the target pose, so its
    ``rel`` block degrades from target-relative offsets to absolute world coordinates
    and its area ratio becomes a raw area. That is acceptable here because the world is
    a fixed 3.5 x 2.5 frame, so absolute coordinates are themselves meaningful; it would
    not be in a domain with a moving reference frame.
    """
    cfg = config if config is not None else StickButton2DEnvConfig()
    objects: list[ObjectGeometry] = []
    for obj in state:
        if str(obj.type.name) == "crv_robot":
            objects.append(_robot_geometry(obj, state))
        else:
            objects.append(_body_geometry(obj, state))

    tx, ty = float(cfg.table_pose.x), float(cfg.table_pose.y)
    tw, th = float(cfg.table_shape[0]), float(cfg.table_shape[1])
    world = (
        float(cfg.world_min_x),
        float(cfg.world_min_y),
        float(cfg.world_max_x),
        float(cfg.world_max_y),
    )
    containers = (
        # The table is the reason reach is bounded: the robot base (`ZOrder.ALL`) cannot
        # overlap it, so a button deep inside is stick-only.
        ContainerGeometry(kind="table", bounds=(tx, ty, tx + tw, ty + th)),
        ContainerGeometry(kind="world", bounds=world),
    )
    return SceneGeometry(
        objects=tuple(sorted(objects, key=lambda o: o.name)),
        containers=containers,
        units="m",
        # Normalisation frame. DD2D names these `drawer_w`/`drawer_d`; `dataset`
        # accepts both spellings, and an absent frame would leave SB2D's coordinates
        # unnormalised out to 3.5 while DD2D's arrive in [0, 1].
        frame={"frame_w": world[2] - world[0], "frame_d": world[3] - world[1]},
    )
