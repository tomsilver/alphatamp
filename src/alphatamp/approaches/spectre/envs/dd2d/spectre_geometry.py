"""Reconstruct DD2D grasp geometry from a persisted ``SceneGeometry`` — no regeneration.

The stored ``EpisodeRecord`` already carries the exact ground-truth geometry (schema
Step 3: per-object item-frame boundary ring + world pose, drawer ``W``/``D``, buffer).
Every grasp query the post-mortem proofs need — *"is the target still graspable once
subset ``S`` is removed?"* — is a pure function of that stored geometry:

- ``grasp_cells`` depends only on ``shape.polygon`` (the item-frame ring we persist),
- obstacles are ``place_polygon(boundary_i, pose_i)`` over the present items, and
- the ``wall_band`` is a fixed ``WALL_BAND``-thick frame rebuilt from ``W``/``D``.

So we reconstruct the obstacle set directly instead of re-running the scene generator.
This matters for **correctness, not just cost**: regenerating from the seed has to
*infer* the generation parameters, and its rejection sampling diverges under any
mismatch, yielding a geometrically-different scene (same object names, different poses)
whose proofs contradict the collected feasibility labels. Reconstructing from the stored
geometry uses the *same poses the refiner labeled*, so a proof can never disagree with a
label (a `blocked-after-removing` can never fire on a subset the collector proved
feasible). See ``docs/decisions.md`` 2026-07-19.
"""

from __future__ import annotations

from typing import Iterable

from shapely import box as shp_box
from shapely.geometry import Polygon

from alphatamp.approaches.spectre.schema import ObjectGeometry, SceneGeometry

from .drawer.grasps import has_grasp
from .drawer.scene import WALL_BAND
from .drawer.shapes import Shape
from .drawer.world import DrawerScene, ItemState, place_polygon


def _item_polygon(geom: ObjectGeometry) -> Polygon:
    """Item-frame footprint polygon (centroid at origin) from the stored boundary ring.

    ``record_ext`` persists ``shape.polygon.exterior.coords[:-1]`` (closing point
    dropped); ``Polygon`` re-closes it, so this is the live ``shape.polygon`` up to the
    4-decimal storage rounding — well below grasp-clearance tolerances.
    """
    return Polygon(geom.boundary)


def reconstruct_wall_band(frame: dict[str, float]) -> Polygon:
    """The drawer's ``WALL_BAND``-thick perimeter frame, from stored ``drawer_w/_d``.

    Mirrors ``dd2d.scene._drawer_geometry``: ``box(-b,-b, W+b, D+b) \\ box(0,0, W, D)``.
    """
    w = float(frame["drawer_w"])
    d = float(frame["drawer_d"])
    outer = shp_box(-WALL_BAND, -WALL_BAND, w + WALL_BAND, d + WALL_BAND)
    return outer.difference(shp_box(0.0, 0.0, w, d))


def reconstruct_scene(
    scene_geometry: SceneGeometry, margin: float = 1.0
) -> DrawerScene:
    """Rebuild a live ``DrawerScene`` from stored geometry — every item in the drawer at
    its stored pose, the reconstructed wall band, and the stored buffer.

    This is the same-poses-the-labeler-used reconstruction extended from single grasp
    queries to a full scene, so the env's own ``_blocker_sets`` / certificate run over
    it unchanged (reconstruct, never regenerate — ``docs/decisions.md`` 2026-07-19).
    """
    if scene_geometry.frame is None:
        raise ValueError("scene_geometry.frame lacks drawer_w/drawer_d")
    w = float(scene_geometry.frame["drawer_w"])
    d = float(scene_geometry.frame["drawer_d"])
    buffer = None
    for c in scene_geometry.containers:
        if c.kind == "buffer":
            buffer = shp_box(*(float(v) for v in c.bounds))
    if buffer is None:
        raise ValueError("scene_geometry has no buffer container")
    items: dict[str, ItemState] = {}
    target_name = None
    for o in scene_geometry.objects:
        items[o.name] = ItemState(
            name=o.name,
            shape=Shape(family=o.family, polygon=_item_polygon(o), concave=o.concave),
            pose=tuple(float(v) for v in o.pose),  # type: ignore[arg-type]
            region="drawer",
            is_target=o.is_target,
        )
        if o.is_target:
            target_name = o.name
    if target_name is None:
        raise ValueError("scene_geometry has no target object")
    return DrawerScene(
        drawer=shp_box(0.0, 0.0, w, d),
        wall_band=reconstruct_wall_band(scene_geometry.frame),
        buffer=buffer,
        items=items,
        target=target_name,
        margin=margin,
        dims={"W": w, "D": d},
    )


def _scene_without(scene: DrawerScene, subset: set[str]) -> DrawerScene:
    """A view of ``scene`` with ``subset`` items removed from the drawer (target
    kept)."""
    kept = {n: s for n, s in scene.items.items() if n not in subset}
    return DrawerScene(
        drawer=scene.drawer,
        wall_band=scene.wall_band,
        buffer=scene.buffer,
        items=kept,
        target=scene.target,
        margin=scene.margin,
        dims=scene.dims,
    )


def grasp_witness_after_removing(
    scene: DrawerScene, subset: Iterable[str]
) -> frozenset[str]:
    """Drawer items that block **every** still-open target corridor once ``subset`` is
    removed — their removal is necessary to open any of those corridors (a hint, §6.4).

    Intersection of the per-corridor blocker sets over corridors that are not walled
    off; empty if the target is already graspable or no corridor is jointly blocked.
    """
    from .drawer.enumerate import _blocker_sets, _footprints

    view = _scene_without(scene, set(subset))
    sets = [s for s in _blocker_sets(view, _footprints(view)) if s]
    if not sets:
        return frozenset()
    witness = set(sets[0])
    for s in sets[1:]:
        witness &= s
    return frozenset(witness)


def target_blocked_after_removing(
    scene_geometry: SceneGeometry, subset: Iterable[str]
) -> bool:
    """True iff the target has **no** clear grasp once ``subset`` is removed from the
    drawer — i.e. removing ``subset`` does not open the target.

    Pure function of the stored geometry; uses the env's own ``has_grasp`` primitives,
    so it matches the live check the labeler ran (up to micron-level storage rounding).
    By removal-monotonicity a subset that leaves the target blocked can only be
    infeasible, which is exactly the ``blocked-at-contents`` proof condition (proposal
    §5).
    """
    removed = set(subset)
    objs = {o.name: o for o in scene_geometry.objects}
    target = next(o for o in scene_geometry.objects if o.is_target)
    present = [n for n in objs if n != target.name and n not in removed]
    if scene_geometry.frame is None:
        raise ValueError("scene_geometry.frame lacks drawer_w/drawer_d for wall_band")
    obstacles = [place_polygon(_item_polygon(objs[n]), objs[n].pose) for n in present]
    obstacles.append(reconstruct_wall_band(scene_geometry.frame))
    tshape = Shape(
        family=target.family,
        polygon=_item_polygon(target),
        concave=target.concave,
    )
    return has_grasp(tshape, target.pose, obstacles) is None
