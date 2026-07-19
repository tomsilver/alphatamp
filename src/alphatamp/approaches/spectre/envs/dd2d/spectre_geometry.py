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

from .dd2d.grasps import has_grasp
from .dd2d.scene import WALL_BAND
from .dd2d.shapes import Shape
from .dd2d.world import place_polygon


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
