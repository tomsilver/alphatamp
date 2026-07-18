"""Shared colored-box construction for PyBullet scenes (tables + objects).

Factored out so the segmentation render backend, the Panda refinement world, and the
video renderer all build the same bodies the same way.
"""

from __future__ import annotations

from ..scene import GeometricScene

# object color -> rgba. red/green carry sorting semantics; the rest are a cosmetic
# palette so stacking's generic blocks render distinctly (colour is not meaningful
# for stacking). Unknown names fall back to yellow in ``add_objects``.
OBJECT_RGBA = {
    "red": (0.86, 0.24, 0.24, 1.0),
    "green": (0.24, 0.70, 0.29, 1.0),
    "blocker": (0.51, 0.51, 0.51, 1.0),
    "blue": (0.23, 0.39, 0.85, 1.0),
    "orange": (0.93, 0.58, 0.16, 1.0),
    "purple": (0.58, 0.30, 0.71, 1.0),
    "cyan": (0.20, 0.72, 0.74, 1.0),
    "yellow": (0.91, 0.80, 0.18, 1.0),
    "magenta": (0.80, 0.28, 0.62, 1.0),
}
# table name -> rgba
TABLE_RGBA = {
    "red_table": (0.55, 0.20, 0.20, 1.0),
    "blue_table": (0.20, 0.30, 0.55, 1.0),
    "green_table": (0.20, 0.50, 0.27, 1.0),
    "purple_table": (0.40, 0.24, 0.50, 1.0),
}
TABLE_THICKNESS = 0.01  # thin slab; top surface at z=0


def add_box(p, half_extents, position, rgba, mass=0.0):
    """Create a single colored box multibody; returns its body id."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=rgba
    )
    return p.createMultiBody(mass, col, vis, basePosition=list(position))


def add_tables(p, scene: GeometricScene) -> dict[int, str]:
    """Add table slabs (top at z=0).

    Returns {body_id: table_name}.
    """
    ids: dict[int, str] = {}
    for t in scene.tables:
        bid = add_box(
            p,
            [t.half_extent, t.half_extent, TABLE_THICKNESS],
            [t.center[0], t.center[1], -TABLE_THICKNESS],
            TABLE_RGBA.get(t.name, (0.3, 0.3, 0.3, 1.0)),
            mass=0.0,
        )
        ids[bid] = t.name
    return ids


def add_objects(p, scene: GeometricScene, mass=0.0) -> dict[int, str]:
    """Add one box per movable object at its scene pose.

    Uses ``o.pose[2]`` (not ``size/2``) so a pre-stacked initial tower is built at
    its true height rather than flattened onto the table. Returns {body_id: object_name}.
    """
    ids: dict[int, str] = {}
    for o in scene.objects:
        he = [o.size[0] / 2, o.size[1] / 2, o.size[2] / 2]
        bid = add_box(
            p,
            he,
            [o.pose[0], o.pose[1], o.pose[2]],
            OBJECT_RGBA.get(o.color, (0.8, 0.8, 0.0, 1.0)),
            mass,
        )
        ids[bid] = o.name
    return ids
