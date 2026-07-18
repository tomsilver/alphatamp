"""Dependency-light top-down rasteriser (numpy only).

Renders the sorting scene as a top-down RGB image plus an integer segmentation buffer by
drawing each table and object as a filled axis-aligned square. Used as the render
fallback when PyBullet is unavailable; proves the segmented-image construction needs no
heavy simulator.
"""

from __future__ import annotations

import numpy as np

from ..scene import GeometricScene
from . import GeometryBackend, RenderResult

_COLORS = {
    "red": (220, 60, 60),
    "green": (60, 180, 75),
    "blocker": (130, 130, 130),
}
_TABLE_TINT = {
    "red_table": (120, 40, 40),
    "blue_table": (40, 60, 120),
    "green_table": (40, 110, 55),
    "purple_table": (90, 50, 110),
}
_BG = (20, 20, 24)


class Numpy2DBackend(GeometryBackend):
    name = "numpy2d"

    @classmethod
    def available(cls) -> bool:
        return True

    def render_segmented(
        self,
        scene: GeometricScene,
        view: str = "topdown",
        width: int = 256,
        height: int = 256,
    ) -> RenderResult:
        # world extent: a square covering the table cross with margin
        extent = 0.95
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = _BG
        seg = np.full((height, width), -1, dtype=np.int32)
        id_to_name: dict[int, str] = {}

        def to_px(x: float, y: float) -> tuple[int, int]:
            u = int((x + extent) / (2 * extent) * (width - 1))
            v = int((extent - y) / (2 * extent) * (height - 1))  # y up -> row down
            return np.clip(v, 0, height - 1), np.clip(u, 0, width - 1)

        def fill_square(cx, cy, half, color, seg_id=None):
            r0, c0 = to_px(cx - half, cy + half)
            r1, c1 = to_px(cx + half, cy - half)
            rgb[r0 : r1 + 1, c0 : c1 + 1] = color
            if seg_id is not None:
                seg[r0 : r1 + 1, c0 : c1 + 1] = seg_id

        # tables first (background layer, not segmented as objects)
        for t in scene.tables:
            fill_square(
                t.center[0],
                t.center[1],
                t.half_extent,
                _TABLE_TINT.get(t.name, (60, 60, 60)),
            )

        # objects on top, each its own segment id; taller blockers drawn larger
        for oid, o in enumerate(scene.objects):
            half = o.footprint_radius * (1.6 if o.is_blocker else 1.2)
            fill_square(
                o.pose[0],
                o.pose[1],
                half,
                _COLORS.get(o.color, (200, 200, 0)),
                seg_id=oid,
            )
            id_to_name[oid] = o.name

        return RenderResult(rgb=rgb, seg=seg, id_to_name=id_to_name, view=view)
