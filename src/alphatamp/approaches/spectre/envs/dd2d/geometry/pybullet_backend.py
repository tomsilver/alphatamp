"""PyBullet render backend: builds the real 3-D sorting scene and renders RGB +
a true per-object segmentation mask **headlessly** (DIRECT mode + TinyRenderer,
no GPU/display).

This is the concrete confirmation that PIGINet-style segmented object images are
doable: PyBullet is the same simulator the PIGINet authors rendered with, and
``getCameraImage`` returns a segmentation buffer keyed by body id. Extending to
the paper's 6 viewpoints is just more ``computeViewMatrix`` calls (see
``rendering.multiview_stub``).
"""

from __future__ import annotations

import numpy as np

from ..scene import GeometricScene
from . import GeometryBackend, RenderResult

_COLORS = {
    "red": (0.86, 0.24, 0.24, 1.0),
    "green": (0.24, 0.70, 0.29, 1.0),
    "blocker": (0.51, 0.51, 0.51, 1.0),
}
_TABLE_RGBA = {
    "red_table": (0.47, 0.16, 0.16, 1.0),
    "blue_table": (0.16, 0.24, 0.47, 1.0),
    "green_table": (0.16, 0.43, 0.22, 1.0),
    "purple_table": (0.35, 0.20, 0.43, 1.0),
}


class PyBulletBackend(GeometryBackend):
    name = "pybullet"

    @classmethod
    def available(cls) -> bool:
        try:
            import pybullet  # noqa: F401

            return True
        except Exception:  # pragma: no cover
            return False

    def render_segmented(
        self,
        scene: GeometricScene,
        view: str = "topdown",
        width: int = 256,
        height: int = 256,
    ) -> RenderResult:
        import pybullet as p

        cid = p.connect(p.DIRECT)
        try:
            id_to_name: dict[int, str] = {}
            # tables as thin static boxes
            for t in scene.tables:
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[t.half_extent, t.half_extent, 0.01]
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[t.half_extent, t.half_extent, 0.01],
                    rgbaColor=_TABLE_RGBA.get(t.name, (0.3, 0.3, 0.3, 1.0)),
                )
                p.createMultiBody(
                    0, col, vis, basePosition=[t.center[0], t.center[1], -0.01]
                )

            # movable objects, each its own body -> its own segmentation id
            for o in scene.objects:
                he = [o.size[0] / 2, o.size[1] / 2, o.size[2] / 2]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=he,
                    rgbaColor=_COLORS.get(o.color, (0.8, 0.8, 0.0, 1.0)),
                )
                body = p.createMultiBody(
                    0.05, col, vis, basePosition=[o.pose[0], o.pose[1], o.pose[2]]
                )
                id_to_name[body] = o.name

            view_m, proj_m = _camera(p, view)
            w, h, rgb_buf, _, seg_buf = p.getCameraImage(
                width, height, view_m, proj_m, renderer=p.ER_TINY_RENDERER
            )
            rgb = np.reshape(np.array(rgb_buf, dtype=np.uint8), (h, w, 4))[:, :, :3]
            seg = np.reshape(np.array(seg_buf, dtype=np.int32), (h, w))
            # PyBullet seg uses body ids (>=0); background is -1. Keep only object ids.
            keep = np.isin(seg, list(id_to_name.keys()))
            seg = np.where(keep, seg, -1)
            return RenderResult(rgb=rgb, seg=seg, id_to_name=id_to_name, view=view)
        finally:
            p.disconnect(cid)


def _camera(p, view: str):
    if view == "topdown":
        eye, target, up = [0.0, 0.0, 1.6], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    elif view == "oblique":
        eye, target, up = [1.1, -1.1, 1.1], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    else:  # pragma: no cover - unknown view falls back to topdown
        eye, target, up = [0.0, 0.0, 1.6], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    view_m = p.computeViewMatrix(eye, target, up)
    proj_m = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=4.0)
    return view_m, proj_m
