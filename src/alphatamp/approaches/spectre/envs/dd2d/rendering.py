"""Rendering confirmation.

We are not building the full PIGINet image pipeline yet, but we must *confirm*
that producing segmented object images is feasible. ``confirm_rendering`` renders
one frame of a sorting scene through a geometry backend, checks that the
segmentation buffer actually separates the objects (>1 distinct id), and writes a
PNG + the raw segmentation array for inspection. ``multiview_stub`` documents the
6-camera extension used in the paper (line 76) without implementing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .geometry import GeometryBackend, RenderResult, get_backend
from .scene import GeometricScene


@dataclass
class RenderCheck:
    backend: str
    view: str
    n_segments: int
    ok: bool
    png_path: str | None
    seg_path: str | None


def confirm_rendering(
    scene: GeometricScene,
    backend: GeometryBackend | None = None,
    view: str = "topdown",
    out_dir: str = "out",
    prefix: str = "render_check",
) -> RenderCheck:
    """Render one frame and confirm per-object segmentation works."""
    backend = backend or get_backend()
    result = backend.render_segmented(scene, view=view)
    n_segments = len(result.segment_ids())
    ok = n_segments > 1

    png_path = seg_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"{prefix}_{backend.name}_{view}.png")
        seg_path = os.path.join(out_dir, f"{prefix}_{backend.name}_{view}_seg.npy")
        _save_png(result.rgb, png_path)
        np.save(seg_path, result.seg)

    return RenderCheck(
        backend=backend.name,
        view=view,
        n_segments=n_segments,
        ok=ok,
        png_path=png_path,
        seg_path=seg_path,
    )


def _save_png(rgb: np.ndarray, path: str) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
    except Exception:  # pragma: no cover - imageio always present here
        from PIL import Image

        Image.fromarray(rgb).save(path)


def multiview_stub() -> dict:
    """Document (not implement) the multi-view setup PIGINet uses.

    The paper renders 6 viewpoints (line 76): several side views to see into
    cabinets/fridges plus top-down views for sinks/pots. For the open tabletop
    sorting scene a top-down view plus a couple of obliques suffices; each extra
    view is one more ``computeViewMatrix`` in the PyBullet backend. Per-object
    crops come from the segmentation bbox already produced in
    ``record.build_image_refs``.
    """
    return {
        "paper_num_views": 6,
        "suggested_sorting_views": ["topdown", "oblique_ne", "oblique_sw"],
        "per_object_crop": "use ImageRef.bbox from the segmentation mask",
        "status": "documented, not implemented (pixels deferred)",
    }
