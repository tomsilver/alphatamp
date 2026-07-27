"""Labelled top-down DD2D scene render — the shared human/VLM-legible view.

The vendored ``dd2d/render.py`` exists for the dataset's per-object crops and the
episode videos, and is unsuitable here on two counts: it fills ``exterior.coords``
only (so the wall band — a frame *with a hole* — paints as a solid slab over the
drawer), and it bakes 6 pt item labels at 100 dpi (so a wider raster only makes the
labels *relatively* smaller). Both matter when the consumer is a VLM reading item
numbers off the image, or a human stepping through the comparison notebook.

So we draw the scene from its polygons here, honouring interior rings and choosing
the label size in points rather than inheriting it from a fixed-dpi raster. This
module is the single home for that render: the comparison notebook's §7 planner
inspector and the VLMPlan baseline's Set-of-Mark prompt image both call it, so the
picture a reviewer sees is the picture the model saw.

Item labels are the trailing segment of the object name (``item_5`` -> ``5``), which
is what the notebook's plan tables and the VLMPlan prompt both use to name items.
"""

from __future__ import annotations

import io

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image
from shapely.geometry.base import BaseGeometry

from .dd2d.world import DrawerScene, place_polygon

# Palette. The target must be unmistakable — a VLM asked "which item is the target"
# should be able to answer from colour alone, since the prompt says it is red.
_TARGET_FACE = "#e8483c"
_CONCAVE_FACE = "#7f8ec9"
_CONVEX_FACE = "#9aa4ad"
_DRAWER_FACE = "#eef2f6"
_BUFFER_FACE = "#f4efe2"
_BUFFER_EDGE = "#a98"
_WALL_FACE = "#4a4f57"

_PAD_CM = 2.5


def _add_polygon(ax: Axes, geom: BaseGeometry, **kwargs: object) -> None:
    """Add a shapely polygon to ``ax``, **honouring interior rings**.

    Each ring becomes one MOVETO/LINETO.../CLOSEPOLY sub-path of a single
    ``PathPatch``, so matplotlib's even-odd fill leaves the holes empty. Rendering
    the holes is what keeps the drawer interior visible under the wall band.
    """
    if geom.is_empty:
        return
    for part in getattr(geom, "geoms", [geom]):
        if part.is_empty or not hasattr(part, "exterior"):
            continue
        verts: list[tuple[float, float]] = []
        codes: list[np.uint8] = []
        for ring in [part.exterior, *part.interiors]:
            pts = list(ring.coords)
            if len(pts) < 3:
                continue
            verts += pts
            codes += (
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(pts) - 2)
                + [MplPath.CLOSEPOLY]
            )
        if verts:
            patch = PathPatch(MplPath(verts, codes), **kwargs)  # type: ignore[arg-type]
            ax.add_patch(patch)


def scene_figure(
    scene: DrawerScene,
    width_in: float = 7.2,
    label_fontsize: float = 13.0,
) -> Figure:
    """Draw ``scene`` top-down and return the figure (caller owns closing it).

    Red = the retrieval target, blue = concave items, grey = convex items; the dark
    frame is the wall band and the dashed box is the buffer. The figure's aspect
    follows the scene's, so ``width_in`` alone fixes the size.
    """
    x0, y0, x1, y1 = scene.drawer.union(scene.buffer).union(scene.wall_band).bounds
    x0, y0 = x0 - _PAD_CM, y0 - _PAD_CM
    x1, y1 = x1 + _PAD_CM, y1 + _PAD_CM

    fig = Figure(figsize=(width_in, width_in * (y1 - y0) / (x1 - x0)))
    ax = fig.add_subplot(1, 1, 1)

    _add_polygon(ax, scene.drawer, facecolor=_DRAWER_FACE, edgecolor="none", zorder=0)
    _add_polygon(
        ax,
        scene.buffer,
        facecolor=_BUFFER_FACE,
        edgecolor=_BUFFER_EDGE,
        linewidth=1.4,
        linestyle="--",
        zorder=0,
    )
    _add_polygon(ax, scene.wall_band, facecolor=_WALL_FACE, edgecolor="none", zorder=1)

    for name, state in scene.items.items():
        poly = place_polygon(state.shape.polygon, state.pose)
        if state.is_target:
            face = _TARGET_FACE
        elif state.shape.concave:
            face = _CONCAVE_FACE
        else:
            face = _CONVEX_FACE
        _add_polygon(
            ax,
            poly,
            facecolor=face,
            edgecolor="#111" if state.is_target else "#555",
            linewidth=2.2 if state.is_target else 1.0,
            zorder=3,
        )
        # representative_point, not centroid: a horseshoe's centroid lands in its
        # C-opening, so a centroid-anchored mark would label empty space — fatal for a
        # Set-of-Mark prompt where the number *is* the item's identity.
        anchor = poly.representative_point()
        ax.text(
            anchor.x,
            anchor.y,
            name.split("_")[-1],
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="#111",
            zorder=7,
            path_effects=[path_effects.withStroke(linewidth=3.5, foreground="white")],
        )

    for poly, label, colour in (
        (scene.drawer, "drawer", "#333"),
        (scene.buffer, "buffer", "#a76"),
    ):
        bounds = poly.bounds
        ax.text(
            (bounds[0] + bounds[2]) / 2,
            bounds[3] + 2.0,
            label,
            ha="center",
            va="bottom",
            color=colour,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_labeled_scene(
    scene: DrawerScene,
    width_px: int = 1024,
    label_fontsize: float = 13.0,
) -> Image.Image:
    """``scene_figure`` rasterised to a PIL image ``width_px`` wide.

    Size is set by dpi rather than by upscaling, so the labels stay the same
    *physical* size as the width grows — the whole reason this module exists.
    """
    width_in = 7.2
    fig = scene_figure(scene, width_in=width_in, label_fontsize=label_fontsize)
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=width_px / width_in)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    finally:
        fig.clear()
