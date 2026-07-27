"""Concave-grasp sanity demo: watch the two-rectangle gripper close on a ``horseshoe`` /
``shoe`` / ``dumbbell``.

DD2D's gripper is the supporting-line abstraction of :mod:`blocks_tamp.dd2d.grasps`: rotate
the footprint by ``-alpha``, take its extreme-x lines, and sit a 2.5 x 2.0 cm finger
rectangle flush against each. A concave family has a feature (C-opening / L-corner / waist)
where the contact set on a supporting line is *disconnected*; the grasp model draws each
slide from the **intersection of the two lines' actual contact runs**, so both fingers land
on material -- never closing onto the gap. This demo renders that per grasp cell (the
supporting lines, the actual contact runs, and per-finger contact length), so you can see
the gripper make full flat contact **in the concave region** of the blocky ``horseshoe``
rather than grasping air. (Historically the model kept only the y-*hull* of each contact
set, which let a finger float across a concavity; :func:`finger_gaps` is the check that this
no longer happens.)

Two clips per sampled item, concatenated into one mp4:

* **isolation sweep** -- the item at the origin, the gripper stepping through its admissible
  ``(alpha, s)`` cells, fingers approaching along the grasp normal and closing flush. Draws
  the supporting lines, the *actual* (possibly disconnected) contact runs on them, the slide
  ticks, and per-finger contact status: green = the finger meets the footprint, orange = it
  would close onto a gap (should never fire for a returned cell now).
* **clutter search** -- the same item as the target of a small crowded drawer, animating
  what ``has_grasp`` actually does: walk the cells in order, drawing fingers red where they
  penetrate a neighbour/wall, until the first collision-free cell, then lift.

    python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.demo_grasp_concave
    # -> out_dd2d/grasp_demos/{horseshoe,shoe,dumbbell}_s{seed}.mp4  (+ a printed contact table)

Units: centimetres, matching the rest of DD2D.
"""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from shapely import box as shp_box
from shapely.affinity import rotate, translate
from shapely.geometry import LineString

from .grasps import (
    FINGER_THICK,
    FINGER_WIDTH,
    MAX_APERTURE,
    Grasp,
    _contact_runs_on_line,
    finger_rects,
    grasp_cells,
    grasp_cfree,
)
from .render import _write
from .scene import BUFFER_GAP, WALL_BAND
from .shapes import Shape, sample_shape
from .world import DrawerScene, ItemState, collar_pose, place_polygon, settle_pose

CONCAVE_FAMILIES = ("horseshoe", "shoe", "dumbbell")

_DPI = 100
_WIDTH = 780
_PAD = 4.0
_CONTACT_TOL = 1e-6  # below this the finger touches the footprint
_POINT_CONTACT_CM = (
    0.15  # a contact run shorter than this is a tangent point, not an edge
)

# animation shape
_APPROACH_FRAMES = 5
_HOLD_FRAMES = 9
_CLUTTER_FRAMES = 5
_LIFT_FRAMES = 10
_APPROACH_CM = 3.5  # how far outside the supporting line a finger starts

_ITEM_FACE = (
    "#7f8ec9"  # the item under inspection, in BOTH clips (identity carries over)
)
_CLUTTER_FACE = "#9aa4ad"
_DRAWER_FACE = "#eef2f6"
_WALL_FACE = "#4a4f57"
_BUFFER_FACE = "#f4efe2"
_LINE = "#888f99"
_CONTACT_RUN = "#2e9e4f"
_OK = "#2e9e4f"  # finger meets material
_GAP = "#e08a1e"  # finger closes onto a concavity gap
_BLOCKED = "#d62728"  # finger penetrates an obstacle


# --------------------------------------------------------------------------- #
# grasp diagnostics
# --------------------------------------------------------------------------- #
def _rot_frame(shape: Shape, g: Grasp):
    """The footprint in the grasp's rotated frame (supporting lines are vertical here)."""
    return rotate(shape.polygon, -g.alpha, origin=(0, 0), use_radians=True)


def _to_world(geom, g: Grasp, pose: tuple[float, float, float]):
    """Map rot-frame geometry to the world, exactly as ``finger_rects`` does."""
    x, y, theta = pose
    return translate(
        rotate(geom, g.alpha + theta, origin=(0, 0), use_radians=True), x, y
    )


def finger_gaps(shape: Shape, g: Grasp) -> tuple[float, float]:
    """Distance from each finger rectangle to the footprint, ``(left, right)`` in cm.

    ``0.0`` means the finger closes onto material. A positive value is the width of the
    concavity the finger closed across -- the supporting line touches the footprint
    *somewhere*, but not at this slide.
    """
    fp = place_polygon(shape.polygon, (0.0, 0.0, 0.0))
    left, right = finger_rects(g, (0.0, 0.0, 0.0))
    return (float(left.distance(fp)), float(right.distance(fp)))


def contact_runs(shape: Shape, g: Grasp, x: float) -> list[tuple[float, float]]:
    """The *actual* contact runs (y intervals) where the footprint meets the supporting
    line at ``x`` (in the grasp's rotated frame).

    Delegates to the grasp model's own :func:`grasps._contact_runs_on_line` so the demo
    draws exactly what the model uses to pick slides. More than one run on a line is the
    concave case; the model only emits a grasp where the two lines' runs y-overlap.
    """
    rot = _rot_frame(shape, g)
    ymin, ymax = rot.bounds[1], rot.bounds[3]
    return _contact_runs_on_line(rot, x, ymin, ymax)


def finger_contacts(shape: Shape, g: Grasp) -> tuple[float, float]:
    """Length of each finger face that actually lands on material, ``(left, right)`` cm.

    The whole contact run on a supporting line overstates this: the finger only spans
    ``s +/- FINGER_WIDTH/2``. A value near 0 with no gap is a *tangent point* -- the
    honest answer for a flat finger on a curved boundary.
    """
    lo_f, hi_f = g.s - FINGER_WIDTH / 2.0, g.s + FINGER_WIDTH / 2.0
    out = []
    for x in (g.xmin, g.xmax):
        out.append(
            sum(
                max(0.0, min(hi, hi_f) - max(lo, lo_f))
                for lo, hi in contact_runs(shape, g, x)
            )
        )
    return (out[0], out[1])


def is_internal_grasp(shape: Shape, g: Grasp, tol: float = 1e-3) -> bool:
    """Does this grasp pinch an *internal* feature (a finger reaches into a concavity)
    rather than the outer envelope? True iff ``[xmin, xmax]`` is strictly inside the
    footprint's rotated x-extent."""
    rb = _rot_frame(shape, g).bounds
    return g.xmin > rb[0] + tol or g.xmax < rb[2] - tol


def select_cells(shape: Shape, cells: list[Grasp], max_cells: int) -> list[Grasp]:
    """Up to ``max_cells`` cells for the sweep, always keeping >= 1 internal
    (concave-region) grasp when the shape has one.

    A shape can admit dozens of near-identical cells; showing every one makes a long,
    repetitive clip. Evenly spaced indices cover the direction range, and an internal
    grasp is forced in because gripping the concave region is the point of the demo.
    """
    if len(cells) <= max_cells:
        return list(cells)
    step = (len(cells) - 1) / (max_cells - 1)
    idx = {int(round(i * step)) for i in range(max_cells)}
    internal = [i for i, g in enumerate(cells) if is_internal_grasp(shape, g)]
    if internal and not any(i in idx for i in internal):
        idx.discard(max(idx))
        idx.add(internal[len(internal) // 2])
    return [cells[i] for i in sorted(idx)]


# --------------------------------------------------------------------------- #
# drawing primitives
# --------------------------------------------------------------------------- #
def _even(n: float) -> int:
    n = int(n)
    return n if n % 2 == 0 else n + 1


def _canvas(bounds, width: int):
    x0, y0, x1, y1 = bounds
    w = _even(width)
    h = max(_even(round(width * (y1 - y0) / (x1 - x0))), 200)
    fig = Figure(figsize=(w / _DPI, h / _DPI), dpi=_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    return fig, canvas, ax


def _rgb(fig, canvas) -> np.ndarray:
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    fig.clf()
    return rgb


def pad_to_common(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Letterbox every frame onto one canvas.

    The two clips are framed on different regions (an item-sized square vs a whole
    drawer), so their canvases differ in aspect; an encoder needs one size for the
    concatenation.
    """
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    out: list[np.ndarray] = []
    for f in frames:
        if f.shape[:2] == (h, w):
            out.append(f)
            continue
        canvas = np.full((h, w, 3), 255, dtype=f.dtype)
        top, left = (h - f.shape[0]) // 2, (w - f.shape[1]) // 2
        canvas[top : top + f.shape[0], left : left + f.shape[1]] = f
        out.append(canvas)
    return out


def _add_poly(ax, poly, **kw) -> None:
    if poly.is_empty:
        return
    for part in getattr(poly, "geoms", [poly]):
        if part.is_empty or not hasattr(part, "exterior"):
            continue
        ax.add_patch(MplPolygon(list(part.exterior.coords), closed=True, **kw))


def _banner(ax, bounds, text: str, va: str = "top", color: str = "#111") -> None:
    x0, y0, _, y1 = bounds
    ax.text(
        x0 + 1.0,
        (y1 - 1.0) if va == "top" else (y0 + 1.0),
        text,
        ha="left",
        va=va,
        fontsize=9,
        color=color,
        zorder=20,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
    )


def _offset_fingers(g: Grasp, pose: tuple[float, float, float], d: float):
    """The finger pair pulled ``d`` cm apart along the grasp normal (the approach)."""
    left, right = finger_rects(g, pose)
    ux, uy = math.cos(g.alpha + pose[2]), math.sin(g.alpha + pose[2])
    return translate(left, -d * ux, -d * uy), translate(right, d * ux, d * uy)


# --------------------------------------------------------------------------- #
# clip 1: isolation sweep
# --------------------------------------------------------------------------- #
def _isolation_bounds(shape: Shape) -> tuple[float, float, float, float]:
    r = shape.r_max + FINGER_THICK + _APPROACH_CM + 1.5
    return (-r, -r - 3.0, r, r + 3.0)  # extra headroom for the banners


def _draw_isolation(shape: Shape, g: Grasp, k: int, n: int, d: float, bounds):
    fig, canvas, ax = _canvas(bounds, _WIDTH)
    pose = (0.0, 0.0, 0.0)
    rot = _rot_frame(shape, g)
    ymin, ymax = rot.bounds[1], rot.bounds[3]
    gl, gr = finger_gaps(shape, g)

    # supporting lines, then the footprint, then the *actual* contact runs on top of it
    # (they sit on the footprint boundary, so anything below the item is invisible)
    for x in (g.xmin, g.xmax):
        line = _to_world(LineString([(x, ymin - 1.5), (x, ymax + 1.5)]), g, pose)
        ax.plot(*line.xy, color=_LINE, lw=1.0, ls="--", zorder=2)

    _add_poly(
        ax,
        place_polygon(shape.polygon, pose),
        facecolor=_ITEM_FACE,
        edgecolor="#2b3350",
        linewidth=1.4,
        zorder=4,
    )

    for x in (g.xmin, g.xmax):
        for lo, hi in contact_runs(shape, g, x):
            if hi - lo < _POINT_CONTACT_CM:
                # a supporting line meeting an arc touches at a point: a zero-length
                # segment draws nothing, so mark it, otherwise the contact looks absent
                dot = _to_world(LineString([(x, lo), (x, lo)]), g, pose)
                ax.plot(
                    *dot.xy,
                    color=_CONTACT_RUN,
                    marker="o",
                    markersize=6,
                    zorder=6,
                )
            else:
                run = _to_world(LineString([(x, lo), (x, hi)]), g, pose)
                ax.plot(
                    *run.xy, color=_CONTACT_RUN, lw=4.0, solid_capstyle="butt", zorder=6
                )

    # slide tick at the commanded contact height, on both lines
    for x in (g.xmin, g.xmax):
        tick = _to_world(LineString([(x - 0.45, g.s), (x + 0.45, g.s)]), g, pose)
        ax.plot(*tick.xy, color="#111", lw=1.4, zorder=6)

    left, right = _offset_fingers(g, pose, d)
    for finger, gap in ((left, gl), (right, gr)):
        color = _OK if gap <= _CONTACT_TOL else _GAP
        _add_poly(
            ax,
            finger,
            facecolor=color + "66",
            edgecolor=color,
            linewidth=1.6,
            zorder=8,
        )

    cl, cr = finger_contacts(shape, g)

    def status(side: str, gap: float, touched: float) -> str:
        if gap > _CONTACT_TOL:
            return f"{side}: GAP {gap:.2f} cm (closes on the concavity)"
        kind = (
            "point contact (flat finger tangent to a curve)"
            if touched < _POINT_CONTACT_CM
            else f"{touched:.2f} of {FINGER_WIDTH:.1f} cm finger on material"
        )
        return f"{side}: contact - {kind}"

    tag = (
        "\nINTERNAL GRASP - a finger reaches into the concave region"
        if is_internal_grasp(shape, g)
        else ""
    )
    _banner(
        ax,
        bounds,
        f"{shape.family}  |  grasp cell {k}/{n}\n"
        f"alpha {math.degrees(g.alpha):.0f} deg   width {g.width:.2f} / {MAX_APERTURE:.0f} cm"
        f"   slide {g.s:+.2f}{tag}",
    )
    worst = max(gl, gr)
    _banner(
        ax,
        bounds,
        f"{status('left ', gl, cl)}\n{status('right', gr, cr)}",
        va="bottom",
        color="#7a4a05" if worst > _CONTACT_TOL else "#1d6b36",
    )
    return _rgb(fig, canvas)


def isolation_frames(shape: Shape, cells: list[Grasp]) -> list[np.ndarray]:
    bounds = _isolation_bounds(shape)
    frames: list[np.ndarray] = []
    for k, g in enumerate(cells, start=1):
        for t in np.linspace(1.0, 0.0, _APPROACH_FRAMES):
            frames.append(
                _draw_isolation(
                    shape, g, k, len(cells), float(t) * _APPROACH_CM, bounds
                )
            )
        frames.extend([frames[-1]] * _HOLD_FRAMES)
    return frames


# --------------------------------------------------------------------------- #
# clip 2: the same item as a blocked target in a crowded drawer
# --------------------------------------------------------------------------- #
def demo_scene(shape: Shape, rng: random.Random, crowd: int = 5, clutter: int = 4):
    """A small drawer with ``shape`` as the target, pincered by a collar of neighbours.

    Built directly rather than via ``generate_scene`` because that generator picks the
    target family itself (and biases it round when crowding); here the concave item
    under inspection *is* the target.
    """
    w, d = 38.0, 30.0
    drawer = shp_box(0.0, 0.0, w, d)
    wall_band = shp_box(
        -WALL_BAND, -WALL_BAND, w + WALL_BAND, d + WALL_BAND
    ).difference(drawer)
    buffer_poly = shp_box(w + BUFFER_GAP, 0.0, w + BUFFER_GAP + 30.0, 14.0)

    pose = (w / 2.0, d / 2.0, rng.uniform(0.0, 2.0 * math.pi))
    items = {"target": ItemState("target", shape, pose, "drawer", is_target=True)}
    tcx, tcy = items["target"].footprint().centroid.coords[0]

    base = rng.uniform(0.0, 2.0 * math.pi)
    idx = 0
    for i in range(crowd):
        neighbour = sample_shape(rng, family="can")
        obstacles = [st.footprint() for st in items.values()] + [wall_band]
        npose = collar_pose(
            neighbour,
            drawer,
            obstacles,
            (tcx, tcy),
            base + 2.0 * math.pi * i / crowd,
            rng,
            backoff=0.06,
        )
        if npose is None:
            continue
        fp = place_polygon(neighbour.polygon, npose)
        if not drawer.buffer(1e-7).covers(fp) or any(
            fp.intersection(o).area > 1e-9 for o in obstacles
        ):
            continue
        items[f"o{idx}"] = ItemState(f"o{idx}", neighbour, npose, "drawer")
        idx += 1

    for _ in range(clutter):
        extra = sample_shape(rng)
        obstacles = [st.footprint() for st in items.values()] + [wall_band]
        npose = settle_pose(extra, drawer, obstacles, rng)
        if npose is None:
            continue
        fp = place_polygon(extra.polygon, npose)
        if not drawer.buffer(1e-7).covers(fp) or any(
            fp.intersection(o).area > 1e-9 for o in obstacles
        ):
            continue
        items[f"o{idx}"] = ItemState(f"o{idx}", extra, npose, "drawer")
        idx += 1

    return DrawerScene(
        drawer=drawer,
        wall_band=wall_band,
        buffer=buffer_poly,
        items=items,
        target="target",
        margin=1.0,
        dims={"W": w, "D": d},
    )


# (collar, settled-clutter) counts tried in order until the target is graspable
_DENSITY_LADDER = ((5, 4), (3, 4), (2, 3), (1, 2), (0, 2), (0, 0))


def graspable_demo_scene(
    shape: Shape,
    rng: random.Random,
    ladder: tuple[tuple[int, int], ...] = _DENSITY_LADDER,
) -> DrawerScene:
    """A demo scene the target can actually be grasped in, thinning the clutter until it
    is.

    A full collar usually pincers the target completely -- a real DD2D outcome, but a
    dull clip: every cell red and nothing picked up. It bites hardest exactly where we
    most want to see a pick, since a C-shaped horseshoe admits only a handful of cells.
    Walking the density down keeps some cells blocked (what we want to see) while
    ending in a grasp. The sparsest rung is returned regardless, so a target that is
    genuinely ungraspable even in near-isolation still renders its blocked verdict.
    """
    scene = demo_scene(shape, rng, *ladder[-1])
    for crowd, clutter in ladder:
        scene = demo_scene(shape, rng, crowd=crowd, clutter=clutter)
        obstacles = [st.footprint() for n, st in scene.items.items() if n != "target"]
        obstacles.append(scene.wall_band)
        pose = scene.items["target"].pose
        if any(grasp_cfree(g, pose, obstacles) for g in grasp_cells(shape)):
            return scene
    return scene


def _scene_bounds(scene: DrawerScene):
    x0, y0, x1, y1 = scene.drawer.bounds
    return (x0 - _PAD, y0 - _PAD - 3.0, x1 + _PAD, y1 + _PAD + 3.0)


def _draw_clutter(scene, shape, g, fingers, colors, header, footer, lift, bounds):
    fig, canvas, ax = _canvas(bounds, _WIDTH)
    # the wall band is a ring (a polygon with a hole) and _add_poly only fills exteriors,
    # so paint it first and let the drawer interior cover the hole -- otherwise the band
    # renders as a solid slab over the whole scene
    _add_poly(ax, scene.wall_band, facecolor=_WALL_FACE, edgecolor="none", zorder=0)
    _add_poly(
        ax,
        scene.drawer,
        facecolor=_DRAWER_FACE,
        edgecolor="#333",
        linewidth=1.2,
        zorder=1,
    )
    for st in scene.items.values():
        if st.is_target and lift > 0.0:
            # elevated-carry convention (render.py): a lifted item is a dashed no-fill
            # outline over a drop shadow, never an item that simply disappears
            _add_poly(
                ax,
                translate(st.footprint(), 0.6 + 2.5 * lift, -(0.9 + 3.5 * lift)),
                facecolor="#00000022",
                edgecolor="none",
                zorder=3,
            )
            _add_poly(
                ax,
                st.footprint(),
                facecolor="none",
                edgecolor=_ITEM_FACE,
                linewidth=2.2,
                linestyle="--",
                zorder=6,
            )
            continue
        _add_poly(
            ax,
            st.footprint(),
            facecolor=_ITEM_FACE if st.is_target else _CLUTTER_FACE,
            edgecolor="#2b3350" if st.is_target else "#555",
            linewidth=1.6 if st.is_target else 0.9,
            zorder=4 if st.is_target else 3,
        )
    if g is not None and lift <= 0.0:
        pose = scene.items["target"].pose
        rot = _rot_frame(shape, g)
        ymin, ymax = rot.bounds[1], rot.bounds[3]
        for x in (g.xmin, g.xmax):
            line = _to_world(LineString([(x, ymin), (x, ymax)]), g, pose)
            ax.plot(*line.xy, color=_LINE, lw=0.9, ls="--", zorder=5)
    for finger, color in zip(fingers, colors):
        _add_poly(
            ax,
            finger,
            facecolor=color + "66",
            edgecolor=color,
            linewidth=1.6,
            zorder=9,
        )
    _banner(ax, bounds, header)
    if footer:
        _banner(ax, bounds, footer[0], va="bottom", color=footer[1])
    return _rgb(fig, canvas)


def clutter_frames(scene: DrawerScene, shape: Shape) -> tuple[list[np.ndarray], str]:
    """Animate ``has_grasp``'s actual search: cells in order until one clears."""
    bounds = _scene_bounds(scene)
    pose = scene.items["target"].pose
    obstacles = [st.footprint() for n, st in scene.items.items() if n != "target"] + [
        scene.wall_band
    ]
    cells = grasp_cells(shape)
    frames: list[np.ndarray] = []
    chosen: Grasp | None = None

    for k, g in enumerate(cells, start=1):
        free = grasp_cfree(g, pose, obstacles)
        gl, gr = finger_gaps(shape, g)
        left, right = finger_rects(g, pose)
        if free:
            colors = [
                _OK if gl <= _CONTACT_TOL else _GAP,
                _OK if gr <= _CONTACT_TOL else _GAP,
            ]
            footer = ("collision-free -> has_grasp returns this cell", "#1d6b36")
        else:
            colors = [_BLOCKED, _BLOCKED]
            footer = (
                "fingers hit a neighbour / the wall -> try the next cell",
                "#8b1a1a",
            )
        header = (
            f"{shape.family} target in clutter  |  has_grasp: cell {k}/{len(cells)}\n"
            f"alpha {math.degrees(g.alpha):.0f} deg   width {g.width:.2f} cm"
        )
        frame = _draw_clutter(
            scene, shape, g, [left, right], colors, header, footer, 0.0, bounds
        )
        frames.extend([frame] * _CLUTTER_FRAMES)
        if free:
            chosen = g
            frames.extend([frame] * _HOLD_FRAMES)
            break

    if chosen is None:
        frame = _draw_clutter(
            scene,
            shape,
            None,
            [],
            [],
            f"{shape.family} target in clutter",
            (
                f"BLOCKED - no collision-free cell in {len(cells)}: the target must be "
                "cleared first",
                "#8b1a1a",
            ),
            0.0,
            bounds,
        )
        frames.extend([frame] * (_HOLD_FRAMES * 3))
        return frames, "blocked"

    for t in np.linspace(0.0, 1.0, _LIFT_FRAMES):
        left, right = finger_rects(chosen, pose)
        frames.append(
            _draw_clutter(
                scene,
                shape,
                chosen,
                [left, right],
                [_OK, _OK],
                f"{shape.family} target in clutter  |  lifting (elevated carry)",
                ("grasped and extracted", "#1d6b36"),
                float(t),
                bounds,
            )
        )
    frames.extend([frames[-1]] * _HOLD_FRAMES)
    return frames, "grasped"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def render_demo(
    family: str,
    seed: int,
    out_dir: str,
    max_cells: int = 8,
    fps: int = 12,
    fmt: str = "mp4",
) -> dict:
    """Render one sampled item of ``family`` to a single mp4; return its contact stats."""
    rng = random.Random(seed)
    shape = sample_shape(rng, family=family)
    cells = grasp_cells(shape)
    if not cells:  # pragma: no cover - sample_shape guarantees >= 1 in isolation
        raise RuntimeError(f"{family} seed {seed} admits no grasp cells")

    shown = select_cells(shape, cells, max_cells)
    frames = isolation_frames(shape, shown)
    scene = graspable_demo_scene(shape, rng)
    clutter, verdict = clutter_frames(scene, shape)
    frames.extend(clutter)

    path = _write(
        pad_to_common(frames),
        os.path.join(out_dir, f"{family}_s{seed}.{fmt}"),
        fps,
        fmt,
    )
    gaps = [max(finger_gaps(shape, g)) for g in cells]
    solid = [
        min(finger_contacts(shape, g))
        for g, gap in zip(cells, gaps)
        if gap <= _CONTACT_TOL
    ]
    return {
        "family": family,
        "seed": seed,
        "path": path,
        "size": shape.size,
        "n_cells": len(cells),
        "n_shown": len(shown),
        "n_floating": sum(1 for gap in gaps if gap > _CONTACT_TOL),
        "max_gap": max(gaps, default=0.0),
        # weakest both-fingers-touching cell: how little of a finger can be on material
        "min_contact": min(solid, default=0.0),
        "n_internal": sum(1 for g in cells if is_internal_grasp(shape, g)),
        "clutter": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--families",
        nargs="+",
        default=list(CONCAVE_FAMILIES),
        help=f"shape families to demo (default: the concave ones, {CONCAVE_FAMILIES})",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=3,
        help="items sampled per family (one clip each)",
    )
    ap.add_argument("--seed", type=int, default=0, help="first sample seed")
    ap.add_argument(
        "--max-cells",
        type=int,
        default=8,
        help="grasp cells shown in the isolation sweep (an internal/concave-region grasp is always kept)",
    )
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--format", dest="fmt", choices=["mp4", "gif"], default="mp4")
    ap.add_argument("--out-dir", default="out_dd2d/grasp_demos")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    stats = []
    for family in args.families:
        for i in range(args.samples):
            row = render_demo(
                family,
                args.seed + i,
                args.out_dir,
                max_cells=args.max_cells,
                fps=args.fps,
                fmt=args.fmt,
            )
            stats.append(row)
            print(f"wrote {row['path']}")

    print(
        f"\n# Gripper contact check over all {sum(r['n_cells'] for r in stats)} admissible "
        f"grasp cells.\n"
        f"#   floating = cells where a finger closes onto a concavity instead of material\n"
        f"#   max gap  = how far it closes across, cm\n"
        f"#   min touch= weakest both-fingers-touching cell: cm of the "
        f"{FINGER_WIDTH:.1f} cm finger face on material\n"
        f"#   internal = cells that grip a concave-region sub-feature (finger in the concavity)\n"
        f"{'family':<10}{'seed':>5}{'size (cm)':>14}{'cells':>7}{'floating':>10}"
        f"{'max gap':>9}{'min touch':>11}{'internal':>10}   in clutter"
    )
    for r in stats:
        w, h = r["size"]
        floating = f"{r['n_floating']}/{r['n_cells']}"
        internal = f"{r['n_internal']}/{r['n_cells']}"
        print(
            f"{r['family']:<10}{r['seed']:>5}{f'{w:.1f}x{h:.1f}':>14}{r['n_cells']:>7}"
            f"{floating:>10}{r['max_gap']:>9.2f}{r['min_contact']:>11.2f}{internal:>10}"
            f"   {r['clutter']}"
        )
    print(f"\n# {len(stats)} clips in {args.out_dir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
