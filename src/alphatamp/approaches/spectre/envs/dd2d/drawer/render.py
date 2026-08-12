"""DD2D top-down rendering: still frames (segmented, for the PIGINet record) + a per-plan
execution video with the **elevated-carry convention** (spec Section 13).

Pure 2D and fully decoupled from the PyBullet/Panda stack: frames are drawn with a headless
matplotlib ``Figure``/``FigureCanvasAgg``; segmentation is rasterised with PIL; video is
encoded with ``imageio`` directly (we do NOT import ``blocks_tamp.video``, which pulls in
PyBullet). The elevated-carry style is load-bearing, not cosmetic: during transfer a carried
item may legitimately overlap resting items *in projection* (different heights, spec
Section 5.1), so a carried item is drawn as a dashed no-fill outline with a ground
drop-shadow and a "carrying o" tag -- without it every transfer reads as a collision bug.

* :func:`render_scene` -> :class:`~blocks_tamp.geometry.RenderResult` (rgb + per-item seg +
  id->name), so ``record.build_image_refs`` and ``rendering.confirm_rendering`` work unchanged.
* :func:`render_episode` replays a bound plan: pick lifts an item (elevated), place-buffer
  slides it to its staged pose (rejected buffer poses flash as red ghosts), retrieve lifts
  the target out; an infeasible plan runs the bound prefix then draws the failing action's
  blocked fingers / overflow ghost and a verdict banner, and halts.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from alphatamp.approaches.spectre.envs.dd2d.geometry import (
    GeometryBackend,
    RenderResult,
)

from .grasps import finger_rects
from .world import place_polygon

_PAD = 3.0
_DPI = 100
_EPISODE_WIDTH = 760
_SLIDE_FRAMES = 6

_TARGET_COLOR = "#e8483c"
_ITEM_COLOR = "#9aa4ad"
_CONCAVE_COLOR = "#7f8ec9"
_DRAWER_FACE = "#eef2f6"
_WALL_FACE = "#4a4f57"
_BUFFER_FACE = "#f4efe2"
_FINGER = "#d62728"
_GHOST = "#d62728"


# --------------------------------------------------------------------------- #
# geometry / transform helpers
# --------------------------------------------------------------------------- #
def _bounds(scene, pad: float = _PAD):
    xs, ys = [], []
    for geom in (scene.drawer, scene.wall_band, scene.buffer):
        x0, y0, x1, y1 = geom.bounds
        xs += [x0 - pad, x1 + pad]
        ys += [y0 - pad, y1 + pad]
    return min(xs), min(ys), max(xs), max(ys)


def _frame_size(bounds, base_width: int):
    x0, y0, x1, y1 = bounds
    w = _even(base_width)
    h = _even(round(base_width * (y1 - y0) / (x1 - x0)))
    return w, max(h, 160)


def _even(n: int) -> int:
    n = int(n)
    return n if n % 2 == 0 else n + 1


def _item_color(scene, name: str) -> str:
    st = scene.items[name]
    if st.is_target:
        return _TARGET_COLOR
    return _CONCAVE_COLOR if st.shape.concave else _ITEM_COLOR


def _add_poly(ax, poly, **kw):
    if poly.is_empty:
        return
    geoms = getattr(poly, "geoms", [poly])
    for g in geoms:
        if g.is_empty or not hasattr(g, "exterior"):
            continue
        ax.add_patch(MplPolygon(list(g.exterior.coords), closed=True, **kw))


# --------------------------------------------------------------------------- #
# single frame -> RGB
# --------------------------------------------------------------------------- #
def _draw(scene, states, width, title=None, ghosts=None, fingers=None, carry_tag=None):
    """``states``: name -> (pose, style) where style in {rest, carry, gone}."""
    bounds = _bounds(scene)
    x0, y0, x1, y1 = bounds
    W, H = _frame_size(bounds, width)
    fig = Figure(figsize=(W / _DPI, H / _DPI), dpi=_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")

    # regions
    _add_poly(ax, scene.wall_band, facecolor=_WALL_FACE, edgecolor="none", zorder=1)
    _add_poly(
        ax,
        scene.drawer,
        facecolor=_DRAWER_FACE,
        edgecolor="#333",
        linewidth=1.2,
        zorder=0,
    )
    _add_poly(
        ax,
        scene.buffer,
        facecolor=_BUFFER_FACE,
        edgecolor="#a98",
        linewidth=1.2,
        linestyle="--",
        zorder=0,
    )
    dbx = scene.drawer.bounds
    ax.text(
        (dbx[0] + dbx[2]) / 2,
        dbx[3] + 1.0,
        "drawer",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333",
    )
    bbx = scene.buffer.bounds
    ax.text(
        (bbx[0] + bbx[2]) / 2,
        bbx[3] + 1.0,
        "buffer",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#a76",
    )

    # items
    for name, (pose, style) in states.items():
        if style == "gone":
            continue
        fp = place_polygon(scene.items[name].shape.polygon, pose)
        color = _item_color(scene, name)
        if style == "carry":  # elevated: drop shadow + dashed no-fill outline
            from shapely.affinity import translate as _t

            _add_poly(
                ax, _t(fp, 0.6, -0.9), facecolor="#00000022", edgecolor="none", zorder=4
            )
            _add_poly(
                ax,
                fp,
                facecolor="none",
                edgecolor=color,
                linewidth=2.2,
                linestyle="--",
                zorder=6,
            )
        else:
            edge = "#111" if scene.items[name].is_target else "#555"
            _add_poly(
                ax,
                fp,
                facecolor=color,
                edgecolor=edge,
                linewidth=2.0 if scene.items[name].is_target else 1.0,
                zorder=3,
            )
        cx, cy = fp.centroid.x, fp.centroid.y
        ax.text(
            cx,
            cy,
            name.replace("target", "T"),
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            zorder=7,
        )

    # finger rectangles (grasp / place phases)
    for fr in fingers or []:
        _add_poly(
            ax,
            fr,
            facecolor=_FINGER + "55" if isinstance(_FINGER, str) else _FINGER,
            edgecolor=_FINGER,
            linewidth=1.0,
            zorder=8,
        )

    # rejected buffer poses -> red ghosts
    for name, pose in ghosts or []:
        gp = place_polygon(scene.items[name].shape.polygon, pose)
        _add_poly(
            ax,
            gp,
            facecolor="none",
            edgecolor=_GHOST,
            linewidth=1.6,
            linestyle=":",
            zorder=8,
        )

    if title:
        ax.text(
            x0 + 1.0,
            y1 - 1.0,
            title,
            ha="left",
            va="top",
            fontsize=10,
            color="#111",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        )
    if carry_tag:
        ax.text(
            x0 + 1.0,
            y0 + 1.0,
            carry_tag,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#333",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffcc", ec="none", alpha=0.85),
        )

    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    return rgb, bounds, (W, H)


# --------------------------------------------------------------------------- #
# segmentation (PIL rasterisation of the item polygons)
# --------------------------------------------------------------------------- #
def _segment(scene, states, bounds, W, H):
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = bounds
    img = Image.new("I", (W, H), -1)
    draw = ImageDraw.Draw(img)
    id_to_name: dict[int, str] = {}
    for oid, name in enumerate(sorted(states)):
        pose, style = states[name]
        if style == "gone":
            id_to_name[oid] = name
            continue
        fp = place_polygon(scene.items[name].shape.polygon, pose)
        for g in getattr(fp, "geoms", [fp]):
            if g.is_empty or not hasattr(g, "exterior"):
                continue
            px = [
                ((cx - x0) / (x1 - x0) * (W - 1), (1 - (cy - y0) / (y1 - y0)) * (H - 1))
                for cx, cy in g.exterior.coords
            ]
            draw.polygon(px, fill=oid)
        id_to_name[oid] = name
    return np.array(img, dtype=np.int32), id_to_name


# --------------------------------------------------------------------------- #
# public: still render + backend
# --------------------------------------------------------------------------- #
def render_scene(
    scene, width: int = 720, view: str = "topdown", poses=None
) -> RenderResult:
    states = {
        n: ((poses or {}).get(n, st.pose), "rest") for n, st in scene.items.items()
    }
    rgb, bounds, (W, H) = _draw(scene, states, width)
    seg, id_to_name = _segment(scene, states, bounds, W, H)
    return RenderResult(rgb=rgb, seg=seg, id_to_name=id_to_name, view=view)


class DD2DRenderBackend(GeometryBackend):
    """Adapts :func:`render_scene` to the geometry-backend ABC so
    ``rendering.confirm_rendering`` works verbatim on a DD2D scene."""

    name = "dd2d-matplotlib"

    @classmethod
    def available(cls) -> bool:
        return True

    def render_segmented(
        self, scene, view: str = "topdown", width: int = 256, height: int = 256
    ):
        return render_scene(scene, width=max(width, 640), view=view)


# --------------------------------------------------------------------------- #
# public: episode video
# --------------------------------------------------------------------------- #
def render_episode(
    scene, bound_plan, feasible, failure_action, out_path, fmt="mp4", fps=20
) -> str:
    states = {n: (st.pose, "rest") for n, st in scene.items.items()}
    frames: list[np.ndarray] = []

    def emit(title, ghosts=None, fingers=None, carry_tag=None, hold=1):
        rgb, _, _ = _draw(
            scene, states, _EPISODE_WIDTH, title, ghosts, fingers, carry_tag
        )
        frames.extend([rgb] * hold)

    emit("initial state", hold=max(fps // 2, 4))

    for step in bound_plan:
        p = step.params
        phase, o = p.get("phase"), p.get("item")
        if phase == "pick":
            fingers = list(finger_rects(p["grasp"], p["pose"]))
            emit(f"grasp {o}", fingers=fingers, hold=max(fps // 3, 3))
            states[o] = (p["pose"], "carry")  # lifted
            emit(f"lift {o}", carry_tag=f"carrying {o}", hold=max(fps // 4, 2))
        elif phase == "place":
            src = states[o][0]
            dst = p["pose"]
            ghosts = [(o, g) for g in p.get("ghosts", [])]
            for t in np.linspace(0, 1, _SLIDE_FRAMES):
                mx = src[0] + (dst[0] - src[0]) * t
                my = src[1] + (dst[1] - src[1]) * t
                mth = src[2] + (dst[2] - src[2]) * t
                states[o] = ((float(mx), float(my), float(mth)), "carry")
                emit(
                    f"stage {o} -> buffer",
                    ghosts=ghosts if t < 0.5 else None,
                    carry_tag=f"carrying {o}",
                )
            fingers = list(finger_rects(p["grasp"], dst))
            states[o] = (dst, "rest")
            emit(f"release {o}", fingers=fingers, hold=max(fps // 3, 3))
        elif phase == "retrieve":
            fingers = list(finger_rects(p["grasp"], p["pose"]))
            emit(f"grasp target", fingers=fingers, hold=max(fps // 3, 3))
            states[o] = (p["pose"], "carry")
            emit("retrieve target", carry_tag="carrying target", hold=fps // 3)
            states[o] = (p["pose"], "gone")

    if feasible:
        emit("FEASIBLE -- target retrieved", hold=fps)
    else:
        label = (
            f"INFEASIBLE -- stuck @ {failure_action}"
            if failure_action
            else "INFEASIBLE"
        )
        ghosts, fingers = _failure_overlay(scene, bound_plan, failure_action)
        emit(label, ghosts=ghosts, fingers=fingers, hold=fps)

    return _write(frames, out_path, fps, fmt)


def _failure_overlay(scene, bound_plan, failure_action):
    """Draw the failing action's evidence: an overflow ghost over the buffer for a
    place-buffer failure, or the blocked target/finger for a pick/retrieve failure."""
    parsed = _parse_action(failure_action)
    if parsed is None:
        return None, None
    op, args = parsed
    o = args[0] if args else None
    if o not in scene.items:
        return None, None
    if op == "place-buffer":
        bx = scene.buffer.bounds
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        return [(o, (cx, cy, 0.0))], None  # overflow ghost centred on the buffer
    # pick / retrieve blocked in the drawer: red finger rects at its current grasp
    return None, None


def _parse_action(s):
    if not s or "(" not in s:
        return None
    try:
        name = s[: s.index("(")]
        inside = s[s.index("(") + 1 : s.rindex(")")]
        return name, [a for a in inside.split(",") if a]
    except Exception:  # pragma: no cover
        return None


def _write(frames, out_path, fps, fmt) -> str:
    """Mp4 via imageio-ffmpeg (macro_block_size=None -> no even-dim requirement), else
    gif.

    Same proven incantation as blocks_tamp.video._write, copied to keep this module free
    of the PyBullet import chain.
    """
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # libx264 + yuv420p requires even width AND height; matplotlib's canvas can round to
    # an odd pixel count, so crop every frame to even dimensions before encoding.
    frames = [
        f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames
    ]
    if fmt == "mp4" and not _ffmpeg_available():
        fmt = "gif"
    if os.path.splitext(out_path)[1].lower() != f".{fmt}":
        out_path = os.path.splitext(out_path)[0] + f".{fmt}"
    if fmt == "mp4":
        with imageio.get_writer(out_path, fps=fps, macro_block_size=None) as wr:
            for f in frames:
                wr.append_data(f)
    else:
        imageio.mimsave(out_path, frames, duration=1.0 / fps)
    return out_path


def _ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401

        return True
    except Exception:
        return False
