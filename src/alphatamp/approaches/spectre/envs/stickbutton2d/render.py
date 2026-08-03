"""Labelled top-down render of a StickButton2D scene, from stored geometry.

Two consumers, one renderer: the VLMPlan prompt's attached image
(``vlmplan/sb2d_adapter.py``) and the comparison notebook's planner inspector. They must
show the *same* picture — a reviewer looking at the inspector should be seeing what the
model saw — which is the reason this is a module rather than two plotting snippets.

Everything comes from ``EpisodeRecord.scene_geometry``: boundary rings, poses, and the
table/world containers. Nothing re-runs the environment (*reconstruct, never
regenerate*, ``decisions/03`` 2026-07-19), so a render is reproducible from an episode
file alone.

**Object names are drawn on the image.** They are the canonicalized names
(``circle_0``…), the same strings the prompt's object list, the parser's vocabulary and
the pool indices use. A render that labelled objects differently from the text would let
the model name something it cannot see, which is the failure the DD2D renderer exists to
prevent.

The **reach boundary is drawn too** — a dashed line at ``robot_reach_max_y``. It is not
decoration: StickButton2D's symbolic model is reach-blind, so a picture without it
depicts a problem in which every plan is equally good. It is the visual counterpart of
the reach disclosure the text prompt carries.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

from alphatamp.approaches.spectre.envs.stickbutton2d.geometry import (  # noqa: E402
    robot_reach_max_y,
)
from alphatamp.approaches.spectre.schema import SceneGeometry  # noqa: E402

# Keyed on the object's **type**, not on `ObjectGeometry.family`. The robot's footprint
# is recorded as a disc, so its family is `circle` — the same as a button — and
# colouring by family paints the robot button-red, i.e. shows the model a sixth button.
# Caught by looking at the render, which is the only way this class of bug is ever
# caught.
_TYPE_FACE = {
    "circle": "#e51400",  # button_unpressed_rgb
    "rectangle": "#66331a",  # stick_rgb
    "crv_robot": "#3a53c8",
}
_UNKNOWN_FACE = "#999999"
_TABLE_FACE = "#2b2b2b"
_BG = "#f5f5f5"


def _type_of(name: str, types: dict[str, str] | None, fallback: str) -> str:
    """Object type, from the episode's registry when available.

    Falls back to the canonical-name prefix (``crv_robot_0`` -> ``crv_robot``) and then
    to the geometry family, so a render still works from a bare ``SceneGeometry``.
    """
    if types and name in types:
        return types[name]
    stem = name.rsplit("_", 1)[0]
    return stem if stem in _TYPE_FACE else fallback


def scene_figure(
    geometry: SceneGeometry,
    types: dict[str, str] | None = None,
    width_in: float = 7.2,
    label_fontsize: float = 11.0,
) -> plt.Figure:
    """Matplotlib figure of one scene: table band, objects, labels, reach line.

    ``types`` is ``EpisodeRecord.object_registry``; see :func:`_type_of` for why colour
    cannot be taken from the geometry family.
    """
    world = next((c for c in geometry.containers if c.kind == "world"), None)
    x0, y0, x1, y1 = world.bounds if world else (0.0, 0.0, 3.5, 2.5)
    aspect = (y1 - y0) / max(x1 - x0, 1e-9)
    fig, ax = plt.subplots(figsize=(width_in, width_in * aspect))
    ax.set_facecolor(_BG)

    table = next((c for c in geometry.containers if c.kind == "table"), None)
    if table is not None:
        tx0, ty0, tx1, ty1 = table.bounds
        ax.add_patch(
            plt.Rectangle(
                (tx0, ty0), tx1 - tx0, ty1 - ty0, facecolor=_TABLE_FACE, zorder=0
            )
        )
        ax.text(
            tx0 + 0.05,
            ty1 - 0.06,
            "table — the robot base cannot drive onto it",
            ha="left",
            va="top",
            fontsize=label_fontsize * 0.8,
            color="#dddddd",
            zorder=1,
        )

    reach = robot_reach_max_y()
    ax.axhline(
        reach,
        linestyle="--",
        linewidth=1.2,
        color="#0066cc",
        zorder=2,
    )
    ax.text(
        x0 + 0.05,
        reach + 0.03,
        f"robot arm reach limit (y = {reach:.2f}) — above this, use the stick",
        fontsize=label_fontsize * 0.8,
        color="#0066cc",
        zorder=3,
    )

    # Label placement is de-collided greedily. Buttons sit close together and the robot
    # starts beside one of them, so the naive "just put the name above the dot"
    # placement stacks two labels on the same pixels -- and a label a reader cannot
    # attribute to an object is worse than no label, because the model will attribute
    # it to *something*.
    placed: list[tuple[float, float, float]] = []  # (x, y, half-width)
    for obj in geometry.objects:
        face = _TYPE_FACE.get(_type_of(obj.name, types, obj.family), _UNKNOWN_FACE)
        ring = [(px + obj.pose[0], py + obj.pose[1]) for px, py in obj.boundary]
        ax.add_patch(
            Polygon(
                ring, closed=True, facecolor=face, edgecolor="black", lw=0.8, zorder=4
            )
        )
        # Offset by the object's own half-height so a big footprint (the robot, the
        # 1.25-long stick) does not have its label sitting on top of it.
        half_h = max((py for _, py in obj.boundary), default=0.05)
        # Rough text extent in data units; good enough only to detect overlap.
        half_w = 0.5 * len(obj.name) * label_fontsize * (x1 - x0) / (width_in * 132.0)
        lx, ly = obj.pose[0], obj.pose[1] + half_h + 0.05
        step = 0.09
        for _ in range(8):
            if not any(
                abs(lx - px) < (half_w + pw) and abs(ly - py) < step
                for px, py, pw in placed
            ):
                break
            ly += step
        placed.append((lx, ly, half_w))
        ax.annotate(
            obj.name,
            xy=(obj.pose[0], obj.pose[1] + half_h),
            xytext=(lx, ly),
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color="black",
            zorder=5,
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 1.0, "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#555555"},
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_labeled_scene(
    geometry: SceneGeometry,
    types: dict[str, str] | None = None,
    width_px: int = 1024,
    label_fontsize: float = 11.0,
) -> Image.Image:
    """:func:`scene_figure` rasterised to a PIL image ``width_px`` wide.

    Size comes from dpi rather than upscaling, so labels keep the same *physical* size
    as the width grows — the same reason DD2D's renderer does it this way.
    """
    width_in = 7.2
    fig = scene_figure(
        geometry, types, width_in=width_in, label_fontsize=label_fontsize
    )
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=width_px / width_in)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    finally:
        plt.close(fig)


def _annotate_scene(ax, geometry: SceneGeometry, label_fontsize: float) -> None:
    """Overlay Set-of-Mark object labels on an axes kinder drew on.

    The label strings are the canonicalized object names (``circle_0``…), identical to
    the prompt's object list and the parser vocabulary — that identity is the whole point,
    it is what lets the model name a disc it can see.

    Placement is edge-aware: a label for an object near the top of the frame is dropped
    *below* it (and vice versa), and every label is clamped inside the frame, so a label
    never runs off the top edge — buttons cluster along the top wall on b5, so the naive
    "always above" placement clipped them.

    ``label_fontsize`` is small on purpose (the button markers are tiny discs, so a large
    label reads as bigger than the thing it names). No reach line is drawn — the table
    band already shows the base-exclusion zone and the exact numeric reach limit is stated
    in the text prompt, so the render stays uncluttered.
    """
    world = next((c for c in geometry.containers if c.kind == "world"), None)
    x0, y0, x1, y1 = world.bounds if world else (0.0, 0.0, 3.5, 2.5)
    xr, yr = (x1 - x0), (y1 - y0)
    placed: list[tuple[float, float, float]] = []  # (x, y, half-width)
    # The kinder figure is `xr` inches wide, so an inch is ~one data unit; estimate the
    # label's half-width in data units from its character count for de-collision.
    for obj in geometry.objects:
        half_h = max((py for _, py in obj.boundary), default=0.05)
        half_w = 0.5 * len(obj.name) * label_fontsize * 0.62 / 72.0
        gap = 0.055 * yr
        # Below objects near the top, above otherwise, so nothing clips the top edge.
        below = obj.pose[1] > y0 + 0.62 * yr
        step = -gap if below else gap
        lx = min(max(obj.pose[0], x0 + half_w + 0.01 * xr), x1 - half_w - 0.01 * xr)
        ly = obj.pose[1] + (-half_h - 0.5 * gap if below else half_h + 0.5 * gap)
        for _ in range(8):
            if not any(
                abs(lx - px) < (half_w + pw) and abs(ly - py) < abs(step)
                for px, py, pw in placed
            ):
                break
            ly += step
        ly = min(max(ly, y0 + 0.03 * yr), y1 - 0.03 * yr)
        placed.append((lx, ly, half_w))
        ax.annotate(
            obj.name,
            xy=(obj.pose[0], obj.pose[1] + (-half_h if below else half_h)),
            xytext=(lx, ly),
            ha="center",
            va=("top" if below else "bottom"),
            fontsize=label_fontsize,
            color="black",
            zorder=8,
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 0.6, "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#444444"},
            annotation_clip=False,
        )


# One kinder env per button count, reused across a run's renders (kinder.make is slow).
_ENV_CACHE: dict[int, Any] = {}


def render_kinder_labeled_scene(
    episode,
    width_px: int = 1024,
    label_fontsize: float = 7.0,
) -> Image.Image:
    """Kinder's own initial-scene pixels with Set-of-Mark labels overlaid.

    This is the SB2D VLMPlan image for the ``stickbutton2d_v1_kinder`` arm: the *real*
    environment render (identical to what PIGINet's crops are sourced from), so the
    representation contrast is measured on the same pixels — but kinder draws every
    unpressed button as an identical unlabeled red disc, which a VLM cannot ground. The
    labels are drawn in **data coordinates via kinder's own ``ax_callback``**, so they sit
    exactly on the objects with no pixel-transform guesswork.

    Reconstructs the env from the stored seed (``env.reset(seed=problem_id)``), the one
    sanctioned exception to *reconstruct, never regenerate*, exactly as
    ``sb2d_render_convert.py`` does. Object *positions and names* come from the stored
    ``scene_geometry`` (canonical names matching the prompt); only the *pixels* come from
    kinder.
    """
    # pylint: disable=import-outside-toplevel
    import kinder
    from kinder.envs.utils import render_2dstate
    from PIL import Image as _Image

    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.stickbutton2d.strata import env_id

    geometry = episode.scene_geometry
    if geometry is None:
        raise ValueError("SB2D episode has no scene_geometry to render from")
    num_buttons = sum(1 for t in episode.object_registry.values() if t == "circle")
    if num_buttons not in _ENV_CACHE:
        register_extra_envs()
        _ENV_CACHE[num_buttons] = kinder.make(
            env_id(num_buttons), render_mode="rgb_array"
        )
    env = _ENV_CACHE[num_buttons]
    env.reset(seed=int(episode.provenance.problem_id))
    oc = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    state = oc._current_state.copy()  # pylint: disable=protected-access
    state.data.update(oc.initial_constant_state.data)
    cache = oc._static_object_body_cache  # pylint: disable=protected-access
    cfg = oc.config
    rgb = render_2dstate(
        state,
        cache,
        cfg.world_min_x,
        cfg.world_max_x,
        cfg.world_min_y,
        cfg.world_max_y,
        cfg.render_dpi,
        ax_callback=lambda ax: _annotate_scene(ax, geometry, label_fontsize),
    )
    img = _Image.fromarray(rgb).convert("RGB")
    if img.width != width_px:
        height = round(img.height * width_px / img.width)
        img = img.resize((width_px, height), _Image.Resampling.LANCZOS)
    return img
