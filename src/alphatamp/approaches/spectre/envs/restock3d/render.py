"""Oblique-camera rendering + world->pixel projection for Restock3D v2.

Shared by the PIGINet baseline (per-object crops) and the VLMPlan baseline (a full-scene
labeled snapshot). The env's own camera is oblique (yaw 55, pitch -28), so a rendered
scene shows **height** -- a cube and a tall block are visually distinct, unlike a top-
down view where they are identical squares. That is the F3 axis both image-consuming
baselines must be able to see (docs/decisions 2026-08-18).

The projection replicates ``pybullet_helpers.camera.capture_image``'s view/projection
matrices so a world point can be mapped to a pixel (for Set-of-Mark labels and per-
object crop centres) without depending on GL read-back.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pybullet as p
from pybullet_helpers.camera import capture_image

from alphatamp.approaches.spectre.schema import SceneGeometry

#: Top-down footprint colours by object family (matches the §5 legend text).
_FAMILY_COLOR = {
    "tall": "#1f77b4",  # tall block -- the F3 axis; darker/blue
    "cube": "#ff7f0e",  # short cube -- orange
    "clutter": "#9467bd",
    "robot": "#7f7f7f",
}


def _restock_short(name: str) -> str:
    """``block_goal1`` -> ``block1`` / ``cube_goal2`` -> ``cube2`` (matches plan
    labels)."""
    return name.replace("_goal", "")


def render_scene_from_geometry(
    scene_geometry: SceneGeometry,
    labels: Optional[dict[str, str]] = None,
    image_width: int = 720,
    image_height: int = 560,
):
    """Sim-free top-down scene render from the stored ``SceneGeometry`` -> PIL.Image.

    Reconstructs the §5 inspector picture from the episode's *stored* geometry (the
    "reconstruct, never regenerate" invariant), so the notebook never spins up PyBullet.
    Draws each object's world-frame footprint coloured by family (tall block vs short
    cube -- the F3 height axis, which a top-down view encodes as colour since a cube and a
    block share a footprint), the shelf store region as a dashed box, and the robot base.
    """
    # Local imports: keep matplotlib/PIL optional at module import.
    import io as _io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.patches import Rectangle
    from PIL import Image

    labels = labels or {}
    dpi = 100.0
    fig, ax = plt.subplots(figsize=(image_width / dpi, image_height / dpi), dpi=dpi)

    xs: list[float] = []
    ys: list[float] = []

    # Shelf / free-space containers as the store target region.
    for c in scene_geometry.containers:
        x0, y0, x1, y1 = c.bounds
        xs += [x0, x1]
        ys += [y0, y1]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=True,
                facecolor="#dddddd",
                edgecolor="#555555",
                linestyle="--",
                alpha=0.5,
                zorder=0,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            y1,
            f"{c.kind} (store here)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    for o in scene_geometry.objects:
        px, py, theta = float(o.pose[0]), float(o.pose[1]), float(o.pose[2])
        ct, st = np.cos(theta), np.sin(theta)
        ring = [
            (px + ct * bx - st * by, py + st * bx + ct * by) for bx, by in o.boundary
        ]
        xs += [q[0] for q in ring]
        ys += [q[1] for q in ring]
        color = _FAMILY_COLOR.get(o.family, "#2ca02c")
        # Tall blocks drawn opaque, short cubes lighter -- the height cue on a flat view.
        alpha = 0.9 if o.family == "tall" else (0.55 if o.family == "cube" else 0.7)
        ax.add_patch(
            MplPolygon(
                ring, closed=True, facecolor=color, edgecolor="black", alpha=alpha
            )
        )
        if o.family != "robot":
            ax.text(
                px,
                py,
                labels.get(o.name, _restock_short(o.name)),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold",
            )
    # Robot base marker (family robot may sit far south of the object field).
    for o in scene_geometry.objects:
        if o.family == "robot":
            ax.text(
                float(o.pose[0]),
                float(o.pose[1]),
                "robot",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold",
            )

    if xs and ys:
        mx = 0.08
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - mx, max(ys) + mx)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)  — south → north (store order)")
    ax.set_title("Restock3D initial scene (top-down)")
    fig.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


_FOV = 60.0
_NEAR, _FAR = 0.1, 100.0


def render_scene(sim, image_width: int = 640, image_height: int = 480) -> np.ndarray:
    """Full-scene oblique RGB (H, W, 3) uint8 from the env's own camera."""
    return capture_image(
        sim.physics_client_id,
        image_width=image_width,
        image_height=image_height,
        **sim.config.get_camera_kwargs(),
    )


def _view_proj(sim, image_width: int, image_height: int):
    ck = sim.config.get_camera_kwargs()
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=ck["camera_target"],
        distance=ck["camera_distance"],
        yaw=ck["camera_yaw"],
        pitch=ck["camera_pitch"],
        roll=0,
        upAxisIndex=2,
        physicsClientId=sim.physics_client_id,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=_FOV,
        aspect=float(image_width) / float(image_height),
        nearVal=_NEAR,
        farVal=_FAR,
        physicsClientId=sim.physics_client_id,
    )
    # pybullet returns column-major 16-tuples.
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order="F")
    pr = np.asarray(proj, dtype=np.float64).reshape(4, 4, order="F")
    return v, pr


def project_points(
    sim, points_world, image_width: int = 640, image_height: int = 480
) -> np.ndarray:
    """Map world ``(N, 3)`` points to ``(N, 2)`` pixel coords (x=col, y=row).

    Points behind the camera or off-frame still return coordinates (the caller clamps /
    filters); the z-clip is only used to drop points behind the near plane.
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    v, pr = _view_proj(sim, image_width, image_height)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)  # (N, 4)
    clip = hom @ v.T @ pr.T  # row-vector convention
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / w
    px = (ndc[:, 0] * 0.5 + 0.5) * (image_width - 1)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (image_height - 1)
    return np.stack([px, py], axis=1)


def object_crops(
    sim,
    state,
    names: list[str],
    crop_px: int = 96,
    image_width: int = 640,
    image_height: int = 480,
):
    """``{name: PIL.Image}`` per-object crops from the oblique full-scene render.

    Each crop is a fixed ``crop_px`` window centred on the object's projected pixel, so
    it carries the object at its 3D-projected height (a tall block occupies a taller
    silhouette than a cube) plus local scene context -- the height signal PIGINet's
    image channel needs.
    """
    from PIL import Image  # local: keep PIL optional at import

    rgb = render_scene(sim, image_width, image_height)
    # Skip any name whose pose can't be read as an (x, y, z) position -- notably the mobile
    # robot, whose base uses a different feature set than the cuboids' pose_x/pose_y/pose_z.
    # Without this guard a single un-poseable name in ``names`` raised and the caller (the
    # PIGINet adapter) swallowed it, returning NO crops at all -- an all-zero image channel that
    # silently disabled restock3D PIGINet's height signal. A skipped object simply gets no crop
    # (the consumer zero-fills it), which is correct for the robot.
    centers = {}
    for n in names:
        try:
            centers[n] = project_points(
                sim, [state.get_object_pose(n).position], image_width, image_height
            )[0]
        except (ValueError, KeyError):
            continue
    half = crop_px // 2
    img = Image.fromarray(rgb)
    out = {}
    for n, (cx, cy) in centers.items():
        cx, cy = int(round(cx)), int(round(cy))
        left = min(max(cx - half, 0), image_width - crop_px)
        top = min(max(cy - half, 0), image_height - crop_px)
        out[n] = img.crop((left, top, left + crop_px, top + crop_px))
    return out


def render_labeled_scene(
    sim,
    state,
    names: list[str],
    labels: Optional[dict[str, str]] = None,
    image_width: int = 768,
    image_height: int = 576,
) -> np.ndarray:
    """Full-scene oblique RGB with Set-of-Mark labels at each object's projected centre.

    ``labels`` maps object name -> the mark text (defaults to the canonical name).
    Consumed by the VLMPlan adapter so the VLM can refer to objects by the same names
    the prompt + grounding use.
    """
    from PIL import Image, ImageDraw  # local

    rgb = render_scene(sim, image_width, image_height)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    labels = labels or {n: n for n in names}
    centers = project_points(
        sim,
        [state.get_object_pose(n).position for n in names],
        image_width,
        image_height,
    )
    for n, (cx, cy) in zip(names, centers):
        cx, cy = int(round(cx)), int(round(cy))
        txt = labels.get(n, n)
        pad = 2
        tw = 6 * len(txt) + 2 * pad
        draw.rectangle([cx, cy - 8, cx + tw, cy + 8], fill=(255, 255, 0))
        draw.text((cx + pad, cy - 6), txt, fill=(0, 0, 0))
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=(255, 0, 0), width=2)
    return np.asarray(img)
