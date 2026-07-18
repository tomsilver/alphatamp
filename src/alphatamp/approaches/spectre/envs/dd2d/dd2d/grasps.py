"""DD2D grasp model -- top-down parallel-jaw, supporting-line construction (spec Section
5.3).

A grasp of an item is ``g = (alpha, s)``: a direction ``alpha`` from the 18-direction
grid (10 deg steps over [0, 180)) and a slide ``s`` (one of 5 positions). Construction:
rotate the footprint by ``-alpha``; its axis-aligned x-extent gives two vertical
**supporting lines**; ``width`` = the extent, admissible iff 0.5 <= width <= aperture
(12 cm). The **contact-overlap interval** is the y-interval hull of ``I_L ^ I_R`` where
``I_L``/``I_R`` are the y-projections of the footprint's contact sets with the left/right
supporting lines; if it is empty (e.g. an L-tool at an angle where the two lines touch
disjoint features) the direction is inadmissible. The two **finger rectangles**
(2.5 x 2.0 cm effective, P7-P8) sit flush against the supporting lines, centred at ``s``.

Grasps are defined in the item frame, so the same ``g`` is reusable at any pose -- but
**all collision facts about a grasp are pose-dependent** and certified only by
:func:`grasp_cfree` against the actual scene (spec Section 5.3 / M6). This is a
kinematic-clearance abstraction: supporting-line contact does not model force closure
(spec Section 5.3 / m7).

Units: centimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import box as shp_box
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon

from .shapes import Shape

# effective finger footprint (base + clearance, spec P7-P8)
FINGER_WIDTH = 2.5  # tangential extent (along the supporting line, the y axis here)
FINGER_THICK = 2.0  # normal extent (across the supporting line, the x axis here)
MAX_APERTURE = 12.0  # spec P9
MIN_APERTURE = 0.5
N_DIRECTIONS = 18  # spec P10 (10 deg steps over [0, 180))
N_SLIDES = 5  # spec P11
_INTERIOR = 0.80  # slides drawn from the interior 80% of the contact-overlap interval
_EPS = 1e-9


@dataclass(frozen=True)
class Grasp:
    """A grounded grasp cell (direction + slide) with its supporting-line extents."""

    alpha: float  # grasp direction in the item frame (radians)
    s: float  # slide position (a y value in the -alpha rotated frame)
    xmin: float  # left supporting line (x in the -alpha rotated frame)
    xmax: float  # right supporting line

    @property
    def width(self) -> float:
        return self.xmax - self.xmin


# --------------------------------------------------------------------------- #
# admissibility of a direction (supporting lines + contact-overlap interval)
# --------------------------------------------------------------------------- #
def _y_interval_on_line(
    poly: Polygon, x: float, ylo: float, yhi: float
) -> tuple[float, float] | None:
    """Y-projection of the footprint's contact with the vertical line at ``x``."""
    inter = poly.intersection(LineString([(x, ylo - 1.0), (x, yhi + 1.0)]))
    if inter.is_empty:
        return None
    miny, maxy = inter.bounds[1], inter.bounds[3]
    return (float(miny), float(maxy))


def direction_admissible(
    shape: Shape, alpha: float
) -> tuple[bool, float, float, tuple[float, float] | None]:
    """Is grasp direction ``alpha`` admissible? Returns (ok, xmin, xmax,
    slide_interval).

    ``slide_interval`` is the contact-overlap interval (empty -> disjoint contacts ->
    inadmissible, e.g. an L-tool at an awkward angle).
    """
    rot = rotate(shape.polygon, -alpha, origin=(0, 0), use_radians=True)
    xmin, ymin, xmax, ymax = rot.bounds
    width = xmax - xmin
    if not (MIN_APERTURE <= width <= MAX_APERTURE):
        return (False, xmin, xmax, None)
    il = _y_interval_on_line(rot, xmin, ymin, ymax)
    ir = _y_interval_on_line(rot, xmax, ymin, ymax)
    if (
        il is None or ir is None
    ):  # pragma: no cover - defensive (extreme x always touches)
        return (False, xmin, xmax, None)
    lo, hi = max(il[0], ir[0]), min(il[1], ir[1])
    if lo > hi + _EPS:  # disjoint contact sets -> inadmissible
        return (False, xmin, xmax, None)
    return (True, xmin, xmax, (lo, hi))


def _slide_positions(interval: tuple[float, float], n: int = N_SLIDES) -> list[float]:
    lo, hi = interval
    span = hi - lo
    if span <= 1e-6:  # a point contact (e.g. a circle across a diameter): one slide
        return [(lo + hi) / 2.0]
    a = lo + (1 - _INTERIOR) / 2 * span
    b = hi - (1 - _INTERIOR) / 2 * span
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# --------------------------------------------------------------------------- #
# grasp cells for a shape
# --------------------------------------------------------------------------- #
def grasp_cells(
    shape: Shape, n_dir: int = N_DIRECTIONS, n_slide: int = N_SLIDES
) -> list[Grasp]:
    """All admissible ``(alpha, s)`` grasp cells of ``shape`` (item frame; no collision
    filtering -- clearance is pose-dependent, spec Section 5.3 / M6)."""
    cells: list[Grasp] = []
    for i in range(n_dir):
        alpha = math.pi * i / n_dir  # 10 deg steps over [0, 180)
        ok, xmin, xmax, interval = direction_admissible(shape, alpha)
        if not ok or interval is None:
            continue
        for s in _slide_positions(interval, n_slide):
            cells.append(Grasp(alpha=alpha, s=s, xmin=xmin, xmax=xmax))
    return cells


def isolation_graspable(shape: Shape) -> bool:
    """Does the shape admit >= 1 grasp in isolation (spec Section 4 resample rule)?"""
    for i in range(N_DIRECTIONS):
        alpha = math.pi * i / N_DIRECTIONS
        if direction_admissible(shape, alpha)[0]:
            return True
    return False


# --------------------------------------------------------------------------- #
# pose-dependent finger geometry + clearance
# --------------------------------------------------------------------------- #
def _finger_rects_rotframe(g: Grasp) -> tuple[Polygon, Polygon]:
    hw = FINGER_WIDTH / 2.0
    left = shp_box(g.xmin - FINGER_THICK, g.s - hw, g.xmin, g.s + hw)
    right = shp_box(g.xmax, g.s - hw, g.xmax + FINGER_THICK, g.s + hw)
    return left, right


def finger_rects(g: Grasp, pose: tuple[float, float, float]) -> tuple[Polygon, Polygon]:
    """The two world-frame finger rectangles for grasp ``g`` when the item is at ``pose
    = (x, y, theta)``.

    Rot-frame geometry maps to world by rotating by
    ``alpha + theta`` about the origin, then translating by ``(x, y)``.
    """
    x, y, theta = pose
    left, right = _finger_rects_rotframe(g)
    ang = g.alpha + theta
    left = translate(rotate(left, ang, origin=(0, 0), use_radians=True), x, y)
    right = translate(rotate(right, ang, origin=(0, 0), use_radians=True), x, y)
    return left, right


def grasp_cfree(
    g: Grasp, pose: tuple[float, float, float], obstacles: list[Polygon]
) -> bool:
    """Do the fingers for grasp ``g`` at ``pose`` clear every obstacle footprint?

    Boundary touching is allowed (shared-edge area is 0); penetration is not.
    """
    left, right = finger_rects(g, pose)
    for obs in obstacles:
        if left.intersection(obs).area > _EPS or right.intersection(obs).area > _EPS:
            return False
    return True


def has_grasp(
    shape: Shape,
    pose: tuple[float, float, float],
    obstacles: list[Polygon],
    cells: list[Grasp] | None = None,
) -> Grasp | None:
    """First admissible grasp cell whose fingers clear ``obstacles`` at ``pose`` (or
    ``None`` if the item is ungraspable there -- hemmed in by neighbours / walls)."""
    for g in cells if cells is not None else grasp_cells(shape):
        if grasp_cfree(g, pose, obstacles):
            return g
    return None
