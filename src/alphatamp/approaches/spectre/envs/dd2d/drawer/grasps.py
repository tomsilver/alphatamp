"""DD2D grasp model -- top-down parallel-jaw, supporting-line construction (spec Section
5.3).

A grasp of an item is ``g = (alpha, s)``: a direction ``alpha`` from the 18-direction
grid (10 deg steps over [0, 180)) and a slide ``s`` (one of 5 positions). Construction:
rotate the footprint by ``-alpha``; its axis-aligned x-extent gives two vertical
**supporting lines**; ``width`` = the extent, admissible iff 0.5 <= width <= aperture
(12 cm). The **slide intervals** are the y ranges where *both* supporting lines actually
touch material -- the intersection of the left/right **contact runs** (each possibly
disconnected on a concave feature), NOT their hull. If the intersection is empty (e.g. an
L-tool at an angle, or the two prong-ends of a C where the lines touch disjoint features)
the direction is inadmissible. Drawing the slide ``s`` from a real intersection interval
guarantees both **finger rectangles** (2.5 x 2.0 cm effective, P7-P8), which sit flush
against the supporting lines centred at ``s``, close onto material rather than onto a
concavity gap.

Besides the global-envelope grasp (fingers on the outer x-extremes), the model also
enumerates **internal** grasps (:func:`_internal_grasps`): a finger reaches into a
concavity to grip a flat sub-feature -- the dumbbell bar, into the horseshoe opening,
a shoe arm -- wherever the fingers physically fit. These require full-face flat contact,
so a curved boundary yields no internal grasp (a circle keeps only its global grasp).

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
# internal-grasp enumeration (grip a sub-feature -- the dumbbell bar, the horseshoe
# spine -- reaching a finger into a concavity where it fits):
_FULL_FACE_FRAC = 0.9  # a finger inner face must contact >= this fraction of its width
_SCAN_STEP = 0.4  # cm between horizontal scan lines when hunting internal features


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
def _contact_runs_on_line(
    poly: Polygon, x: float, ylo: float, yhi: float
) -> list[tuple[float, float]]:
    """The footprint's *actual* contact runs (y intervals) with the vertical line at
    ``x``.

    Unlike a y-**hull**, this keeps every disconnected run: a concave feature (a
    C-opening / waist) meets a supporting line along two or more separate segments, and
    the gaps between them are exactly where a flat finger would close onto air. A single
    tangent point (a smooth convex boundary, e.g. a circle across its diameter) is kept
    as a degenerate ``(y, y)`` run.
    """
    inter = poly.intersection(LineString([(x, ylo - 1.0), (x, yhi + 1.0)]))
    if inter.is_empty:
        return []
    runs: list[tuple[float, float]] = []
    for part in getattr(inter, "geoms", [inter]):
        if part.is_empty:
            continue
        runs.append((float(part.bounds[1]), float(part.bounds[3])))
    return runs


def _intersect_runs(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Overlap of two run lists: the y where *both* supporting lines touch material.

    Touching (a shared endpoint / tangent point) counts, so a circle's single-point
    contact survives; a genuine disjoint opening does not overlap and drops out. Result
    is merged so coincident runs do not spawn duplicate slides.
    """
    out: list[tuple[float, float]] = []
    for alo, ahi in a:
        for blo, bhi in b:
            lo, hi = max(alo, blo), min(ahi, bhi)
            if hi >= lo - _EPS:  # overlap or touch
                out.append((lo, hi))
    if not out:
        return []
    out.sort()
    merged = [out[0]]
    for lo, hi in out[1:]:
        plo, phi = merged[-1]
        if lo <= phi + _EPS:
            merged[-1] = (plo, max(phi, hi))
        else:
            merged.append((lo, hi))
    return merged


def direction_admissible(
    shape: Shape, alpha: float
) -> tuple[bool, float, float, list[tuple[float, float]]]:
    """Is grasp direction ``alpha`` admissible? Returns (ok, xmin, xmax,
    slide_intervals).

    ``slide_intervals`` are the y ranges where **both** fingers sit on material -- the
    intersection of the two supporting lines' *actual* contact runs (not their hull), so
    a slide drawn from any of them puts both fingers on the footprint rather than closing
    across a concavity. Empty (``ok=False``) when the contact runs do not y-overlap (a
    C-opening / awkward L-angle) or the width is out of aperture.
    """
    rot = rotate(shape.polygon, -alpha, origin=(0, 0), use_radians=True)
    xmin, ymin, xmax, ymax = rot.bounds
    width = xmax - xmin
    if not (MIN_APERTURE <= width <= MAX_APERTURE):
        return (False, xmin, xmax, [])
    left = _contact_runs_on_line(rot, xmin, ymin, ymax)
    right = _contact_runs_on_line(rot, xmax, ymin, ymax)
    if not left or not right:  # pragma: no cover - defensive (extreme x always touches)
        return (False, xmin, xmax, [])
    valid = _intersect_runs(left, right)
    if not valid:  # disjoint contact sets -> both fingers can't touch at one slide
        return (False, xmin, xmax, [])
    return (True, xmin, xmax, valid)


def _slide_positions(
    intervals: list[tuple[float, float]], n: int = N_SLIDES
) -> list[float]:
    """Spread up to ``n`` slides across the valid contact intervals.

    Slides are drawn from the interior 80% of each interval and allocated in proportion
    to interval length (each real interval gets >= 1). When every interval is a point
    (e.g. a circle across a diameter), one slide at that point; point intervals are
    otherwise skipped in favour of the real-contact ones.
    """
    total = sum(max(hi - lo, 0.0) for lo, hi in intervals)
    if total <= 1e-6:  # all point contacts -> a single tangent-point slide
        lo, hi = intervals[0]
        return [(lo + hi) / 2.0]
    out: list[float] = []
    for lo, hi in intervals:
        span = hi - lo
        if span <= 1e-6:
            continue  # skip a stray tangent point when real contact exists
        k = max(1, round(n * span / total))
        a = lo + (1 - _INTERIOR) / 2 * span
        b = hi - (1 - _INTERIOR) / 2 * span
        if k == 1:
            out.append((a + b) / 2.0)
        else:
            out.extend(a + (b - a) * i / (k - 1) for i in range(k))
    return out


# --------------------------------------------------------------------------- #
# internal grasps (grip a sub-feature -- reach a finger into a concavity)
# --------------------------------------------------------------------------- #
def _segments_on_scanline(poly: Polygon, s: float) -> list[tuple[float, float]]:
    """Material x-intervals where the horizontal line at ``y=s`` crosses the footprint."""
    x0, x1 = poly.bounds[0], poly.bounds[2]
    inter = poly.intersection(LineString([(x0 - 1.0, s), (x1 + 1.0, s)]))
    if inter.is_empty:
        return []
    segs: list[tuple[float, float]] = []
    for part in getattr(inter, "geoms", [inter]):
        if part.is_empty:
            continue
        segs.append((float(part.bounds[0]), float(part.bounds[2])))
    return segs


def _face_contact_len(poly: Polygon, x: float, s: float) -> float:
    """How much of a finger inner face (the vertical segment at ``x`` over the band
    around ``s``) lies on the footprint boundary -- i.e. how much flat material it
    meets.

    Near ``FINGER_WIDTH`` means a full-face flat contact; near 0 means a tangent/curved
    touch (which is fine for the global envelope but not for an internal pinch).
    """
    edge = LineString([(x, s - FINGER_WIDTH / 2.0), (x, s + FINGER_WIDTH / 2.0)])
    return float(edge.intersection(poly.boundary).length)


def _internal_grasps(rot: Polygon, alpha: float, n_slide: int) -> list[Grasp]:
    """Antipodal grasps on an *internal* flat feature -- the dumbbell bar, the horseshoe
    spine, a shoe arm -- reaching a finger into the concavity beside/inside the feature.

    Complements (does not replace) the global-envelope grasp. Scans horizontal lines;
    each strictly-internal material segment ``[a, b]`` is a grasp iff (1) both finger
    rects clear the item's own material ("the grippers fit" in the concavity) and (2)
    each finger inner face makes full-face flat contact (>= ``_FULL_FACE_FRAC`` of its
    width on the boundary) -- which also excludes curved-shape sliver pinches, so a
    circle keeps only its global tangent grasp.
    """
    xlo, ylo, xhi, yhi = rot.bounds
    min_face = _FULL_FACE_FRAC * FINGER_WIDTH
    # key by rounded (a, b) to group one flat feature; keep the exact endpoints + slides
    feats: dict[tuple[float, float], tuple[float, float, list[float]]] = {}
    n = max(int((yhi - ylo) / _SCAN_STEP), 1)
    for i in range(n + 1):
        s = ylo + (yhi - ylo) * i / n
        for a, b in _segments_on_scanline(rot, s):
            if not (MIN_APERTURE <= b - a <= MAX_APERTURE):
                continue
            if a <= xlo + _EPS and b >= xhi - _EPS:
                continue  # the global envelope, already emitted by the global path
            g = Grasp(alpha=alpha, s=s, xmin=a, xmax=b)
            lf, rf = _finger_rects_rotframe(g)
            if lf.intersection(rot).area > _EPS or rf.intersection(rot).area > _EPS:
                continue  # a finger would collide with the item -- grippers don't fit
            if _face_contact_len(rot, a, s) < min_face:
                continue
            if _face_contact_len(rot, b, s) < min_face:
                continue
            key = (round(a, 3), round(b, 3))
            if key not in feats:
                feats[key] = (a, b, [])
            feats[key][2].append(s)
    cells: list[Grasp] = []
    for a, b, slides in feats.values():
        # emit only *validated* slides (thin to <= n_slide) -- never interpolate a new
        # one, since a slide between two valid scans may itself hit a block / miss the
        # face (the feature's graspable band need not be contiguous). Emit the *exact*
        # endpoints, not the rounded key, so the fingers stay flush.
        slides.sort()
        if len(slides) <= n_slide:
            picks = slides
        else:
            step = (len(slides) - 1) / (n_slide - 1)
            picks = [slides[round(k * step)] for k in range(n_slide)]
        for s in picks:
            cells.append(Grasp(alpha=alpha, s=s, xmin=a, xmax=b))
    return cells


# --------------------------------------------------------------------------- #
# grasp cells for a shape
# --------------------------------------------------------------------------- #
def grasp_cells(
    shape: Shape, n_dir: int = N_DIRECTIONS, n_slide: int = N_SLIDES
) -> list[Grasp]:
    """All admissible ``(alpha, s)`` grasp cells of ``shape`` (item frame; no collision
    filtering -- clearance is pose-dependent, spec Section 5.3 / M6).

    Two families of cell, both with **both fingers on material** (no air-grasps): the
    **global-envelope** grasp per direction (fingers on the outer x-extremes; slides from
    the true contact-run intersection) plus **internal** grasps that reach a finger into a
    concavity to grip a sub-feature (the dumbbell bar, into the horseshoe opening) where
    the fingers fit.
    """
    cells: list[Grasp] = []
    for i in range(n_dir):
        alpha = math.pi * i / n_dir  # 10 deg steps over [0, 180)
        ok, xmin, xmax, intervals = direction_admissible(shape, alpha)
        if ok:
            for s in _slide_positions(intervals, n_slide):
                cells.append(Grasp(alpha=alpha, s=s, xmin=xmin, xmax=xmax))
        rot = rotate(shape.polygon, -alpha, origin=(0, 0), use_radians=True)
        cells.extend(_internal_grasps(rot, alpha, n_slide))
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


def grasp_blocker(
    g: Grasp, pose: tuple[float, float, float], obstacles: list[Polygon]
) -> int:
    """Index of the **first** obstacle the fingers of ``g`` penetrate at ``pose``, or
    ``-1`` if they clear every obstacle.

    This is :func:`grasp_cfree` with the witness kept instead of discarded: the loop,
    the arithmetic and the short-circuit are identical, so the boolean derived from it
    is bit-for-bit the old answer. The witness is what the v3 refiner instrumentation
    records as a *culprit* -- an observation of a collision check that already ran, not
    a new geometric query (see ``docs/SPECTRE_v3_proposal.md`` Section 6.1).

    Only the first blocker per grasp cell is reported, because that is the one the
    short-circuit actually computed; enumerating *all* blockers would mean doing extra
    intersection work and would no longer be observation-only.
    """
    left, right = finger_rects(g, pose)
    for i, obs in enumerate(obstacles):
        if left.intersection(obs).area > _EPS or right.intersection(obs).area > _EPS:
            return i
    return -1


def grasp_cfree(
    g: Grasp, pose: tuple[float, float, float], obstacles: list[Polygon]
) -> bool:
    """Do the fingers for grasp ``g`` at ``pose`` clear every obstacle footprint?

    Boundary touching is allowed (shared-edge area is 0); penetration is not.
    """
    return grasp_blocker(g, pose, obstacles) < 0


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


def has_grasp_witness(
    shape: Shape,
    pose: tuple[float, float, float],
    obstacles: list[Polygon],
    cells: list[Grasp] | None = None,
) -> tuple[Grasp | None, frozenset[int]]:
    """:func:`has_grasp`, plus the set of obstacle indices that blocked the cells tried.

    Returns ``(grasp, blockers)``. ``grasp`` is *exactly* what :func:`has_grasp` returns
    for the same arguments -- same cell order, same short-circuit -- which
    ``test_dd2d_instrumentation`` pins. ``blockers`` is empty when a grasp was found on
    the first cell, and otherwise holds the first-blocking obstacle of every cell tried
    before the search stopped: the observed reason the item was hard (or impossible) to
    grasp here.
    """
    blockers: set[int] = set()
    for g in cells if cells is not None else grasp_cells(shape):
        b = grasp_blocker(g, pose, obstacles)
        if b < 0:
            return g, frozenset(blockers)
        blockers.add(b)
    return None, frozenset(blockers)
