"""DD2D shape library -- parametric household footprints (spec Section 4).

Items come from parametric families whose dimension ranges are anchored to common
product sizes; each sampled item draws family + dimensions + small shape noise and is
polygonised to a **Shapely** ``Polygon`` in its own frame (centroid at the origin, so a
pose ``(x, y, theta)`` places it by rotate-about-origin then translate). Curved shapes
are polygonised at 24-32 vertices (spec Section 2). The ``dumbbell``/``shoe``/``banana``
families are concave (a waist / L-corner / C-opening) and carry ``concave=True``; every
label-dependent result downstream is stratified on this flag (spec Section 4, Section
11(d)) -- stratified in analysis, never steered in generation.

No engineered puzzle pieces: no equal-area constraint, no forced concavity, no
complementarity engineering. Two library splits (spec Section 4): ``"train"`` and a
``"holdout"`` with dimension ranges shifted +/-15% and one family swapped, for the
generalisation diagnostic (deferred; the ``split`` hook is present).

Units are centimetres throughout (spec Section 2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from shapely import Polygon
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.ops import unary_union

# families whose footprint is non-convex (a waist / L-corner / C-opening feature)
_CONCAVE_FAMILIES = {"dumbbell", "shoe", "banana"}

# sampling weights (boxes/cans slightly upweighted, spec Section 4)
_FAMILY_WEIGHTS = {
    "can": 1.3,
    "bowl": 1.0,
    "box": 1.3,
    "pillcase": 1.0,
    "dumbbell": 1.0,
    "shoe": 1.0,
    "banana": 1.0,
}
FAMILIES = tuple(_FAMILY_WEIGHTS)


@dataclass(frozen=True)
class Shape:
    """A polygonised item footprint in the item frame (centroid at the origin)."""

    family: str
    polygon: Polygon  # centroid at (0, 0)
    concave: bool

    @property
    def size(self) -> tuple[float, float]:
        """Axis-aligned bounding-box (w, h) at the item's canonical orientation."""
        x0, y0, x1, y1 = self.polygon.bounds
        return (x1 - x0, y1 - y0)

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def r_max(self) -> float:
        """Max centroid-to-boundary distance (spec P16 rotation-grid constant)."""
        return max(
            Point(0.0, 0.0).distance(Point(px, py))
            for px, py in self.polygon.exterior.coords
        )


# --------------------------------------------------------------------------- #
# polygonisation primitives
# --------------------------------------------------------------------------- #
def _circle(diameter: float, n: int = 28) -> Polygon:
    r = diameter / 2.0
    return Polygon([(r * math.cos(t), r * math.sin(t)) for t in _angles(n)])


def _capsule(length: float, width: float, n_cap: int = 14) -> Polygon:
    """A rectangle of length x width with semicircular ends (a pillcase)."""
    r = width / 2.0
    half = max(length / 2.0 - r, 0.0)
    pts: list[tuple[float, float]] = []
    # right cap (from -90 to +90 deg), centred at (+half, 0)
    for t in _lin(-math.pi / 2, math.pi / 2, n_cap):
        pts.append((half + r * math.cos(t), r * math.sin(t)))
    # left cap (from +90 to +270 deg), centred at (-half, 0)
    for t in _lin(math.pi / 2, 3 * math.pi / 2, n_cap):
        pts.append((-half + r * math.cos(t), r * math.sin(t)))
    return Polygon(pts)


def _rounded_rect(w: float, h: float, radius: float, n_corner: int = 4) -> Polygon:
    radius = max(0.0, min(radius, 0.49 * min(w, h)))
    if radius <= 1e-6:
        return _rect(w, h)
    ax, ay = w / 2.0 - radius, h / 2.0 - radius
    corners = [
        (ax, ay, 0.0),
        (-ax, ay, math.pi / 2),
        (-ax, -ay, math.pi),
        (ax, -ay, 3 * math.pi / 2),
    ]
    pts: list[tuple[float, float]] = []
    for cx, cy, base in corners:
        for t in _lin(base, base + math.pi / 2, n_corner):
            pts.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
    return Polygon(pts)


def _rect(w: float, h: float) -> Polygon:
    return Polygon([(w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2)])


def _angles(n: int) -> list[float]:
    return [2 * math.pi * i / n for i in range(n)]


def _lin(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _recenter(poly: Polygon) -> Polygon:
    """Translate so the centroid sits at the origin (poses rotate about the
    centroid)."""
    c = poly.centroid
    return translate(poly, -c.x, -c.y)


# --------------------------------------------------------------------------- #
# family builders (each returns a raw, not-yet-recentred polygon)
# --------------------------------------------------------------------------- #
# small overlap (cm) for unioned multi-rect families, so shared edges merge into one polygon
_OVERLAP = 0.4


def _u(rng: random.Random, lo: float, hi: float, shift: float) -> float:
    """Uniform in [lo, hi], with the whole band scaled by ``shift`` (holdout split)."""
    return rng.uniform(lo * shift, hi * shift)


def _build(family: str, rng: random.Random, shift: float) -> Polygon:
    if family == "can":
        return _circle(_u(rng, 4, 8, shift))  # small-medium circle
    if family == "bowl":
        return _circle(
            _u(rng, 8, 12, shift)
        )  # medium-large circle (capped near the 12 cm aperture)
    if family == "box":
        w, h = _u(rng, 5, 20, shift), _u(rng, 4, 12, shift)  # small..large rectangle
        if rng.random() < 0.5:
            return _rect(w, h)  # half the boxes are sharp-cornered
        return _rounded_rect(
            w, h, rng.uniform(0.3, 1.0)
        )  # half get a small corner radius
    if family == "pillcase":
        return _capsule(_u(rng, 10, 18, shift), _u(rng, 2, 4, shift))
    if family == "dumbbell":
        # two identical end blocks joined by a thinner, longer bar (a concave waist). The bar
        # overlaps into each end (_OVERLAP) so the union is always one connected polygon.
        end_w, end_h = _u(rng, 3, 5, shift), _u(rng, 4, 7, shift)
        bar_len, bar_t = _u(rng, 4, 8, shift), _u(rng, 1.5, 2.5, shift)
        off = bar_len / 2.0 + end_w / 2.0
        end = _rect(end_w, end_h)
        bar = _rect(bar_len + 2 * _OVERLAP, bar_t)
        return unary_union([translate(end, -off, 0.0), bar, translate(end, off, 0.0)])
    if family == "shoe":
        # an L of two similarly-sized rectangles (equal arm thickness); the concave inner
        # corner. The horizontal arm overlaps into the vertical one so the union is connected.
        t = _u(rng, 3, 5, shift)
        arm_v, arm_h = _u(rng, 7, 11, shift), _u(rng, 7, 11, shift)
        vert = _rect(t, arm_v)  # vertical arm, base at y=-arm_v/2
        # horizontal arm's short side (its left end) is flush with the vertical arm, at the base
        horiz = translate(
            _rect(arm_h + _OVERLAP, t),
            t / 2.0 + arm_h / 2.0 - _OVERLAP / 2.0,
            -arm_v / 2.0 + t / 2.0,
        )
        return unary_union([vert, horiz])
    if family == "banana":
        # a C-shape built directly as one simple polygon: outer arc out, inner arc back,
        # leaving a gap (the C opening) on the +x side (no annulus hole -> always valid).
        r_out = _u(rng, 5, 7, shift)
        thick = _u(rng, 2, 3, shift)
        r_in = max(r_out - thick, 0.5)
        half = math.radians(rng.uniform(55, 75))  # half-angle of the C opening
        arc = _lin(half, 2 * math.pi - half, 24)  # body spans everything but the +x gap
        pts = [(r_out * math.cos(t), r_out * math.sin(t)) for t in arc]
        pts += [(r_in * math.cos(t), r_in * math.sin(t)) for t in reversed(arc)]
        return Polygon(pts)
    raise ValueError(f"unknown family {family!r}")


# --------------------------------------------------------------------------- #
# public sampler
# --------------------------------------------------------------------------- #
def _family_swap(split: str) -> dict[str, str]:
    """Holdout swaps one family for a shape-shifted stand-in (spec Section 4)."""
    if split == "holdout":
        return {"bowl": "can"}  # swap the large-circle family out on the holdout split
    return {}


def sample_shape(
    rng: random.Random,
    family: str | None = None,
    split: str = "train",
    require_graspable: bool = True,
    max_tries: int = 40,
) -> Shape:
    """Sample one item footprint. ``split='holdout'`` shifts dimension ranges +/-15% and
    swaps one family (the generalisation diagnostic; deferred but wired). The returned
    polygon is valid, non-empty, and recentred on its centroid.

    Per spec Section 4, every sampled shape must admit >= 1 grasp in isolation (some
    direction with width <= aperture and a non-empty contact-overlap interval); shapes
    that do not (e.g. a bowl whose diameter exceeds the 12 cm aperture) are resampled.
    ``grasps`` is imported lazily to avoid a module cycle (``grasps`` imports ``shapes``).
    """
    shift = 1.15 if split == "holdout" else 1.0
    swap = _family_swap(split)
    forced = family
    from .grasps import isolation_graspable  # lazy: break the shapes<->grasps cycle

    for _ in range(max_tries):
        family = forced
        if family is None:
            fams = list(_FAMILY_WEIGHTS)
            weights = [_FAMILY_WEIGHTS[f] for f in fams]
            family = _weighted_choice(fams, weights, rng)
        family = swap.get(family, family)

        poly = _build(family, rng, shift)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = _recenter(poly)
        if poly.is_empty or poly.area <= 0:  # pragma: no cover - defensive
            continue
        shape = Shape(family=family, polygon=poly, concave=family in _CONCAVE_FAMILIES)
        if not require_graspable or isolation_graspable(shape):
            return shape
    raise RuntimeError(
        f"could not sample a graspable {forced or 'random'} shape in {max_tries} tries "
        f"(split={split!r}); the aperture ({12.0} cm) may be too small for this family"
    )


def _weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(weights)
    r = rng.uniform(0, total)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]
