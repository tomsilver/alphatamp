"""Authoritative 2D geometry primitives (TTD spec §2.8, §7.1, §4.3).

All feasibility-relevant geometry is computed here in Shapely (spec §12.1); PyBullet
(a later chunk) only renders. Lengths are cm, areas cm^2 (spec §2).

Conventions pinned here — chunks 2 (nester) and 4 (sampler/refiner) depend on them:

* **Reference point.** A shape is placed by translating its *local origin* (the §4.2
  star center) to a target position. :func:`nfp` and :func:`ifp` are both expressed in
  that frame: pass the moving shape with its reference point at the origin, and the
  returned region is the locus of valid / overlapping reference-point translations.
* **NFP orbit.** ``nfp(a, b)`` keeps ``a`` stationary and orbits ``b``; it equals the
  Minkowski sum ``a ⊕ (−b)``. A translation ``t`` of ``b``'s reference point overlaps
  ``a`` iff ``t`` is strictly inside the NFP (touching on the boundary).
* **Inflated area.** Authoritative ``Ã(s, r) = inflate(s, r).area`` at a fixed
  :data:`~alphatamp.approaches.spectre.ttd.ttd_core.params.BUFFER_QUAD_SEGS`, shared by
  labeling and Phi-occupancy so they never disagree. :func:`inflated_area_approx`
  (``A + P·r + πr²``) is only a sanity approximation and the proposer's ΔΦ estimate.

The nester (``N(S, r)`` / ``η``) and intensified mode are deliberately *not* here — they
are chunk 2 and compose the primitives below.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt
from shapely import affinity, set_precision
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate, unary_union

from .counters import OpCounter
from .params import BUFFER_QUAD_SEGS, GEOM_EPS, SNAP_GRID

Vertices = npt.NDArray[np.float64]
"""(N, 2) float64 array of polygon vertices, CCW, open (no repeated first vertex)."""


class GeometryError(ValueError):
    """Raised when a vertex list does not form a valid simple polygon."""


def _bump(counter: OpCounter | None, kind: str, n: int = 1) -> None:
    if counter is not None:
        counter.bump(kind, n)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- #
# Construction / normalization
# --------------------------------------------------------------------------------- #
def signed_area(verts: Vertices) -> float:
    """Signed polygon area (positive iff CCW) via the shoelace formula."""
    v = np.asarray(verts, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_ccw(verts: Vertices) -> Vertices:
    """Return ``verts`` reordered counter-clockwise (reversed if it was clockwise)."""
    v = np.asarray(verts, dtype=np.float64)
    if signed_area(v) < 0.0:
        return v[::-1].copy()
    return v.copy()


def to_polygon(verts: Vertices, *, counter: OpCounter | None = None) -> Polygon:
    """Build a validated, CCW-normalized :class:`Polygon` from a vertex list.

    Raises :class:`GeometryError` if the polygon is invalid or self-intersecting.
    """
    v = normalize_ccw(verts)
    poly = Polygon(v)
    _bump(counter, "poly_construct")
    if (not poly.is_valid) or (not poly.is_simple) or poly.area <= 0.0:
        raise GeometryError("vertices do not form a valid simple polygon")
    return poly


def to_vertices(poly: Polygon) -> Vertices:
    """Exterior coordinates of ``poly`` as CCW, open (M, 2) float64 vertices."""
    coords = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    return normalize_ccw(coords)


def is_valid_simple(poly: Polygon) -> bool:
    """True iff ``poly`` is a valid, simple, positive-area polygon."""
    return bool(poly.is_valid and poly.is_simple and poly.area > 0.0)


def snap(geom: BaseGeometry, grid: float = SNAP_GRID) -> BaseGeometry:
    """Snap coordinates to ``grid`` (spec §2.8 determinism); wraps set_precision."""
    return set_precision(geom, grid)


# --------------------------------------------------------------------------------- #
# Inflation / Ã (spec §2.8)
# --------------------------------------------------------------------------------- #
def inflate(
    poly: Polygon,
    r: float,
    *,
    quad_segs: int = BUFFER_QUAD_SEGS,
    counter: OpCounter | None = None,
) -> Polygon:
    """Round-join Minkowski inflation by radius ``r`` >= 0 (spec §2.8)."""
    if r < 0.0:
        raise ValueError("inflation radius must be non-negative")
    _bump(counter, "buffer")
    result = poly.buffer(r, quad_segs=quad_segs, join_style="round")
    return result


def inflated_area(
    poly: Polygon,
    r: float,
    *,
    quad_segs: int = BUFFER_QUAD_SEGS,
    counter: OpCounter | None = None,
) -> float:
    """Authoritative Ã(s, r) = area of the round-inflated polygon (spec §2.8)."""
    return float(inflate(poly, r, quad_segs=quad_segs, counter=counter).area)


def inflated_area_approx(area: float, perimeter: float, r: float) -> float:
    """Linear inflated-area approximation A + P·r + πr² (spec §2.8, sanity only)."""
    return area + perimeter * r + np.pi * r * r


# --------------------------------------------------------------------------------- #
# Convex decomposition + Minkowski (spec §7.1)
# --------------------------------------------------------------------------------- #
def _polygon_kernel_point(poly: Polygon) -> Point | None:
    """A point that sees the whole boundary (the polygon's kernel), or None.

    The kernel is the intersection of the interior half-planes of every edge. It is non-
    empty iff the polygon is star-shaped — which the §4.2 family always is (star center
    at the local origin), so this is the primary decomposition path.
    """
    verts = to_vertices(poly)
    minx, miny, maxx, maxy = poly.bounds
    extent = max(maxx - minx, maxy - miny)
    big = 10.0 * extent + 1.0
    region: BaseGeometry = box(minx - big, miny - big, maxx + big, maxy + big)
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        direction = b - a
        left = np.array([-direction[1], direction[0]], dtype=np.float64)
        norm = float(np.hypot(left[0], left[1]))
        if norm < GEOM_EPS:
            continue
        left /= norm
        # Quad covering the interior (left) side of the edge.
        half_plane = Polygon([a, b, b + left * big, a + left * big])
        region = region.intersection(half_plane)
        if region.is_empty or region.area <= GEOM_EPS:
            return None
    pt = region.representative_point()
    return pt if poly.contains(pt) else None


def convex_decompose(
    poly: Polygon,
    *,
    kernel_pt: Point | None = None,
    counter: OpCounter | None = None,
) -> list[Polygon]:
    """Decompose ``poly`` into convex parts whose union equals ``poly`` (spec §7.1).

    Primary path: fan-triangulate from a kernel point (exact for star-shaped polygons,
    the §4.2 family). Fallback for non-star-shaped input (e.g. deferred out-of-family
    real footprints, §4.2.1): filtered Delaunay triangulation, which is approximate and
    should be replaced with a constrained decomposition in that chunk.
    """
    if kernel_pt is None:
        kernel_pt = _polygon_kernel_point(poly)
    if kernel_pt is not None:
        verts = to_vertices(poly)
        k = np.array([kernel_pt.x, kernel_pt.y], dtype=np.float64)
        parts: list[Polygon] = []
        n = len(verts)
        for i in range(n):
            tri = Polygon([k, verts[i], verts[(i + 1) % n]])
            if tri.area > GEOM_EPS:
                parts.append(tri)
        _bump(counter, "poly_construct", len(parts))
        return parts
    # Fallback: Delaunay of the vertices, keep triangles inside the polygon.
    parts = [t for t in triangulate(poly) if poly.contains(t.representative_point())]
    _bump(counter, "poly_construct", len(parts))
    return parts


def reflect(poly: Polygon) -> Polygon:
    """Reflect ``poly`` through the origin (p ↦ −p), for NFP = A ⊕ (−B)."""
    return affinity.scale(poly, xfact=-1.0, yfact=-1.0, origin=(0.0, 0.0))


def minkowski_sum_convex(
    a: Polygon, b: Polygon, *, counter: OpCounter | None = None
) -> Polygon:
    """Minkowski sum of two convex polygons = convex hull of pairwise vertex sums."""
    va = np.asarray(a.exterior.coords, dtype=np.float64)[:-1]
    vb = np.asarray(b.exterior.coords, dtype=np.float64)[:-1]
    sums = (va[:, None, :] + vb[None, :, :]).reshape(-1, 2)
    _bump(counter, "minkowski")
    hull = MultiPoint([tuple(p) for p in sums]).convex_hull
    return hull


# --------------------------------------------------------------------------------- #
# NFP / IFP (spec §7.1)
# --------------------------------------------------------------------------------- #
def nfp(
    a: Polygon,
    b: Polygon,
    *,
    a_parts: Sequence[Polygon] | None = None,
    b_parts: Sequence[Polygon] | None = None,
    counter: OpCounter | None = None,
) -> Polygon | MultiPolygon:
    """No-fit polygon of ``b`` (reference point = origin) relative to stationary ``a``.

    Returns the locus of ``b``'s reference-point translations at which ``a`` and ``b``
    overlap: strictly inside ⇔ interiors overlap; on the boundary ⇔ touching; outside ⇔
    disjoint (spec §7.1). Optional precomputed convex decompositions ``a_parts`` /
    ``b_parts`` let the chunk-2 nester reuse work.
    """
    if a_parts is None:
        a_parts = convex_decompose(a, counter=counter)
    neg_b = reflect(b)
    if b_parts is None:
        neg_parts = convex_decompose(neg_b, counter=counter)
    else:
        neg_parts = [reflect(p) for p in b_parts]
    pieces = [
        minkowski_sum_convex(ai, bj, counter=counter)
        for ai in a_parts
        for bj in neg_parts
    ]
    _bump(counter, "union")
    _bump(counter, "nfp")
    result = unary_union(pieces)
    return snap(result)  # type: ignore[return-value]


def ifp(
    a: Polygon, container: Polygon, *, counter: OpCounter | None = None
) -> Polygon | MultiPolygon:
    """Inner-fit polygon: ref-point translations keeping ``a`` inside ``container``.

    Reference point = origin of ``a``. ``a``'s reference point inside the IFP ⇔ the
    translated ``a`` is contained in ``container`` (spec §7.1). Exact for a convex
    container (the tray is a rectangle, §4.1): erosion = ∩ over ``a``'s vertices of the
    container translated by −vertex.
    """
    va = np.asarray(a.exterior.coords, dtype=np.float64)[:-1]
    region: BaseGeometry = container
    for vx, vy in va:
        shifted = affinity.translate(container, xoff=-float(vx), yoff=-float(vy))
        region = region.intersection(shifted)
        if region.is_empty:
            break
    _bump(counter, "ifp")
    return snap(region)  # type: ignore[return-value]


# --------------------------------------------------------------------------------- #
# Point extraction (serves the chunk-2 nester and the chunk-4 sampler)
# --------------------------------------------------------------------------------- #
def _coords_of(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Flatten any geometry into its coordinate pairs."""
    out: list[tuple[float, float]] = []
    geoms = getattr(geom, "geoms", None)
    if geoms is not None:
        for g in geoms:
            out.extend(_coords_of(g))
        return out
    if isinstance(geom, Polygon):
        out.extend((float(x), float(y)) for x, y in geom.exterior.coords)
        for ring in geom.interiors:
            out.extend((float(x), float(y)) for x, y in ring.coords)
    elif hasattr(geom, "coords"):
        coords = geom.coords  # type: ignore[attr-defined]
        out.extend((float(x), float(y)) for x, y in coords)
    return out


def region_vertices(
    regions: Sequence[BaseGeometry],
    *,
    include_midpoints: bool = True,
    include_intersections: bool = False,
    counter: OpCounter | None = None,
) -> Vertices:
    """Deterministic candidate points on the NFP/IFP arrangement (spec §7.2).

    Boundary vertices of each region, optionally edge midpoints and pairwise boundary-
    intersection points (the intensified-mode arrangement vertices). Snapped,
    lexicographically sorted, and de-duplicated for reproducibility.
    """
    pts: list[tuple[float, float]] = []
    boundaries = [snap(r.boundary) for r in regions]
    for bnd in boundaries:
        coords = _coords_of(bnd)
        pts.extend(coords)
        if include_midpoints:
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                pts.append((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    if include_intersections:
        for i in range(len(boundaries)):
            for j in range(i + 1, len(boundaries)):
                inter = boundaries[i].intersection(boundaries[j])
                if not inter.is_empty:
                    pts.extend(_coords_of(inter))
        _bump(counter, "union")
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(pts, dtype=np.float64)
    snapped = np.round(arr / SNAP_GRID) * SNAP_GRID
    uniq = np.unique(snapped, axis=0)  # np.unique returns lexicographically sorted rows
    return uniq


def sample_boundary(
    region: BaseGeometry, spacing: float, *, backoff: float = 0.0
) -> Vertices:
    """Arc-length samples along ``region``'s boundary, offset outward by ``backoff``.

    Deterministic (ordered by arc length). Used by the chunk-4 compaction sampler to
    draw contact proposals on NFP/IFP boundaries; the ε_s back-off pushes each point
    along the outward normal (out of the overlap region).
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    boundary = region.boundary
    length = float(boundary.length)
    if length <= 0.0:
        return np.empty((0, 2), dtype=np.float64)
    n = max(1, int(length // spacing))
    pts: list[tuple[float, float]] = []
    eps = min(spacing * 0.25, 1e-3)
    for i in range(n):
        d = i * spacing
        p = boundary.interpolate(d)
        if backoff != 0.0:
            p_ahead = boundary.interpolate(min(d + eps, length))
            p_back = boundary.interpolate(max(d - eps, 0.0))
            tangent = np.array(
                [p_ahead.x - p_back.x, p_ahead.y - p_back.y], dtype=np.float64
            )
            tnorm = float(np.hypot(tangent[0], tangent[1]))
            if tnorm >= GEOM_EPS:
                tangent /= tnorm
                normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
                probe = Point(p.x + normal[0] * eps, p.y + normal[1] * eps)
                if region.contains(probe):  # normal points inward → flip to outward
                    normal = -normal
                pts.append((p.x + normal[0] * backoff, p.y + normal[1] * backoff))
                continue
        pts.append((float(p.x), float(p.y)))
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------------- #
# Predicates / descriptors
# --------------------------------------------------------------------------------- #
def interior_disjoint(polys: Sequence[Polygon]) -> bool:
    """True iff no two polygons have overlapping interiors (touching is allowed)."""
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersection(polys[j]).area > GEOM_EPS:
                return False
    return True


def within_container(poly: Polygon, container: Polygon) -> bool:
    """True iff ``poly`` lies within ``container`` (up to numerical tolerance)."""
    return bool(poly.difference(container).area <= GEOM_EPS)


def count_reflex_vertices(verts: Vertices) -> int:
    """Count reflex vertices of a CCW polygon (interior turn, cross product < 0)."""
    v = normalize_ccw(verts)
    n = len(v)
    count = 0
    for i in range(n):
        prev = v[i] - v[(i - 1) % n]
        nxt = v[(i + 1) % n] - v[i]
        cross = float(prev[0] * nxt[1] - prev[1] * nxt[0])
        if cross < -GEOM_EPS:
            count += 1
    return count


def min_edge_length(verts: Vertices) -> float:
    """Length of the shortest edge of the polygon."""
    v = np.asarray(verts, dtype=np.float64)
    edges = np.roll(v, -1, axis=0) - v
    return float(np.min(np.hypot(edges[:, 0], edges[:, 1])))


def convexity_defect(poly: Polygon) -> float:
    """Relative convexity defect: convex_hull.area / poly.area − 1 (>= 0)."""
    return float(poly.convex_hull.area / poly.area - 1.0)


def aspect_ratio(poly: Polygon) -> float:
    """Long/short side ratio of the minimum-area rotated bounding rectangle (>= 1)."""
    rect = poly.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords, dtype=np.float64)[:-1]
    edges = np.roll(coords, -1, axis=0) - coords
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    long_side = float(np.max(lengths))
    short_side = float(np.min(lengths))
    if short_side < GEOM_EPS:
        return float("inf")
    return long_side / short_side


# --------------------------------------------------------------------------------- #
# Antipodal grasp primitive (spec §4.3.1) — shared by the §4.2 shape filter
# --------------------------------------------------------------------------------- #
class AntipodalPair(NamedTuple):
    """An admissible antipodal edge pair (spec §4.3.1).

    ``d_cm`` is the face separation along the mean outward normal; ``overlap`` is the
    projected-overlap interval on the mean tangent (world-origin-referenced), whose
    midpoint is the grasp point.
    """

    edge_i: int
    edge_j: int
    d_cm: float
    overlap: tuple[float, float]


def _edge_outward_normals(verts: Vertices) -> Vertices:
    """Unit outward normals of each CCW edge (edge k: verts[k] → verts[k+1])."""
    v = normalize_ccw(verts)
    d = np.roll(v, -1, axis=0) - v
    # Outward normal for CCW polygon is the right-hand normal (dy, -dx).
    normals = np.stack([d[:, 1], -d[:, 0]], axis=1)
    lengths = np.hypot(normals[:, 0], normals[:, 1])[:, None]
    return normals / lengths


def antipodal_edge_pairs(
    verts: Vertices,
    *,
    tol_deg: float = 10.0,
    d_range: tuple[float, float] = (0.5, 14.0),
    min_overlap_cm: float = 0.0,
) -> list[AntipodalPair]:
    """Admissible antipodal edge pairs of a footprint (spec §4.3.1).

    A pair qualifies iff its outward normals are anti-parallel within ``tol_deg``, its
    face separation lies in ``d_range`` (the P14 aperture limits), and its projected
    overlap on the mean tangent exceeds ``min_overlap_cm`` (default: non-empty overlap,
    per §4.3.1; the grasps chunk tightens this to the finger-column width). Returned
    deterministically, sorted by (edge_i, edge_j) with edge_i < edge_j.
    """
    v = normalize_ccw(verts)
    n = len(v)
    normals = _edge_outward_normals(v)
    cos_tol = np.cos(np.deg2rad(tol_deg))
    lo, hi = d_range
    pairs: list[AntipodalPair] = []
    for i in range(n):
        ai, bi = v[i], v[(i + 1) % n]
        for j in range(i + 1, n):
            if float(np.dot(normals[i], normals[j])) > -cos_tol:
                continue  # not anti-parallel enough
            aj, bj = v[j], v[(j + 1) % n]
            mean_normal = normals[i] - normals[j]
            mnorm = float(np.hypot(mean_normal[0], mean_normal[1]))
            if mnorm < GEOM_EPS:
                continue
            mean_normal /= mnorm
            tangent = np.array([-mean_normal[1], mean_normal[0]], dtype=np.float64)
            # Face separation along the mean normal.
            proj_i = 0.5 * (
                float(np.dot(ai, mean_normal)) + float(np.dot(bi, mean_normal))
            )
            proj_j = 0.5 * (
                float(np.dot(aj, mean_normal)) + float(np.dot(bj, mean_normal))
            )
            d = abs(proj_i - proj_j)
            if d < lo or d > hi:
                continue
            # Overlap of the two edges projected onto the tangent.
            ti = sorted([float(np.dot(ai, tangent)), float(np.dot(bi, tangent))])
            tj = sorted([float(np.dot(aj, tangent)), float(np.dot(bj, tangent))])
            ov_lo = max(ti[0], tj[0])
            ov_hi = min(ti[1], tj[1])
            if (ov_hi - ov_lo) <= max(min_overlap_cm, GEOM_EPS):
                continue
            pairs.append(AntipodalPair(i, j, d, (ov_lo, ov_hi)))
    return pairs


def has_admissible_antipodal_pair(
    verts: Vertices,
    *,
    tol_deg: float = 10.0,
    d_range: tuple[float, float] = (0.5, 14.0),
    min_overlap_cm: float = 0.0,
) -> bool:
    """True iff the footprint has >= 1 admissible antipodal pair (spec §4.2 filter)."""
    return bool(
        antipodal_edge_pairs(
            verts, tol_deg=tol_deg, d_range=d_range, min_overlap_cm=min_overlap_cm
        )
    )
