"""Arrangement-complete negative packing certificate for DD2D (``dd2d_spec.md`` §8.4).

Proves **infeasible-by-packing(S)**: that no δ/2-clearance packing of a subset ``S`` of
item shapes into the (rectangular) buffer exists, so a provisional ``marginal(budget)``
label can be upgraded to a hard ``infeasible``. The property that makes every downstream
number trustworthy is **soundness / zero false-infeasible**: a subset that truly packs
must NEVER be certified infeasible. We therefore only ever *weaken* toward "cannot
certify" (→ ``None`` → marginal) when anything is uncertain.

Built from scratch on Shapely — the scrapped ``ttd_core`` is deliberately not used.

Algorithm (spec §8.4), cheap → expensive, each stage a *sound* one-directional test:

1. **Area bound H1** (unconditionally sound; catches "too much stuff", any |S|):
   Σ area(deflate(sᵢ, δ/2)) > area(buffer) ⇒ infeasible. Uses the *exact* deflated
   areas (already computed for the DFS) — the tightest area bound available. (A
   Brunn–Minkowski ``(√A − r√π)²`` term was tried and removed: that expression is an
   *upper* bound on the deflated area, so it fabricates infeasibilities — see
   ``area_bound_infeasible``.)
2. **Arrangement DFS** on the **δ/2-deflated** shapes over the per-shape Lipschitz
   rotation grid Δθ_o = δ/(4·r_max(o)):
   - exact convex decomposition via ``shapely.constrained_delaunay_triangles`` (each
     part ⊆ shape, ⋃ parts == shape — handles the concave families where a plain
     Delaunay fallback would be unsound);
   - exact convex Minkowski sums → exact NFP(a,b)=a⊕(−b) and IFP(b,buffer);
   - candidate positions = **all vertices of the free region** IFP(i) ∖ ⋃ₚ NFP(i,p);
     Shapely's exact set difference materialises the NFP–NFP / NFP–IFP arrangement
     vertices on the free boundary, giving arrangement-completeness at a fixed rotation;
   - **all placement orders** are tried (bounded ``MAX_ORDER_ITEMS``). Completeness:
     bottom-left-compact any packing; the most-bottom-left item is pinned into a buffer
     corner (an IFP vertex), and inductively each item in BL order lands on a
     free-region vertex — so the BL order (∈ all orders) reaches it. Fixed single-order
     search would *not* be sound (the first item's free region is the whole IFP →
     interior positions unreachable), so we never claim ``infeasible`` from a
     single-order search.
   - **budget** P19 (5 s / 1e5 EGEs). Exhausted-without-a-packing ⇒ infeasible; budget
     hit ⇒ ``None`` (marginal, reason=budget), **never** infeasible. Subsets too large
     to enumerate all orders within budget also yield ``None``.

Rotation-grid lemma (§8.4, unit-tested in ``test_certificate.py``): snapping a continuous
δ/2-clearance packing to the grid moves each boundary point ≤ δ/8, leaving pairwise
clearance ≥ δ/4 > 0, so the deflated shapes admit a grid-rotation packing the search
finds; contrapositive: exhaustive failure certifies no continuous packing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Optional

import shapely
from shapely import MultiPoint, MultiPolygon, Polygon
from shapely.affinity import rotate, scale, translate
from shapely.ops import unary_union

from .world import DrawerScene

# Budget P19 (dd2d_spec.md §8.4): 5 s wall-clock or 1e5 checker EGEs per candidate.
DEFAULT_EGE_BUDGET = 100_000
DEFAULT_TIME_BUDGET_S = 5.0
# All-orders search is |S|! — cap the subset size we attempt to fully exhaust; larger
# subsets that no area bound settles fall to marginal(budget). (5! = 120 orders.)
MAX_ORDER_ITEMS = 5
_EPS_AREA = 1e-9


class _BudgetExceeded(Exception):
    """Raised when the EGE / wall-clock budget for one certificate call is hit."""


class _CannotCertify(Exception):
    """Raised when a soundness precondition fails (e.g. a shape's δ/2-deflation is
    degenerate, or its exact convex decomposition cannot be verified). The caller maps
    this to ``None`` (marginal) — we never certify infeasible on shaky geometry."""


@dataclass
class _Budget:
    """EGE + wall-clock accounting for one certificate call (spec §2 / P19)."""

    ege_cap: int
    time_cap_s: float
    eges: int = 0
    t0: float = 0.0

    def start(self) -> "_Budget":
        self.t0 = time.perf_counter()
        return self

    def spend(self, n: int = 1) -> None:
        self.eges += n
        if self.eges > self.ege_cap:
            raise _BudgetExceeded
        # Wall-clock is checked coarsely (every op is cheap; the EGE cap dominates).
        if (self.eges & 0x3FF) == 0 and time.perf_counter() - self.t0 > self.time_cap_s:
            raise _BudgetExceeded


# --------------------------------------------------------------------------- #
# exact geometry primitives (Shapely)
# --------------------------------------------------------------------------- #
def _poly_vertices(poly: Polygon) -> list[tuple[float, float]]:
    """Exterior ring vertices (without the duplicated closing point)."""
    return list(poly.exterior.coords)[:-1]


def _reflect(poly: Polygon) -> Polygon:
    """Reflect through the origin: ``-poly`` (for NFP = a ⊕ (−b))."""
    return scale(poly, xfact=-1.0, yfact=-1.0, origin=(0, 0))


def convex_parts(poly: Polygon) -> list[Polygon]:
    """Exact convex decomposition via constrained Delaunay triangulation.

    Every triangle lies inside ``poly`` and the triangles exactly cover it, so the
    Minkowski-sum-of-parts NFP below is exact (not an over-approximation that could
    shrink the free region and fabricate an infeasibility). Raises ``_CannotCertify``
    if the exact-cover invariant cannot be verified — soundness before sharpness.
    """
    tris = shapely.constrained_delaunay_triangles(poly)
    parts = [
        g
        for g in getattr(tris, "geoms", [])
        if isinstance(g, Polygon) and g.area > _EPS_AREA
    ]
    if not parts:
        raise _CannotCertify("empty convex decomposition")
    covered = unary_union(parts)
    # symmetric-difference area ~ 0 ⇔ the parts tile the polygon exactly.
    if covered.symmetric_difference(poly).area > 1e-6 * max(poly.area, 1.0):
        raise _CannotCertify("convex decomposition does not exactly cover the shape")
    return parts


def _minkowski_convex(a: Polygon, b: Polygon) -> Polygon:
    """Exact Minkowski sum of two convex polygons: hull of pairwise vertex sums."""
    va = _poly_vertices(a)
    vb = _poly_vertices(b)
    pts = [(ax + bx, ay + by) for (ax, ay) in va for (bx, by) in vb]
    return MultiPoint(pts).convex_hull


def nfp(a_parts: list[Polygon], b_parts: list[Polygon]) -> Polygon:
    """No-fit polygon ``a ⊕ (−b)`` from convex decompositions of ``a`` and ``b``.

    ``t ∈ NFP(a,b)`` ⇔ placing ``b``'s reference point (centroid/origin) at ``t`` makes
    ``b`` overlap ``a`` (both at their given, already-rotated orientations, ``a`` at the
    origin). Minkowski distributes over the union of convex parts, so this is exact.
    """
    pieces = [_minkowski_convex(ai, _reflect(bj)) for ai in a_parts for bj in b_parts]
    return unary_union(pieces)


def ifp(b_poly: Polygon, container: Polygon) -> Polygon:
    """Inner-fit polygon: ``{t : place(b, t) ⊆ container}``.

    ``= ∩_{v ∈ verts(b)} (container − v)``; exact for a convex container (the buffer is
    a rectangle). Empty ⇒ ``b`` does not fit in the container at this orientation.
    """
    region: Polygon = container
    for vx, vy in _poly_vertices(b_poly):
        region = region.intersection(translate(container, -vx, -vy))
        if region.is_empty:
            return region
    return region


def grid_angles(r_max: float, delta: float) -> list[float]:
    """Per-shape Lipschitz rotation grid (spec P16): spacing ≤ Δθ = δ/(4·r_max)."""
    dtheta = delta / (4.0 * r_max)
    n = max(1, math.ceil(2.0 * math.pi / dtheta))
    return [2.0 * math.pi * i / n for i in range(n)]


# Congruence tolerance for rotation-grid dedup. Two grid rotations that map the shape to
# an (essentially) identical polygon give identical NFP/IFP and thus a redundant search;
# dropping the redundant one is SOUND. The tolerance is deliberately tiny so that only a
# true rotational symmetry — never a near-symmetry — is collapsed (a spurious collapse
# would drop distinct placements and could fabricate an infeasibility).
_SYMMETRY_TOL = 1e-7


def rotation_grid(defl_poly: Polygon, r_max: float, delta: float) -> list[float]:
    """Lipschitz grid with exactly-congruent rotations deduplicated.

    Detects the shape's rotational-symmetry period on the grid (the first grid step whose
    rotation maps the shape to itself within ``_SYMMETRY_TOL``) and keeps one representative
    per period. Circles collapse to a single angle; an exact rectangle/capsule halves the
    grid; an asymmetric shape (shoe/banana) keeps the full grid. Sound: removed angles are
    congruent copies with identical NFP/IFP.
    """
    angles = grid_angles(r_max, delta)
    if len(angles) <= 1:
        return angles
    base = defl_poly  # angles[0] == 0.0 by construction
    scale_ref = max(r_max, 1.0)
    for i in range(1, len(angles)):
        rot = rotate(defl_poly, angles[i], origin=(0, 0), use_radians=True)
        if base.hausdorff_distance(rot) <= _SYMMETRY_TOL * scale_ref:
            return angles[:i]  # period found: [0, angles[i]) is one fundamental domain
    return angles


def _region_vertices(region) -> list[tuple[float, float]]:
    """All boundary vertices of a (Multi)Polygon free region — the arrangement vertices
    on the free boundary (exterior + holes of every part)."""
    verts: list[tuple[float, float]] = []
    polys = region.geoms if isinstance(region, MultiPolygon) else [region]
    for p in polys:
        if p.is_empty or not isinstance(p, Polygon):
            continue
        verts.extend(list(p.exterior.coords)[:-1])
        for ring in p.interiors:
            verts.extend(list(ring.coords)[:-1])
    return verts


# --------------------------------------------------------------------------- #
# sound area bounds (unconditional, any |S|)
# --------------------------------------------------------------------------- #
def _deflated_areas(defl_polys: list[Polygon]) -> list[float]:
    return [float(p.area) for p in defl_polys]


def area_bound_infeasible(defl_polys: list[Polygon], buffer_area: float) -> bool:
    """H1 (sound): Σ **exact** δ/2-deflated areas > buffer area ⇒ no interior-disjoint
    packing can exist.

    We use the exact deflated areas (already computed for the DFS), which is the tightest
    possible area bound. A Brunn–Minkowski-style ``(√A − r√π)²`` term was tried and
    removed: for a fixed original area the disk *maximises* eroded area (isoperimetric
    inequality), so that expression is an *upper* bound on the deflated area, not a lower
    one — using it overestimates the packed area and fabricates infeasibilities on tight
    buffers (caught by the near-threshold battery in ``test_certificate.py``)."""
    return sum(_deflated_areas(defl_polys)) > buffer_area + _EPS_AREA


# --------------------------------------------------------------------------- #
# arrangement DFS (exact at fixed rotations, all orders, Lipschitz rotation grid)
# --------------------------------------------------------------------------- #
@dataclass
class _Oriented:
    """One grid orientation of an item, with position-independent geometry precomputed."""

    parts_rot: list[Polygon]  # convex parts rotated to this orientation, at the origin
    ifp: Polygon  # IFP(poly_rot, buffer) — where this orientation fits in the buffer


@dataclass
class _Item:
    name: str
    area: float
    orients: list[_Oriented]


def _build_items(
    subset,
    defl_polys: list[Polygon],
    scene: DrawerScene,
    buffer: Polygon,
    delta: float,
    budget: _Budget,
) -> list[_Item]:
    """Precompute, per item and per grid orientation, the rotated convex parts and the
    (position-independent) IFP against the buffer. Doing this once — rather than at every
    DFS node — is what keeps the exhaustive search inside the P19 budget."""
    items: list[_Item] = []
    for name, dp in zip(subset, defl_polys):
        parts = convex_parts(dp)
        r_max = float(scene.items[name].shape.r_max)
        orients: list[_Oriented] = []
        for theta in rotation_grid(dp, r_max, delta):
            parts_rot = [
                rotate(p, theta, origin=(0, 0), use_radians=True) for p in parts
            ]
            poly_rot = rotate(dp, theta, origin=(0, 0), use_radians=True)
            orients.append(_Oriented(parts_rot, ifp(poly_rot, buffer)))
            budget.spend(1)
        items.append(_Item(name, float(dp.area), orients))
    return items


def _pack_exists(items: list[_Item], buffer_area: float, budget: _Budget) -> bool:
    """True iff a grid-rotation packing exists; tries all placement orders (soundness:
    only the bottom-left order is guaranteed to place each item at a free-region vertex,
    and we do not know it a priori). ``nfp_base`` is cached across orders/nodes since it
    depends only on the two orientations, not on positions."""
    n = len(items)
    nfp_cache: dict[tuple[int, int, int, int], Polygon] = {}

    def nfp_base(i: int, oi: int, j: int, oj: int) -> Polygon:
        key = (i, oi, j, oj)
        cached = nfp_cache.get(key)
        if cached is None:
            cached = nfp(items[i].orients[oi].parts_rot, items[j].orients[oj].parts_rot)
            nfp_cache[key] = cached
            budget.spend(1)
        return cached

    def place(order: tuple[int, ...]) -> bool:
        suffix_area = [0.0] * (n + 1)
        for k in range(n - 1, -1, -1):
            suffix_area[k] = suffix_area[k + 1] + items[order[k]].area

        # placed: list of (item_idx, orient_idx, tx, ty)
        def rec(level: int, placed: list, used_area: float) -> bool:
            if level == n:
                return True
            if suffix_area[level] > buffer_area - used_area + _EPS_AREA:
                return False  # sound area prune
            ci = order[level]
            for oi, orient in enumerate(items[ci].orients):
                if orient.ifp.is_empty:
                    continue
                free = orient.ifp
                for pj, poj, ptx, pty in placed:
                    free = free.difference(
                        translate(nfp_base(pj, poj, ci, oi), ptx, pty)
                    )
                    budget.spend(1)
                    if free.is_empty:
                        break
                if free.is_empty:
                    continue
                for tx, ty in _region_vertices(free):
                    if rec(
                        level + 1,
                        placed + [(ci, oi, tx, ty)],
                        used_area + items[ci].area,
                    ):
                        return True
            return False

        return rec(0, [], 0.0)

    order0 = tuple(sorted(range(n), key=lambda i: -items[i].area))
    if place(order0):
        return True
    for order in permutations(range(n)):
        if order != order0 and place(order):
            return True
    return False


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def certify_infeasible_by_packing(
    scene: DrawerScene,
    subset,
    *,
    ege_budget: int = DEFAULT_EGE_BUDGET,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
) -> Optional[bool]:
    """Is ``subset`` provably impossible to pack into ``scene.buffer`` at δ/2 clearance?

    Returns ``True`` (provably infeasible-by-packing — a sound certificate),
    ``False`` (a δ/2-deflated grid-rotation packing was found ⇒ NOT infeasible), or
    ``None`` (undecided within budget / geometry too degenerate ⇒ caller keeps it
    marginal(budget)). ``δ = scene.margin``.
    """
    subset = list(subset)
    if not subset:
        return False
    delta = scene.margin
    half = delta / 2.0
    buffer = scene.buffer
    buffer_area = float(buffer.area)

    # δ/2-deflated shapes (item frame). A degenerate deflation ⇒ cannot certify soundly.
    defl_polys: list[Polygon] = []
    for n in subset:
        d = scene.items[n].shape.polygon.buffer(-half)
        if d.is_empty or not isinstance(d, Polygon) or d.area <= _EPS_AREA:
            return None  # thin shape: don't attempt a proof, stay marginal
        defl_polys.append(d)

    # 1) sound area bound (cheap, any |S|): exact deflated areas vs buffer area.
    if area_bound_infeasible(defl_polys, buffer_area):
        return True

    # 2) arrangement DFS — only attempt full (all-orders) exhaustion for small |S|.
    if len(subset) > MAX_ORDER_ITEMS:
        return None
    budget = _Budget(ege_budget, time_budget_s).start()
    try:
        items = _build_items(subset, defl_polys, scene, buffer, delta, budget)
        found = _pack_exists(items, buffer_area, budget)
        return False if found else True
    except _BudgetExceeded:
        return None
    except _CannotCertify:
        return None
