"""Tests for the arrangement-complete negative packing certificate (dd2d_spec.md §8.4).

The load-bearing property is **soundness / zero false-infeasible**: a subset that truly
packs must NEVER be certified infeasible (a false infeasible contaminates every
downstream label-dependent number). The randomized batteries below hammer that property
in the *tight* regime — buffers only slightly larger than the packed shapes — because
that is where an over-approximation (e.g. the inverted Brunn–Minkowski bound removed
during development) fabricates infeasibilities; a loose-buffer-only battery misses it.
"""

from __future__ import annotations

import math
import random

import pytest
from shapely import MultiPoint, Point, Polygon, box
from shapely.affinity import rotate, translate

from alphatamp.approaches.spectre.envs.dd2d.drawer import certificate as C
from alphatamp.approaches.spectre.envs.dd2d.drawer.shapes import Shape
from alphatamp.approaches.spectre.envs.dd2d.drawer.world import DrawerScene, ItemState


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sq(side: float) -> Polygon:
    h = side / 2.0
    return Polygon([(-h, -h), (h, -h), (h, h), (-h, h)])


_L = Polygon(
    [(-1.5, -1.5), (2.2, -1.5), (2.2, 0.0), (0.0, 0.0), (0.0, 2.2), (-1.5, 2.2)]
)
_L = translate(_L, -_L.centroid.x, -_L.centroid.y)  # concave L, centroid at origin


def _scene(polys: list[Polygon], buffer: Polygon, margin: float) -> DrawerScene:
    """A minimal real ``DrawerScene`` exercising the actual types; the certificate only
    reads ``items[*].shape.{polygon,area,r_max}``, ``buffer`` and ``margin``."""
    items = {}
    for i, p in enumerate(polys):
        p = translate(p, -p.centroid.x, -p.centroid.y)
        concave = not p.equals(p.convex_hull)
        shape = Shape(family="test", polygon=p, concave=concave)
        items[f"o{i}"] = ItemState(
            name=f"o{i}", shape=shape, pose=(0.0, 0.0, 0.0), region="drawer"
        )
    return DrawerScene(
        drawer=buffer,
        wall_band=Polygon(),
        buffer=buffer,
        items=items,
        target="o0",
        margin=margin,
        dims={},
    )


def _names(scene: DrawerScene) -> list[str]:
    return list(scene.items)


# --------------------------------------------------------------------------- #
# geometry primitives
# --------------------------------------------------------------------------- #
def test_nfp_two_squares_is_minus_sum():
    # NFP(a,b)=a⊕(−b); two side-2 squares → [-2,2]^2.
    n = C.nfp([_sq(2)], [_sq(2)])
    assert [round(v, 6) for v in n.bounds] == [-2.0, -2.0, 2.0, 2.0]


@pytest.mark.parametrize("pa,pb", [(_sq(2), _sq(2)), (_L, _sq(1.5)), (_L, _L)])
def test_nfp_inside_iff_overlap(pa, pb):
    """Strictly-inside-NFP ⇔ interiors overlap, over a random translation grid —
    including the concave L (the CDT decomposition must be exact)."""
    npoly = C.nfp(C.convex_parts(pa), C.convex_parts(pb))
    rng = random.Random(0)
    ax0, ay0, ax1, ay1 = pa.bounds
    bx0, by0, bx1, by1 = pb.bounds
    reach = max(ax1 - ax0, ay1 - ay0) + max(bx1 - bx0, by1 - by0)
    for _ in range(2000):
        t = (rng.uniform(-reach, reach), rng.uniform(-reach, reach))
        overlap = pa.intersection(translate(pb, *t)).area > 1e-7
        inside = npoly.buffer(-1e-6).contains(Point(t))
        outside = not npoly.buffer(1e-6).contains(Point(t))
        if inside:
            assert overlap
        if outside:
            assert not overlap


def test_ifp_rectangle():
    # IFP of a side-2 square in a 6x4 buffer is the eroded 4x2 rectangle.
    region = C.ifp(_sq(2), box(0, 0, 6, 4))
    assert [round(v, 6) for v in region.bounds] == [1.0, 1.0, 5.0, 3.0]


def test_convex_parts_exact_cover_concave():
    parts = C.convex_parts(_L)
    assert len(parts) >= 2
    union = parts[0]
    for p in parts[1:]:
        union = union.union(p)
    assert union.symmetric_difference(_L).area < 1e-9
    # every part is convex (equals its own convex hull)
    for p in parts:
        assert p.symmetric_difference(p.convex_hull).area < 1e-9


def test_rotation_grid_dedup_square():
    # a square (4-fold symmetric) collapses when the grid aligns with 90°; either way the
    # grid is a subset of the full Lipschitz grid and never larger.
    full = C.grid_angles(2.0, 1.0)
    dedup = C.rotation_grid(_sq(3.0).buffer(0), 2.0, 1.0)
    assert len(dedup) <= len(full)
    assert dedup[0] == 0.0


# --------------------------------------------------------------------------- #
# H1 area bound (sound)
# --------------------------------------------------------------------------- #
def test_h1_area_bound():
    # two side-5 squares (area 25 each) cannot pack disjointly in a 6x6=36 buffer.
    assert C.area_bound_infeasible([_sq(5), _sq(5)], 36.0) is True
    assert C.area_bound_infeasible([_sq(2), _sq(2)], 36.0) is False


def test_certify_h1_infeasible_instant():
    sc = _scene([_sq(7), _sq(7)], box(0, 0, 6, 6), 1.0)
    assert C.certify_infeasible_by_packing(sc, _names(sc)) is True


# --------------------------------------------------------------------------- #
# arrangement DFS: both directions of the §8.4 lemma (coarse grid so it exhausts fast)
# --------------------------------------------------------------------------- #
def test_infeasible_by_shape_not_area():
    # two side-2 (δ/2-deflated from side-6, δ=4) squares in 3x3: area 8 < 9 (H1 does NOT
    # fire) yet no disjoint packing exists → the DFS must certify infeasible.
    sc = _scene([_sq(6), _sq(6)], box(0, 0, 3.0, 3.0), 4.0)
    assert C.certify_infeasible_by_packing(sc, _names(sc)) is True


def test_feasible_tight_not_certified():
    # the mirror case: two side-2 (deflated) squares DO pack side-by-side in 4.2x2.1;
    # certifying this infeasible would be the catastrophic false-positive (regression:
    # the inverted BM bound did exactly this).
    sc = _scene([_sq(6), _sq(6)], box(0, 0, 4.2, 2.1), 4.0)
    assert C.certify_infeasible_by_packing(sc, _names(sc)) is False


def test_feasible_found_returns_false():
    sc = _scene([_sq(3), _sq(3)], box(0, 0, 12, 8), 1.0)
    assert C.certify_infeasible_by_packing(sc, _names(sc)) is False


def test_thin_deflation_returns_none():
    # a shape that vanishes under δ/2 deflation cannot be reasoned about → None (marginal),
    # never a proof.
    sc = _scene([_sq(1.0), _sq(1.0)], box(0, 0, 6, 6), 1.0)
    assert C.certify_infeasible_by_packing(sc, _names(sc)) is None


def test_budget_timeout_returns_none():
    # a tiny budget forces a timeout on a case the DFS would otherwise decide → None,
    # NEVER a (possibly wrong) infeasible.
    sc = _scene([_sq(6), _sq(6)], box(0, 0, 3.0, 3.0), 4.0)
    assert (
        C.certify_infeasible_by_packing(
            sc, _names(sc), ege_budget=5, time_budget_s=0.001
        )
        is None
    )


# --------------------------------------------------------------------------- #
# zero-false-infeasible batteries (the soundness gate)
# --------------------------------------------------------------------------- #
def _constructed_feasible_scene(rng: random.Random) -> DrawerScene:
    """Build a random valid ≥δ-clearance packing, then wrap it in a *tight* buffer (bbox
    + δ margin) so the certificate must not fabricate an infeasibility."""
    delta = 1.0
    lib = [
        _sq(3.0),
        Polygon([(-2, -1), (2, -1), (2, 1), (-2, 1)]),
        Point(0, 0).buffer(1.6),
        _L,
    ]
    k = rng.randint(2, 4)
    polys, placed = [], []
    cursor = 1.0
    for _ in range(k):
        base = rng.choice(lib)
        base = translate(base, -base.centroid.x, -base.centroid.y)
        th = rng.uniform(0, 2 * math.pi)
        fp0 = rotate(base, th, origin=(0, 0), use_radians=True)
        bx0, by0, bx1, by1 = fp0.bounds
        fp = translate(fp0, cursor - bx0, 1.0 - by0)
        polys.append(base)
        placed.append(fp)
        cursor = (cursor - bx0) + bx1 + delta * 1.05
    x0 = min(p.bounds[0] for p in placed) - delta
    y0 = min(p.bounds[1] for p in placed) - delta
    x1 = max(p.bounds[2] for p in placed) + delta
    y1 = max(p.bounds[3] for p in placed) + delta
    return _scene(polys, box(x0, y0, x1, y1), delta)


def test_zero_false_infeasible_tight_battery():
    """No constructed-feasible tight packing is ever certified infeasible."""
    rng = random.Random(3)
    for _ in range(40):
        sc = _constructed_feasible_scene(rng)
        assert C.certify_infeasible_by_packing(sc, _names(sc)) is not True


@pytest.mark.slow
def test_zero_false_infeasible_tight_battery_large():
    rng = random.Random(17)
    for _ in range(300):
        sc = _constructed_feasible_scene(rng)
        assert C.certify_infeasible_by_packing(sc, _names(sc)) is not True
