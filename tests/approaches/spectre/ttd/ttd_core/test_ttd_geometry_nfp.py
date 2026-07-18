"""Tests for no-fit polygon construction and semantics (spec §7.1)."""

from __future__ import annotations

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point

from alphatamp.approaches.spectre.ttd.ttd_core import geometry


def _square(half: float) -> "geometry.Polygon":
    """A square of half-width ``half`` centered at the origin."""
    return geometry.to_polygon(
        np.array([[-half, -half], [half, -half], [half, half], [-half, half]], float)
    )


def _l_shape() -> "geometry.Polygon":
    """A concave L-shaped polygon (reference point near the origin)."""
    return geometry.to_polygon(
        np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], float)
    )


def test_nfp_convex_pair_closed_form() -> None:
    """NFP of two side-2 squares is the [-2,2]² square (spec §7.1)."""
    sq = _square(1.0)
    region = geometry.nfp(sq, sq)
    minx, miny, maxx, maxy = region.bounds
    assert (round(minx, 6), round(miny, 6), round(maxx, 6), round(maxy, 6)) == (
        -2.0,
        -2.0,
        2.0,
        2.0,
    )


def test_nfp_interior_iff_overlap_convex() -> None:
    """Reference point strictly inside NFP ⇔ the squares' interiors overlap."""
    sq = _square(1.0)
    region = geometry.nfp(sq, sq)
    assert region.contains(Point(0.0, 0.0))  # coincident → overlap
    assert not region.contains(Point(3.0, 0.0))  # far apart → disjoint
    assert not region.contains(Point(2.0, 0.0))  # touching → on the boundary


def test_nfp_interior_iff_overlap_concave() -> None:
    """On a concave pair, strictly-inside-NFP agrees with actual interior overlap."""
    a = _l_shape()
    b = _square(0.5)
    region = geometry.nfp(a, b)
    band = 0.05  # skip points near the NFP boundary (touching is ambiguous)
    rng = np.linspace(-3.0, 5.0, 25)
    for tx in rng:
        for ty in rng:
            t = Point(float(tx), float(ty))
            if region.boundary.distance(t) < band:
                continue
            inside = region.contains(t)
            overlaps = a.intersection(translate(b, tx, ty)).area > 1e-9
            assert inside == overlaps, (tx, ty, inside, overlaps)
