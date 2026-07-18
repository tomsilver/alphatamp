"""Tests for inflation and the Ã inflated-area helper (spec §2.8)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from alphatamp.approaches.spectre.ttd.ttd_core import geometry
from alphatamp.approaches.spectre.ttd.ttd_core import shapes as ttd_shapes


def test_inflate_area_monotone_in_r() -> None:
    """Inflated area is non-decreasing in the inflation radius (spec §2.8)."""
    poly = geometry.to_polygon(np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float))
    areas = [geometry.inflated_area(poly, r) for r in (0.0, 0.5, 1.0, 2.0)]
    assert all(a <= b for a, b in zip(areas, areas[1:]))


def test_inflated_area_approx_sanity() -> None:
    """Ã ≈ A + P·r + πr² within ~1% for a generated shape (spec §2.8, α≈0.98-0.99)."""
    shape = ttd_shapes.generate_shape_retry(7, 50.0)
    poly = shape.polygon()
    r = 0.5
    exact = geometry.inflated_area(poly, r)
    approx = geometry.inflated_area_approx(poly.area, poly.length, r)
    assert abs(exact - approx) / exact < 0.01


def test_inflation_direction_is_inflate_objects() -> None:
    """R-inflated shapes disjoint-in-tray ⇒ originals ≥2r apart and ≥r from walls.

    Verifies the §7.3 inflation direction: inflating the objects (never dilating the
    container) encodes 2r pairwise clearance and r wall clearance in one radius.
    """
    r = 1.0
    tray = box(0.0, 0.0, 20.0, 20.0)
    # Two originals 2r apart in y; a third original exactly r from the left wall.
    orig1 = box(3.0, 3.0, 5.0, 5.0)
    orig2 = box(3.0, 7.0, 5.0, 9.0)  # y gap 7-5 = 2 = 2r
    orig3 = box(1.0, 12.0, 3.0, 14.0)  # left edge x=1 = r from the x=0 wall
    inflated = [geometry.inflate(p, r) for p in (orig1, orig2, orig3)]
    # Premise: the inflated shapes are interior-disjoint and inside the tray.
    assert geometry.interior_disjoint(inflated)
    for inf in inflated:
        assert geometry.within_container(inf, tray)
    # Conclusion: originals are ≥2r apart and ≥r from every wall.
    assert orig1.distance(orig2) >= 2.0 * r - 1e-9
    for orig in (orig1, orig2, orig3):
        assert tray.boundary.distance(orig) >= r - 1e-9
