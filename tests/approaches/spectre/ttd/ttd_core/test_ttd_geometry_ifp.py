"""Tests for inner-fit polygon construction and semantics (spec §7.1)."""

from __future__ import annotations

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, box

from alphatamp.approaches.spectre.ttd.ttd_core import geometry


def test_ifp_rectangle_closed_form() -> None:
    """IFP of a 2x2 square in a 6x6 tray is the centered [1,5]² box (spec §7.1)."""
    tray = box(0.0, 0.0, 6.0, 6.0)
    a = geometry.to_polygon(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float))
    region = geometry.ifp(a, tray)
    minx, miny, maxx, maxy = region.bounds
    assert (round(minx, 6), round(miny, 6), round(maxx, 6), round(maxy, 6)) == (
        1.0,
        1.0,
        5.0,
        5.0,
    )


def test_ifp_containment_iff_subset_concave() -> None:
    """Reference point in IFP(A, tray) ⇔ the translated concave A ⊆ tray."""
    tray = box(0.0, 0.0, 10.0, 10.0)
    a = geometry.to_polygon(
        np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], float)
    )
    region = geometry.ifp(a, tray)
    band = 0.05
    rng = np.linspace(-2.0, 11.0, 20)
    for tx in rng:
        for ty in rng:
            t = Point(float(tx), float(ty))
            if region.boundary.distance(t) < band:
                continue
            inside = region.contains(t)
            contained = geometry.within_container(translate(a, tx, ty), tray)
            assert inside == contained, (tx, ty, inside, contained)
