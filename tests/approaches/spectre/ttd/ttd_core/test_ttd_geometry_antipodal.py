"""Tests for the antipodal grasp primitive (spec §4.3.1)."""

from __future__ import annotations

import numpy as np

from alphatamp.approaches.spectre.ttd.ttd_core import geometry


def _rectangle() -> geometry.Vertices:
    """A 4x2 axis-aligned rectangle (CCW): edges 0=bottom,1=right,2=top,3=left."""
    return np.array([[0, 0], [4, 0], [4, 2], [0, 2]], float)


def test_known_antipodal_pair() -> None:
    """The bottom/top edge pair of a 4x2 rectangle has d=2 and overlap length 4."""
    pairs = geometry.antipodal_edge_pairs(_rectangle())
    by_edges = {(p.edge_i, p.edge_j): p for p in pairs}
    assert (0, 2) in by_edges  # bottom & top
    bottom_top = by_edges[(0, 2)]
    assert abs(bottom_top.d_cm - 2.0) < 1e-9
    lo, hi = bottom_top.overlap
    assert abs((hi - lo) - 4.0) < 1e-9
    assert (1, 3) in by_edges  # right & left, separation 4
    assert abs(by_edges[(1, 3)].d_cm - 4.0) < 1e-9


def test_no_pair_when_out_of_aperture() -> None:
    """No admissible pair survives when both separations exceed the aperture range."""
    assert not geometry.antipodal_edge_pairs(_rectangle(), d_range=(5.0, 14.0))
    assert not geometry.has_admissible_antipodal_pair(_rectangle(), d_range=(5.0, 14.0))


def test_min_overlap_filters_short_faces() -> None:
    """A min-overlap requirement drops pairs whose projected overlap is too short."""
    # The right/left pair overlaps by 2 on the tangent; require > 3 to drop it.
    pairs = geometry.antipodal_edge_pairs(_rectangle(), min_overlap_cm=3.0)
    edge_sets = {(p.edge_i, p.edge_j) for p in pairs}
    assert (1, 3) not in edge_sets  # overlap length 2 < 3
    assert (0, 2) in edge_sets  # overlap length 4 > 3
