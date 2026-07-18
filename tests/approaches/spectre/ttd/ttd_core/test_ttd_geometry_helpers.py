"""Tests for geometry helpers: normalization, reflex, edges, decomposition."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.ops import unary_union

from alphatamp.approaches.spectre.ttd.ttd_core import geometry


def test_normalize_ccw_flips_clockwise() -> None:
    """A clockwise vertex list is reordered counter-clockwise."""
    cw = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], float)  # clockwise
    ccw = geometry.normalize_ccw(cw)
    assert geometry.signed_area(ccw) > 0.0


def test_count_reflex_vertices() -> None:
    """A convex polygon has 0 reflex vertices; an L-shape has exactly 1."""
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert geometry.count_reflex_vertices(square) == 0
    l_shape = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], float)
    assert geometry.count_reflex_vertices(l_shape) == 1


def test_min_edge_length() -> None:
    """min_edge_length returns the shortest edge length."""
    verts = np.array([[0, 0], [4, 0], [4, 1], [0, 1]], float)
    assert abs(geometry.min_edge_length(verts) - 1.0) < 1e-9


def test_to_polygon_rejects_self_intersecting() -> None:
    """A self-intersecting (bowtie) vertex list raises GeometryError."""
    bowtie = np.array([[0, 0], [2, 2], [2, 0], [0, 2]], float)
    with pytest.raises(geometry.GeometryError):
        geometry.to_polygon(bowtie)


def test_convex_decompose_covers_concave_polygon() -> None:
    """Star-fan decomposition of a concave polygon unions back to the polygon."""
    l_shape = geometry.to_polygon(
        np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], float)
    )
    parts = geometry.convex_decompose(l_shape)
    assert len(parts) >= 2
    union_area = unary_union(parts).area
    assert abs(union_area - l_shape.area) < 1e-6


def test_region_vertices_deterministic_and_sorted() -> None:
    """region_vertices returns lexicographically sorted, de-duplicated points."""
    sq = geometry.to_polygon(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float))
    v1 = geometry.region_vertices([sq], include_midpoints=False)
    v2 = geometry.region_vertices([sq], include_midpoints=False)
    assert np.array_equal(v1, v2)
    # Lexicographic (column 0 then column 1) non-decreasing.
    assert np.all(np.diff(v1[:, 0]) >= -1e-12)
