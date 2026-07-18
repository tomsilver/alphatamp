"""Tests for the procedural shape library (spec §4.2, §8.8)."""

from __future__ import annotations

import numpy as np
import pytest

from alphatamp.approaches.spectre.ttd.ttd_core import geometry, params
from alphatamp.approaches.spectre.ttd.ttd_core import shapes as ttd_shapes


def _assert_shape_invariants(shape: ttd_shapes.Shape, lo: float, hi: float) -> None:
    """Assert a single shape satisfies the §4.2 acceptance invariants."""
    poly = shape.polygon()
    assert geometry.is_valid_simple(poly)
    assert geometry.count_reflex_vertices(shape.vertices) >= 1
    assert lo - 1e-6 <= shape.area_cm2 <= hi + 1e-6
    assert geometry.min_edge_length(shape.vertices) >= 1.0 - 1e-9
    assert geometry.has_admissible_antipodal_pair(shape.vertices)


def test_shape_invariants_small_set() -> None:
    """A small deterministic sample of shapes satisfies the §4.2 invariants."""
    for seed in range(10):
        shape = ttd_shapes.generate_shape_retry(seed, 50.0)
        _assert_shape_invariants(shape, 50.0, 50.0)


def test_seed_determinism() -> None:
    """The same identity seed yields byte-identical vertices (spec §2)."""
    a = ttd_shapes.generate_shape_retry(4242, 45.0)
    b = ttd_shapes.generate_shape_retry(4242, 45.0)
    assert np.array_equal(a.vertices, b.vertices)
    assert a.seed == b.seed == 4242


def test_p5_vs_p5b_area_targeting() -> None:
    """P5 and P5b area bands are respected by the library builder (spec §3)."""
    lib_p5 = ttd_shapes.build_library(n_train=12, n_held_out=0, band="P5")
    for shape in lib_p5.train:
        assert 25.0 - 1e-6 <= shape.area_cm2 <= 80.0 + 1e-6
    lib_p5b = ttd_shapes.build_library(
        n_train=12, n_held_out=0, band="P5b", op=params.OP_A
    )
    for shape in lib_p5b.train:
        assert 28.0 - 1e-6 <= shape.area_cm2 <= 46.0 + 1e-6


def test_train_heldout_seed_disjoint() -> None:
    """Train and held-out shapes have disjoint identity seeds (spec §8.8)."""
    lib = ttd_shapes.build_library(n_train=20, n_held_out=10)
    train_seeds = {s.seed for s in lib.train}
    held_seeds = {s.seed for s in lib.held_out}
    assert train_seeds.isdisjoint(held_seeds)
    assert len(lib.held_out) == 10


def test_json_roundtrip() -> None:
    """Shape and library JSON round-trips preserve vertices and descriptors."""
    shape = ttd_shapes.generate_shape_retry(99, 60.0)
    back = ttd_shapes.shape_from_json(ttd_shapes.shape_to_json(shape))
    assert np.allclose(back.vertices, shape.vertices)
    assert abs(back.area_cm2 - shape.area_cm2) < 1e-6
    assert back.seed == shape.seed
    lib = ttd_shapes.build_library(n_train=4, n_held_out=2)
    lib_back = ttd_shapes.library_from_json(ttd_shapes.library_to_json(lib))
    assert len(lib_back.train) == 4
    assert len(lib_back.held_out) == 2


@pytest.mark.slow
def test_build_full_library_invariants() -> None:
    """The full 500/100 library builds and every shape satisfies the invariants."""
    lib = ttd_shapes.build_library(n_train=500, n_held_out=100, band="P5")
    assert len(lib.train) == 500
    assert len(lib.held_out) == 100
    for shape in lib.train + lib.held_out:
        _assert_shape_invariants(shape, 25.0, 80.0)
