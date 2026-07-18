"""Tests for the op-level cost counters (spec §5.3)."""

from __future__ import annotations

import numpy as np

from alphatamp.approaches.spectre.ttd.ttd_core import geometry
from alphatamp.approaches.spectre.ttd.ttd_core.counters import OpCounter


def test_counter_bumped_by_geometry_ops() -> None:
    """Inflate, construction, and NFP bump the expected C-op counters (spec §5.3)."""
    counter = OpCounter()
    sq = geometry.to_polygon(
        np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float), counter=counter
    )
    geometry.inflate(sq, 0.5, counter=counter)
    geometry.nfp(sq, sq, counter=counter)
    assert counter.c_ops.get("poly_construct", 0) >= 1
    assert counter.c_ops.get("buffer", 0) == 1
    assert counter.c_ops.get("nfp", 0) == 1
    assert counter.c_ops.get("minkowski", 0) >= 1
    assert counter.total_c() == sum(counter.c_ops.values())
    assert counter.p_ops == 0  # no prepared-predicate evals until the streams chunk


def test_counter_merge() -> None:
    """Merging folds another counter's C-ops and P-ops in place."""
    a = OpCounter(p_ops=2, c_ops={"buffer": 3})
    b = OpCounter(p_ops=1, c_ops={"buffer": 1, "nfp": 4})
    a.merge(b)
    assert a.p_ops == 3
    assert a.c_ops == {"buffer": 4, "nfp": 4}
