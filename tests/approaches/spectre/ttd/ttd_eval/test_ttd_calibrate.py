"""Tests for the §5.3 op-cost calibration table."""

from __future__ import annotations

from alphatamp.approaches.spectre.ttd.ttd_core.counters import OpCounter
from alphatamp.approaches.spectre.ttd.ttd_eval import calibrate as cal


def test_calibrate_table_has_all_kinds() -> None:
    """Calibrate() times every C-op kind with a positive cost (spec §5.3)."""
    table = cal.calibrate(n_iters=5)
    for kind in ("poly_construct", "buffer", "nfp", "ifp", "union", "minkowski"):
        assert table.us_per_op[kind] > 0.0
    assert table.n_iters == 5
    assert isinstance(table.hardware, str) and table.hardware


def test_cost_us_is_weighted_sum() -> None:
    """cost_us is the dot product of op counts with the µs table (spec §5.3)."""
    table = cal.CalibrationTable(
        us_per_op={"nfp": 1000.0, "buffer": 40.0}, hardware="test", n_iters=1
    )
    counter = OpCounter(c_ops={"nfp": 3, "buffer": 2, "ifp": 5})
    # ifp has no entry → contributes 0.
    assert table.cost_us(counter) == 3 * 1000.0 + 2 * 40.0
