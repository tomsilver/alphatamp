"""Tests for the §10.0 pre-pilot machinery (sampling, go criteria, run)."""

from __future__ import annotations

import numpy as np
import pytest

from alphatamp.approaches.spectre.ttd.ttd_core import nesting, params
from alphatamp.approaches.spectre.ttd.ttd_eval import pre_pilot as pp


def test_sigma_bands_partition_area_range() -> None:
    """sigma_bands_for splits the achievable ΣA range into 3 contiguous bands."""
    bands = pp.sigma_bands_for(params.OP_A, k=5)
    assert [b.name for b in bands] == ["low", "mid", "high"]
    assert bands[0].lo_area_cm2 == 5 * params.OP_A.member_area_lo_cm2
    assert bands[-1].hi_area_cm2 == 5 * params.OP_A.member_area_hi_cm2
    for a, b in zip(bands, bands[1:]):
        assert abs(a.hi_area_cm2 - b.lo_area_cm2) < 1e-9


def test_sample_k_subsets_respects_band_and_size() -> None:
    """Sampled subsets have size k, ΣA in the band, and are unique (spec §10.0)."""
    areas = [10.0, 20.0, 30.0, 40.0, 50.0]
    band = pp.SigmaBand("mid", 45.0, 75.0)
    subsets = pp.sample_k_subsets(areas, k=2, band=band, n_samples=3, seed=1)
    assert len(subsets) == 3
    assert len(set(subsets)) == 3
    for idx in subsets:
        assert len(idx) == 2
        total = sum(areas[i] for i in idx)
        assert 45.0 <= total <= 75.0


def test_go_criteria_pass_and_fail() -> None:
    """Go criteria: witness ≥ 0.10, decoy ≥ 0.10, band ≤ 0.15 (spec §10.0)."""
    r_i, r_f = 0.9, 1.2
    failing = np.array([1.5, 1.2, 0.5, float("-inf"), 0.95])
    witness, decoy, band, passed = pp.go_criteria(failing, r_i, r_f)
    assert witness == pytest.approx(0.4)
    assert decoy == pytest.approx(0.4)
    assert band == pytest.approx(0.2)  # one marginal → 0.2 > 0.15
    assert not passed
    passing = np.array([1.5, 1.3, 1.25, 1.4, 0.5, 0.4, 0.7, 0.8, 0.6, 0.3])
    _, _, band2, passed2 = pp.go_criteria(passing, r_i, r_f)
    assert band2 == pytest.approx(0.0)
    assert passed2


def test_go_criteria_counts_neg_inf_as_decoy() -> None:
    """An unpackable subset (η = -inf) counts as decoy supply, not witness/band."""
    etas = np.array([float("-inf"), float("-inf"), 2.0, 2.0])
    witness, decoy, band, _ = pp.go_criteria(etas, 0.9, 1.2)
    assert decoy == pytest.approx(0.5)
    assert witness == pytest.approx(0.5)
    assert band == pytest.approx(0.0)


def test_go_criteria_empty_band_cannot_pass() -> None:
    """An empty band (no sampled subset) yields zero supplies and no pass (spec
    §10.0)."""
    witness, decoy, band, passed = pp.go_criteria(np.array([]), 0.9, 1.2)
    assert (witness, decoy, band) == (0.0, 0.0, 0.0)
    assert not passed


def _cell(op_name: str, band_name: str, passed: bool) -> pp.CellResult:
    """Build a synthetic CellResult for serialization/merge tests (no nester)."""
    return pp.CellResult(
        op_name=op_name,
        band_name=band_name,
        c_v_cm=1.2,
        r_i_cm=0.6,
        r_f_cm=0.9,
        n_subsets=10,
        n_packable=5,
        rho_hat=0.7,
        eta_median=0.8,
        eta_iqr=0.2,
        witness_supply=0.2,
        decoy_supply=0.3,
        band_occupancy=0.1,
        timed_out_frac=0.0,
        passed=passed,
    )


def test_report_json_roundtrip() -> None:
    """A pre-pilot report survives JSON serialization and reconstruction."""
    report = pp.PrePilotReport(
        cells=[_cell("OP-A", "low", False), _cell("OP-A", "mid", True)],
        op_counts={"nfp": 10},
        calibrated_cost_s=1.5,
        go=True,
        passing_cell=_cell("OP-A", "mid", True),
    )
    back = pp.report_from_json(pp.report_to_json(report))
    assert len(back.cells) == 2
    assert back.go and back.op_counts == {"nfp": 10}
    assert back.passing_cell is not None and back.passing_cell.band_name == "mid"


def test_merge_reports_combines_shards() -> None:
    """merge_reports concatenates cells, sums op counts, and re-decides GO (spec
    §10.0)."""
    r1 = pp.PrePilotReport(cells=[_cell("OP-A", "low", False)], op_counts={"nfp": 3})
    r2 = pp.PrePilotReport(
        cells=[_cell("OP-B", "mid", True)],
        op_counts={"nfp": 2, "buffer": 5},
        go=True,
        passing_cell=_cell("OP-B", "mid", True),
    )
    merged = pp.merge_reports([r1, r2])
    assert len(merged.cells) == 2
    assert merged.op_counts == {"nfp": 5, "buffer": 5}
    assert merged.go
    assert merged.passing_cell is not None and merged.passing_cell.op_name == "OP-B"


@pytest.mark.slow
def test_sharding_covers_all_cells() -> None:
    """Sharded runs partition the (OP, band) work and their union is the full grid."""
    cfg = nesting.NesterConfig(rot_grid_deg=90.0, n_restarts=1, node_cap=300, seed=0)
    common = {
        "ops": (params.OP_A, params.OP_B),
        "c_vs": (1.2,),
        "k": 2,
        "n_subsets": 1,
        "n_pool": 8,
        "cfg": cfg,
        "r_hi": 0.4,
        "tol": 0.2,
        "seed": 0,
    }
    full = pp.run_pre_pilot(**common)  # type: ignore[arg-type]
    s0 = pp.run_pre_pilot(**common, shard_index=0, n_shards=2)  # type: ignore[arg-type]
    s1 = pp.run_pre_pilot(**common, shard_index=1, n_shards=2)  # type: ignore[arg-type]

    def cell_key(cell: pp.CellResult) -> tuple[str, str]:
        return (cell.op_name, cell.band_name)

    k0 = {cell_key(c) for c in s0.cells}
    k1 = {cell_key(c) for c in s1.cells}
    assert k0.isdisjoint(k1)
    assert {cell_key(c) for c in full.cells} == k0 | k1


@pytest.mark.slow
def test_run_pre_pilot_tiny_end_to_end() -> None:
    """A tiny pre-pilot run produces a well-formed report (spec §10.0).

    Uses k=2 with a small r_hi so every subset packs at both bisection endpoints — η
    returns r_hi immediately (two fast FOUND nests, no expensive exhaustion).
    """
    cfg = nesting.NesterConfig(rot_grid_deg=90.0, n_restarts=1, node_cap=300, seed=0)
    report = pp.run_pre_pilot(
        ops=(params.OP_A,),
        c_vs=(1.2,),
        k=2,
        n_subsets=2,
        n_pool=8,
        cfg=cfg,
        r_hi=0.4,
        tol=0.2,
        seed=0,
    )
    assert len(report.cells) == 3  # 1 op × 3 bands × 1 c_v
    assert isinstance(report.go, bool)
    for cell in report.cells:
        assert 0.0 <= cell.witness_supply <= 1.0
        assert 0.0 <= cell.decoy_supply <= 1.0
        assert 0.0 <= cell.band_occupancy <= 1.0
    assert "Verdict" in pp.format_report(report)
