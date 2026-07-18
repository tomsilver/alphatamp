"""Tests for the C7 dial-consistency check (spec §2.8)."""

from __future__ import annotations

from alphatamp.approaches.spectre.ttd.ttd_core import dials, geometry, params
from alphatamp.approaches.spectre.ttd.ttd_core import shapes as ttd_shapes


def test_c7_necessary_condition_pass_and_fail() -> None:
    """Brunn–Minkowski passes on OP-A but fails on v1.0's 18x14 tray (spec §2.8)."""
    areas = [40.0] * 5
    r_f = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.55).r_f_cm
    lhs = dials.brunn_minkowski_lhs(areas, r_f)
    # v1.0 changelog measured ~328 vs a 252 cm² tray; ours is in that ballpark.
    assert 300.0 < lhs < 340.0
    assert dials.brunn_minkowski_ok(areas, r_f, params.OP_A.tray_area_cm2)  # 468
    assert not dials.brunn_minkowski_ok(areas, r_f, 18.0 * 14.0)  # 252


def test_phi_occupancy_matches_buffer_area() -> None:
    """Phi occupancy equals Σ Ã / tray_area computed from the buffered shapes."""
    shapes = [ttd_shapes.generate_shape_retry(s, 40.0) for s in range(4)]
    dial = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.55)
    tray_area = params.OP_A.tray_area_cm2
    expected = (
        sum(geometry.inflated_area(s.polygon(), dial.r_f_cm) for s in shapes)
        / tray_area
    )
    assert abs(dials.phi_f(shapes, dial, params.OP_A) - expected) < 1e-9


def test_occupancy_rule_uses_rho_hat() -> None:
    """dial_consistency decides occupancy only when ρ̂ is supplied (spec §2.8)."""
    shapes = [ttd_shapes.generate_shape_retry(s, 40.0) for s in range(4)]
    dial = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.55)
    without = dials.dial_consistency(shapes, dial, params.OP_A)
    assert without.occupancy_ok is None
    assert without.overall_ok is None
    assert isinstance(without.bm_ok, bool)
    with_rho = dials.dial_consistency(
        shapes, dial, params.OP_A, rho_hat=0.85, h_sel=0.05
    )
    assert with_rho.occupancy_ok is not None
    assert with_rho.overall_ok is not None
    assert with_rho.occupancy_ok == (with_rho.phi_f <= 0.85 - 0.05)
