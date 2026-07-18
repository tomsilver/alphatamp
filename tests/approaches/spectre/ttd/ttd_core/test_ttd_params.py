"""Tests for TTD params, operating points, and dials (spec §3, §2.8)."""

from __future__ import annotations

from alphatamp.approaches.spectre.ttd.ttd_core import params


def test_mu_decomposition() -> None:
    """MU equals EPS_S + EPS_V + EPS_DISC (spec §2.8)."""
    assert abs((params.EPS_S + params.EPS_V + params.EPS_DISC) - params.MU) < 1e-12


def test_ri_rf_from_cv() -> None:
    """r_i = c_v/2 and r_f = c_v/2 + MU for both clearance dials (spec §2.8)."""
    d12 = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.55)
    d16 = params.DifficultyDials(c_v_cm=1.6, phi_f_target=0.55)
    assert d12.r_i_cm == 0.6
    assert abs(d12.r_f_cm - 0.9) < 1e-12
    assert d16.r_i_cm == 0.8
    assert abs(d16.r_f_cm - 1.1) < 1e-12


def test_operating_point_areas() -> None:
    """OP-A and OP-B raw tray areas match the spec (§3 P2)."""
    assert params.OP_A.tray_area_cm2 == 26.0 * 18.0 == 468.0
    assert params.OP_B.tray_area_cm2 == 28.0 * 20.0 == 560.0


def test_pilot_grid_cardinality() -> None:
    """The §10.2 pilot grid is c_v x Phi_f-band x m_p x k (spec §10.1)."""
    bands = [0.50, 0.55, 0.60]
    cells = list(params.iter_pilot_dials(bands, ks=(4, 5)))
    assert len(cells) == 2 * len(bands) * 3 * 2


def test_provisional_fields_marked() -> None:
    """Provisional (†) fields are surfaced on OperatingPoint and DifficultyDials."""
    assert set(params.provisional_fields(params.OP_A)) == {
        "member_area_lo_cm2",
        "member_area_hi_cm2",
    }
    dials = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.55)
    assert params.provisional_fields(dials) == ["phi_f_target"]


def test_finger_column_extents() -> None:
    """Finger-column extents follow w_f + 2c_f and t_f + 2c_f (spec §4.3.2)."""
    tp = params.default_params()
    assert abs(tp.finger_col_tangential_cm - 2.1) < 1e-12
    assert abs(tp.finger_col_normal_cm - 1.6) < 1e-12
