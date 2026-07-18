"""Tests for the nester, N(S,r), η, and the label rule (spec §7, §2.8)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from alphatamp.approaches.spectre.ttd.ttd_core import geometry, nesting, params

# Coarse, single-restart config: squares are rotation-symmetric so a 90° grid suffices,
# and the hand cases are tiny, so exhaustion is fast.
FAST = nesting.NesterConfig(rot_grid_deg=90.0, n_restarts=1, node_cap=5000)


def _square(half: float) -> "geometry.Polygon":
    """An axis-aligned square of half-width ``half`` centered at the origin."""
    return geometry.to_polygon(
        np.array([[-half, -half], [half, -half], [half, half], [-half, half]], float)
    )


def test_nest_found_loose() -> None:
    """Two small squares nest inside a roomy container (spec §7.2)."""
    result = nesting.nest([_square(0.5), _square(0.5)], box(0, 0, 6, 6), 0.0, FAST)
    assert result.status is nesting.NestStatus.FOUND
    assert result.placements is not None
    assert len(result.placements) == 2
    assert nesting.verify_placements(
        [_square(0.5), _square(0.5)], box(0, 0, 6, 6), 0.0, result.placements
    )


def test_nest_infeasible_when_too_tight() -> None:
    """Two 2x2 squares cannot both fit interior-disjoint in a 3x3 tray (spec §7.2)."""
    result = nesting.nest([_square(1.0), _square(1.0)], box(0, 0, 3, 3), 0.0, FAST)
    assert result.status is nesting.NestStatus.INFEASIBLE
    assert result.placements is None


def test_nest_timeout_under_tiny_cap() -> None:
    """A node cap hit before exhaustion yields TIMEOUT, and packs() maps it to False."""
    tiny = nesting.NesterConfig(rot_grid_deg=90.0, n_restarts=1, node_cap=1)
    result = nesting.nest([_square(1.0), _square(1.0)], box(0, 0, 3, 3), 0.0, tiny)
    assert result.status is nesting.NestStatus.TIMEOUT
    assert not nesting.packs([_square(1.0), _square(1.0)], box(0, 0, 3, 3), 0.0, tiny)


def test_packs_non_increasing_in_r() -> None:
    """N(S, r) is non-increasing: packs at small r, fails at large r (spec §2.8)."""
    polys = [_square(1.0)]
    container = box(0, 0, 4, 4)
    assert nesting.packs(polys, container, 0.5, FAST)  # inflated 3x3 fits
    assert not nesting.packs(polys, container, 1.5, FAST)  # inflated 5x5 does not


def test_eta_single_square() -> None:
    """Packing-margin radius of a 2x2 square in a 4x4 tray is ~1.0 (spec §2.8)."""
    eta, timed_out = nesting.packing_margin_radius(
        [_square(1.0)], box(0, 0, 4, 4), FAST, r_hi=2.0, tol=0.02
    )
    assert not timed_out
    assert abs(eta - 1.0) <= 0.05


def test_eta_neg_inf_when_unpackable_at_zero() -> None:
    """Packing margin is -inf when shapes cannot pack even at r = 0 (spec §2.8)."""
    eta, _ = nesting.packing_margin_radius(
        [_square(1.0), _square(1.0)], box(0, 0, 3, 3), FAST, r_hi=1.0
    )
    assert eta == float("-inf")


def test_verify_placements_detects_overlap() -> None:
    """verify_placements rejects a certificate whose shapes overlap."""
    polys = [_square(0.5), _square(0.5)]
    overlapping = [
        nesting.Placement(0, 2.0, 2.0, 0.0),
        nesting.Placement(1, 2.2, 2.0, 0.0),  # overlaps the first
    ]
    assert not nesting.verify_placements(polys, box(0, 0, 6, 6), 0.0, overlapping)


def test_label_feasible_with_certificate() -> None:
    """A roomy single-shape subset is FEASIBLE with a valid nest certificate (§7.3)."""
    polys = [_square(0.5)]
    container = box(0, 0, 6, 6)
    dials = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.5)
    result = nesting.label_candidate(
        polys, container, dials, feasible_cfg=FAST, infeasible_cfg=FAST
    )
    assert result.label is nesting.Label.FEASIBLE
    assert result.certificate is not None
    assert nesting.verify_placements(
        polys, container, dials.r_f_cm, result.certificate.placements or []
    )


def test_label_infeasible() -> None:
    """A subset with no nest even at r_i is INFEASIBLE (spec §7.3)."""
    polys = [_square(1.0), _square(1.0)]
    dials = params.DifficultyDials(c_v_cm=1.2, phi_f_target=0.5)
    result = nesting.label_candidate(
        polys, box(0, 0, 3, 3), dials, feasible_cfg=FAST, infeasible_cfg=FAST
    )
    assert result.label is nesting.Label.INFEASIBLE


def test_label_marginal() -> None:
    """A subset with r_i ≤ η < r_f is MARGINAL and gets dropped (spec §7.3)."""
    polys = [_square(1.0)]  # η ≈ 1.0 in a 4x4 tray
    dials = params.DifficultyDials(c_v_cm=1.8, phi_f_target=0.5)  # r_i=0.9, r_f=1.2
    result = nesting.label_candidate(
        polys, box(0, 0, 4, 4), dials, feasible_cfg=FAST, infeasible_cfg=FAST
    )
    assert result.label is nesting.Label.MARGINAL
