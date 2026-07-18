"""C7 dial-consistency check (TTD spec §2.8, §12.5 P0).

Two conditions gate any parameter set before the pilot:

* **Brunn–Minkowski necessary condition** (holds for any compact shapes): the inflated
  witness footprints cannot even in principle out-area the tray, i.e.
  ``Σ (√Aᵢ + r_f·√π)² ≤ W·H``. v1.0's 18×14 tray at defaults violated this outright.
* **Empirical occupancy design rule**: ``Φ_f(S) = Σ Ã(sᵢ, r_f) / (W·H) ≤ ρ̂ − h_sel``,
  where ρ̂ (the achievable inflated-nest occupancy frontier) is *measured* in the §10.0
  pre-pilot. Until then ρ̂ is supplied by the caller; when absent, only the necessary
  condition is decided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import geometry
from .counters import OpCounter
from .params import DifficultyDials, OperatingPoint
from .shapes import Shape


@dataclass(frozen=True)
class C7Report:
    """Result of :func:`dial_consistency` (spec §2.8).

    ``occupancy_ok`` / ``overall_ok`` are ``None`` when ρ̂ is unavailable (pre-§10.0),
    in which case only the Brunn–Minkowski necessary condition has been evaluated.
    """

    bm_ok: bool
    bm_lhs: float
    bm_rhs: float
    phi_i: float
    phi_f: float
    occupancy_ok: bool | None
    overall_ok: bool | None


def brunn_minkowski_lhs(areas: Sequence[float], r_f: float) -> float:
    """Σ (√Aᵢ + r_f·√π)² — the Brunn–Minkowski lower bound on inflated packed area."""
    root_pi = math.sqrt(math.pi)
    return float(sum((math.sqrt(a) + r_f * root_pi) ** 2 for a in areas))


def brunn_minkowski_ok(areas: Sequence[float], r_f: float, tray_area: float) -> bool:
    """True iff the Brunn–Minkowski necessary condition holds (spec §2.8 C7)."""
    return brunn_minkowski_lhs(areas, r_f) <= tray_area


def phi_occupancy(
    shapes: Sequence[Shape],
    r: float,
    tray_area: float,
    *,
    counter: OpCounter | None = None,
) -> float:
    """Inflated occupancy Φ = Σ Ã(sᵢ, r) / tray_area (spec §2.8), authoritative Ã."""
    total = sum(geometry.inflated_area(s.polygon(), r, counter=counter) for s in shapes)
    return float(total / tray_area)


def phi_f(
    shapes: Sequence[Shape],
    dials: DifficultyDials,
    op: OperatingPoint,
    *,
    counter: OpCounter | None = None,
) -> float:
    """Occupancy at the planting radius r_f (spec §2.8)."""
    return phi_occupancy(shapes, dials.r_f_cm, op.tray_area_cm2, counter=counter)


def phi_i(
    shapes: Sequence[Shape],
    dials: DifficultyDials,
    op: OperatingPoint,
    *,
    counter: OpCounter | None = None,
) -> float:
    """Occupancy at the reachability radius r_i (spec §2.8)."""
    return phi_occupancy(shapes, dials.r_i_cm, op.tray_area_cm2, counter=counter)


def occupancy_rule_ok(
    shapes: Sequence[Shape],
    dials: DifficultyDials,
    op: OperatingPoint,
    rho_hat: float,
    h_sel: float,
    *,
    counter: OpCounter | None = None,
) -> bool:
    """True iff Φ_f ≤ ρ̂ − h_sel (spec §2.8 design rule; ρ̂ measured in §10.0)."""
    return phi_f(shapes, dials, op, counter=counter) <= rho_hat - h_sel


def dial_consistency(
    shapes: Sequence[Shape],
    dials: DifficultyDials,
    op: OperatingPoint,
    *,
    rho_hat: float | None = None,
    h_sel: float = 0.0,
    counter: OpCounter | None = None,
) -> C7Report:
    """Evaluate the full C7 check for a witness subset (spec §2.8, §12.5 P0)."""
    areas = [s.area_cm2 for s in shapes]
    tray_area = op.tray_area_cm2
    bm_lhs = brunn_minkowski_lhs(areas, dials.r_f_cm)
    bm_ok = bm_lhs <= tray_area
    occ_i = phi_i(shapes, dials, op, counter=counter)
    occ_f = phi_f(shapes, dials, op, counter=counter)
    if rho_hat is None:
        occupancy_ok: bool | None = None
        overall_ok: bool | None = None
    else:
        occupancy_ok = occ_f <= rho_hat - h_sel
        overall_ok = bm_ok and occupancy_ok
    return C7Report(
        bm_ok=bm_ok,
        bm_lhs=bm_lhs,
        bm_rhs=tray_area,
        phi_i=occ_i,
        phi_f=occ_f,
        occupancy_ok=occupancy_ok,
        overall_ok=overall_ok,
    )
