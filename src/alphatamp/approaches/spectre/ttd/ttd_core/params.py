"""Parameters, operating points, and difficulty dials (TTD spec §3, §2.8).

This module is deliberately geometry-free (no shapely import) so that ``dials.py`` and
``geometry.py`` can import it without a cycle. All lengths are in centimeters and all
areas in cm^2 (spec §2).

The §2.8 inflation semantics are encoded here as the module constants ``MU`` /
``EPS_*`` and the derived ``DifficultyDials.r_i_cm`` / ``r_f_cm`` thresholds. Values
marked provisional (†) in the spec — the operating-point tray sizes / member-area bands
(P2/P5b) and the Phi_f occupancy target (P8) — are frozen only pending the §10.0
pre-pilot; :func:`provisional_fields` surfaces them programmatically.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Iterator, Sequence

# --- §2.8 robustness band (fixed) -------------------------------------------------
EPS_S: float = 0.15
"""Sampler contact back-off (spec §2.8, §5.4)."""

EPS_V: float = 0.05
"""Numerical validity guard (spec §2.8, §5.2 tray-pose-valid)."""

EPS_DISC: float = 0.10
"""Nester rotation-grid discretization slack (spec §2.8, §7.2)."""

MU: float = 0.30
"""Robustness band = EPS_S + EPS_V + EPS_DISC (spec §2.8, fixed)."""

assert abs((EPS_S + EPS_V + EPS_DISC) - MU) < 1e-12, "MU must equal its decomposition"

# --- geometry numerics (shared across ttd_core so labels/Phi never disagree) -------
BUFFER_QUAD_SEGS: int = 16
"""Round-join buffer resolution; shared by inflate() and Phi so Ã is consistent."""

GEOM_EPS: float = 1e-9
"""Boundary-classification tolerance in cm (touching vs overlapping)."""

SNAP_GRID: float = 1e-6
"""shapely.set_precision grid in cm for deterministic union/NFP vertex ordering."""

# --- library seed layout ----------------------------------------------------------
HELD_OUT_OFFSET: int = 10_000_000
"""Held-out shape seeds live in a disjoint block base_seed + this offset (spec §8.8)."""


@dataclass(frozen=True)
class OperatingPoint:
    """A tray size + candidate-member area band (spec §3 P2/P5b).

    ``tray_*`` and ``member_area_*`` are provisional (†) pending the §10.0 pre-pilot.
    """

    name: str
    tray_w_cm: float
    tray_h_cm: float
    member_area_lo_cm2: float = field(metadata={"provisional": True})
    member_area_hi_cm2: float = field(metadata={"provisional": True})

    @property
    def tray_area_cm2(self) -> float:
        """Raw tray interior area (no eroded region exists in v1.3; spec §2.8/§4.1)."""
        return self.tray_w_cm * self.tray_h_cm


OP_A = OperatingPoint(
    "OP-A",
    tray_w_cm=26.0,
    tray_h_cm=18.0,
    member_area_lo_cm2=28.0,
    member_area_hi_cm2=46.0,
)
"""Operating point A (spec §3 P2/P5b, provisional †)."""

OP_B = OperatingPoint(
    "OP-B",
    tray_w_cm=28.0,
    tray_h_cm=20.0,
    member_area_lo_cm2=32.0,
    member_area_hi_cm2=50.0,
)
"""Operating point B (spec §3 P2/P5b, provisional †)."""


@dataclass(frozen=True)
class DifficultyDials:
    """The §10.0/§10.2 pilot grid axes (spec §3 P7/P8/P9/P20).

    ``c_v_cm`` (placement clearance), ``m_p`` (sampler strength), and ``k`` (witness
    subset size) take fixed pilot-grid values; ``phi_f_target`` is provisional (†),
    set from the §10.0 frontier ρ̂.
    """

    c_v_cm: float
    phi_f_target: float = field(metadata={"provisional": True})
    m_p: int = 15
    k: int = 5

    @property
    def r_i_cm(self) -> float:
        """Reachability radius r_i = c_v/2 — the inflation the refiner can realize."""
        return self.c_v_cm / 2.0

    @property
    def r_f_cm(self) -> float:
        """Planting radius r_f = c_v/2 + MU — the feasible-label inflation."""
        return self.c_v_cm / 2.0 + MU


@dataclass(frozen=True)
class TTDParams:
    """Full core-tier parameter set (spec §3 P1–P27).

    Bundles a chosen operating point and difficulty dials with the fixed P-table
    values. Difficulty dials (P7/P8/P9/P20) live on :class:`DifficultyDials`; the
    provisional operating-point values live on :class:`OperatingPoint`.
    """

    operating_point: OperatingPoint
    dials: DifficultyDials
    # P1 — tote interior
    tote_w_cm: float = 40.0
    tote_h_cm: float = 30.0
    tote_wall_thick_cm: float = 1.5
    tote_wall_h_cm: float = 12.0
    # P2 — tray lip
    tray_lip_thick_cm: float = 1.0
    tray_lip_h_cm: float = 2.0
    # P3 — tote/tray gap
    tote_tray_gap_cm: float = 6.0
    # P4 — object height
    object_height_cm: float = 6.0
    # P5 — library footprint area band
    lib_area_lo_cm2: float = 25.0
    lib_area_hi_cm2: float = 80.0
    # P6 — objects per instance N
    n_objects_range: tuple[int, int] = (12, 14)
    # P10 — scramble / tote-scene account
    scramble_radius_cm: float = 0.5
    tote_min_sep_cm: float = 1.2
    scramble_rot_deg: float = 4.0
    # P11–P13 — fingers
    finger_width_cm: float = 1.5
    finger_thick_cm: float = 1.0
    finger_clear_cm: float = 0.3
    # P14 — gripper aperture (also the antipodal face-separation range)
    aperture_range_cm: tuple[float, float] = (0.5, 14.0)
    # P15/P16 — descent / carry heights
    grasp_descent_z_cm: float = 3.0
    carry_z_cm: float = 15.0
    # P17 — antipodal tolerance
    antipodal_tol_deg: float = 10.0
    # P18 — refinement budget (control-flow unit only; spec §5.3)
    refine_budget_calls: int = 300
    # P19 — stage caps (t_g grasp draws, t_p pose draws/grasp, rho revision tokens)
    stage_caps: tuple[int, int, int] = (3, 5, 2)
    # P21 — nester rotation grid
    nester_rot_grid_deg: float = 5.0
    # P22 — candidates per instance K
    candidates_range: tuple[int, int] = (28, 36)
    # P23 — feasible candidates F
    feasible_range: tuple[int, int] = (3, 5)
    # P24 — minimum decoys
    min_decoys: int = 3
    # P26 — MI leakage threshold (bits)
    mi_tau_bits: float = 0.10
    # P27 — witnesses per instance W
    witnesses_range: tuple[int, int] = (3, 5)

    @property
    def finger_col_tangential_cm(self) -> float:
        """Finger-column tangential extent w_f + 2*c_f (spec §4.3.2)."""
        return self.finger_width_cm + 2.0 * self.finger_clear_cm

    @property
    def finger_col_normal_cm(self) -> float:
        """Finger-column normal extent t_f + 2*c_f (spec §4.3.2)."""
        return self.finger_thick_cm + 2.0 * self.finger_clear_cm


def provisional_fields(obj: object) -> list[str]:
    """Return names of dataclass fields marked provisional (†) on ``obj`` (spec §3)."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("provisional_fields expects a dataclass instance")
    return [
        f.name for f in dataclasses.fields(obj) if f.metadata.get("provisional", False)
    ]


def default_params(
    op: OperatingPoint = OP_A,
    *,
    c_v_cm: float = 1.2,
    phi_f_target: float = 0.55,
    m_p: int = 15,
    k: int = 5,
) -> TTDParams:
    """Build a :class:`TTDParams` at the core-tier defaults (spec §3)."""
    dials = DifficultyDials(c_v_cm=c_v_cm, phi_f_target=phi_f_target, m_p=m_p, k=k)
    return TTDParams(operating_point=op, dials=dials)


def iter_pilot_dials(
    phi_f_bands: Sequence[float],
    ks: Sequence[int] = (4, 5),
    *,
    c_vs: Sequence[float] = (1.2, 1.6),
    m_ps: Sequence[int] = (5, 15, 40),
) -> Iterator[DifficultyDials]:
    """Iterate the §10.2 pilot grid: c_v x Phi_f band x m_p x k (spec §10.1).

    The three ``phi_f_bands`` come from the §10.0 pre-pilot. With the spec defaults
    this yields ``len(c_vs) * len(phi_f_bands) * len(m_ps) * len(ks)`` cells.
    """
    for c_v in c_vs:
        for phi_f in phi_f_bands:
            for m_p in m_ps:
                for k in ks:
                    yield DifficultyDials(c_v_cm=c_v, phi_f_target=phi_f, m_p=m_p, k=k)
