"""Restock3D-**v3** strata: fixed block count per stratum + generator acceptance bands + problem-id
encoding.

v3 varies **per-object width + height** (v2 kept them constant), so difficulty is set by block count
``n`` (fixed per stratum, for the constant-object env), the tightness of the feasible-split set
(``rho`` = feasible / all splits), and — on the hard strata — whether the two named greedy hand-rules
are *both* defeated (the "no universal rule" property). The generator (``generator_v3``) samples
freely and filters on these bands; the 400-problem calibration run (Phase 6) re-tunes them.

**Banding.** Unlike ``strata_v2`` (five strata → a v2-local ``SPLIT_BAND // 5`` band + its own
decoder), v3 has **four** strata and deliberately rides the **shared** ``compare.STRATUM_BAND =
SPLIT_BAND // 4`` band, so ``compare.stratum_of`` / ``train._keep`` decode ``restock3d_v3`` correctly
with **no routing edit** (the default 4-stratum path, ``min(3, ...)``, is exact for 0..3). This is
the simpler choice the v2 5th-stratum collision forced us away from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alphatamp.approaches.spectre.compare import SPLIT_BAND, STRATUM_BAND

Split = Literal["train", "val", "test"]

ENV_VARIANT = "restock3d_v3"

#: v3 rides the shared 4-stratum band (no v2-style local sub-band) — see module docstring.
V3_STRATUM_BAND = STRATUM_BAND  # == SPLIT_BAND // 4

#: Banding strata (difficulty index).
STRATA: tuple[int, ...] = (0, 1, 2, 3)

SPLIT_BANDS: dict[str, int] = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class StratumV3:
    """Per-stratum generator parameters (starting values; re-tuned by the calibration
    run)."""

    n: int  # block count (fixed per stratum, for the constant-object env)
    rho_band: tuple[float, float]  # accept iff feasible/all splits in this range
    n_forced: int  # blocks sampled h > short-cutoff (must go to the tall section)
    n_near: (
        int  # blocks sampled straddling the short/tall cutoff (the genuine decisions)
    )
    require_crack: bool  # accept iff BOTH greedy hand-rules pick an infeasible split


#: Starting configs — an increasing gradient of count + tightness. Calibrated to the raw
#: (rho, fill, crack) distributions measured 2026-08-20: fill ~= f(n) (n=6->~0.70, n=9->~0.97), the
#: feasible fraction collapses beyond n=9 (n=10 ~ 1%), and the both-greedy-crack property peaks at
#: n=9 (~0.53). So difficulty rides **n + the rho band + the crack requirement**; fill is a wide
#: sanity guard (below). Re-tuned by the Phase 6 calibration run.
CONFIGS: dict[int, StratumV3] = {
    0: StratumV3(n=6, rho_band=(0.08, 0.55), n_forced=1, n_near=2, require_crack=False),
    1: StratumV3(n=7, rho_band=(0.02, 0.30), n_forced=1, n_near=2, require_crack=False),
    2: StratumV3(n=8, rho_band=(0.005, 0.15), n_forced=2, n_near=2, require_crack=True),
    3: StratumV3(n=9, rho_band=(0.002, 0.06), n_forced=2, n_near=3, require_crack=True),
}

#: Fill-fraction sanity band (loosest feasible packing's used/capacity). Fill is ~determined by
#: ``n``, so this is a wide guard — a floor against a roomy degenerate draw and a ceiling below the
#: exact-fit brittle tail (where the analytic label is likeliest to disagree with the real refiner;
#: Gate G1 re-tunes the ceiling). NOT the primary difficulty knob (rho is).
FILL_BAND: tuple[float, float] = (0.55, 0.995)

#: Per-stratum ``(K_max, r_cap_s)``. Sized (2026-08-20) against the geometry-prior's
#: FP-to-first-feasible distribution so each kept problem comfortably has its feasible skeleton in
#: the pool (K_max ~ 2.5-3x the geom FP mean), and r_cap exceeds the feasible-success time (scaling
#: with block count). The **synthetic** collection uses the analytic refiner (cheap), so a large
#: K_max is affordable; the same budgets carry to the (future) real-refiner eval, where per-candidate
#: cost is ~40 s+. n=6/7/8/9 -> K_max 40/60/150/200, r_cap 50/70/90/110 s.
BUDGETS: dict[int, tuple[int, float]] = {
    0: (40, 50.0),
    1: (60, 70.0),
    2: (150, 90.0),
    3: (200, 110.0),
}

#: Per-split keeper targets — 100/25/25 per stratum (= 400/100/100 across the 4 strata).
SIZES: dict[int, dict[str, int]] = {
    s: {"train": 100, "val": 25, "test": 25} for s in STRATA
}


def params(stratum: int) -> StratumV3:
    if stratum not in CONFIGS:
        raise ValueError(f"unknown v3 stratum {stratum}")
    return CONFIGS[stratum]


def budget(stratum: int) -> tuple[int, float]:
    return BUDGETS[stratum]


def sizes(stratum: int) -> dict[str, int]:
    return SIZES[stratum]


def env_id(stratum: int) -> str:
    """The registered v3 gym id for a banding stratum (recipe key = the stratum itself)."""
    if stratum not in STRATA:
        raise ValueError(f"unknown v3 stratum {stratum}")
    return f"spectre/Restock3Dv3-r{stratum}-v0"


def problem_id(split: str, stratum: int, index: int) -> int:
    """Encode ``(split, banding-stratum, index)`` into a collection-wide problem id."""
    if stratum not in STRATA:
        raise ValueError(f"unknown v3 stratum {stratum}")
    if index >= V3_STRATUM_BAND:
        raise ValueError(
            f"index {index} overflows the stratum band ({V3_STRATUM_BAND})"
        )
    return SPLIT_BANDS[split] * SPLIT_BAND + stratum * V3_STRATUM_BAND + index


def decode(pid: int) -> tuple[str, int, int]:
    """Inverse of :func:`problem_id`: ``(split, banding-stratum, index)``."""
    band, rest = divmod(int(pid), SPLIT_BAND)
    stratum, index = divmod(rest, V3_STRATUM_BAND)
    split = next(s for s, b in SPLIT_BANDS.items() if b == band)
    return split, stratum, index


def stratum_of(pid: int) -> int:
    """Banding stratum (0..3) from a problem id — identical to ``compare.stratum_of`` on
    the shared 4-band, provided for local (generator/gate) use."""
    return (int(pid) % SPLIT_BAND) // V3_STRATUM_BAND
