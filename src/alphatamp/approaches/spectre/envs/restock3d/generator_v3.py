"""Per-seed problem generator for **Restock3D v3** (per-object widths + heights near the cutoff).

Replaces v2's fill-until-full XY loop (which correlated width with block count and never measured
hardness). Since split feasibility is now pure arithmetic (``feasibility_v3``), v3 samples freely and
**filters** on exactly the properties that make block *selection* matter:

1. sample per-block **heights** in bands (a few *forced* > short-cutoff, a few *near-threshold*
   straddling it, the rest short-eligible) and **widths** ~U[0.02, 0.08];
2. enumerate every split; **accept** iff there is >= 1 feasible split, the loosest feasible packing's
   fill is in ``FILL_BAND``, ``rho`` = feasible/all splits is in the stratum band, and (hard strata)
   both named greedy hand-rules pick an *infeasible* split (the no-universal-rule property);
3. sample floor XY with v2's region-rejection sampler and **shuffle** name↔spot so an object's role
   is not recoverable from its spawn order.

``build_spec_v3(seed, stratum)`` is a deterministic pure function of ``(seed, stratum)`` (LRU-cached),
so the env, the analytic classifier, and the PIGINet scene reconstruction all agree. The env consumes
it through ``stratum_env_args_v3`` (fed to :class:`kinematic_env.ObjectCentricRestock3DEnvV3`, which
rebuilds movable bodies per seed).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import feasibility_v3 as F
from . import strata_v3
from .generator import (
    _EXCLUSION_RADIUS,
    _MAX_RESEED,
    _OBJECT_REGION_X,
    _OBJECT_REGION_Y,
    _SAMPLE_BUDGET,
    _Rng,
    _rng_shuffle,
    _sample_positions,
)
from .kinematic_env import ObjectSpec, PoseFn, Restock3DEnvConfig
from .section_geometry import compute_section_infos

#: v3 reseed cap — larger than v2's since the acceptance bands are stricter (measured by Phase 6).
_MAX_RESEED_V3 = 600
#: Fixed object y-depth half-extent (v3 varies x-width + z-height only; depth stays constant).
_DEPTH_HALF = 0.025


@dataclass(frozen=True)
class RestockSpecV3:
    """One generated v3 problem: per-object widths, heights, and floor spots (names parallel)."""

    stratum: int
    n: int
    names: tuple[str, ...]
    widths: tuple[float, ...]  # full x-widths
    heights: tuple[float, ...]  # full z-heights
    floor: tuple[tuple[float, float], ...]  # (x, y) floor spot per object

    def blocks(self) -> list[F.Block]:
        return [
            F.Block(
                self.names[i],
                self.widths[i],
                self.heights[i],
                self.floor[i][0],
                self.floor[i][1],
            )
            for i in range(self.n)
        ]


def _u(rng: _Rng, lo: float, hi: float) -> float:
    return round(lo + rng.uniform() * (hi - lo), 4)


def _sample_heights(rng: _Rng, n: int, n_forced: int, n_near: int) -> list[float]:
    """Heights in role bands: forced (> short cutoff, tall-only), near-threshold (straddle the
    cutoff), free (short-eligible). Shuffled so the role order is not the sampling order.
    """
    n_forced = min(n_forced, n)
    n_near = min(n_near, n - n_forced)
    n_free = n - n_forced - n_near
    hs: list[float] = []
    for _ in range(n_forced):
        hs.append(_u(rng, F.SHORT_CUTOFF + 0.001, F.TALL_CUTOFF))  # must go tall
    for _ in range(n_near):
        hs.append(_u(rng, 0.09, 0.15))  # straddles the 0.12 short cutoff
    for _ in range(n_free):
        hs.append(_u(rng, 0.05, F.SHORT_CUTOFF))  # short-eligible
    _rng_shuffle(hs, rng)
    return hs


def _accept(blocks: list[F.Block], p: strata_v3.StratumV3) -> bool:
    n_feas, _total, rho = F.feasible_ratio(blocks)
    if n_feas < 1:
        return False
    lo, hi = p.rho_band
    if not (lo <= rho <= hi):
        return False
    fill = F.min_fill_over_feasible(blocks)
    if fill is None or not (strata_v3.FILL_BAND[0] <= fill <= strata_v3.FILL_BAND[1]):
        return False
    if p.require_crack:
        # both greedy hand-rules must FAIL (pick an infeasible split) -> no universal rule
        for rule in F.HAND_RULES.values():
            if F.split_is_feasible(rule(blocks), blocks):
                return False
    return True


@lru_cache(maxsize=4096)
def build_spec_v3(seed: int, stratum: int) -> RestockSpecV3:
    """Deterministically generate the accepted v3 problem for ``(seed, stratum)``.

    Raises ``RuntimeError`` if no instance passes the acceptance bands within ``_MAX_RESEED_V3``
    reseeds (the collector treats that as a skipped seed; the calibration run measures the rate).
    """
    p = strata_v3.params(stratum)
    n = p.n
    names = tuple(f"obj_goal{i}" for i in range(1, n + 1))
    for attempt in range(_MAX_RESEED_V3):
        rng = _Rng(seed * 97 + stratum + attempt * 100003)
        widths = [_u(rng, F.WIDTH_MIN, F.WIDTH_MAX) for _ in range(n)]
        heights = _sample_heights(rng, n, p.n_forced, p.n_near)
        blocks = [F.Block(names[i], widths[i], heights[i]) for i in range(n)]
        if not _accept(blocks, p):
            continue
        spots = _sample_positions(
            n,
            rng,
            _OBJECT_REGION_X,
            _OBJECT_REGION_Y,
            _EXCLUSION_RADIUS,
            _SAMPLE_BUDGET,
        )
        if spots is None:
            continue
        _rng_shuffle(
            spots, rng
        )  # decorrelate position from (already index-independent) dims
        return RestockSpecV3(
            stratum=stratum,
            n=n,
            names=names,
            widths=tuple(widths),
            heights=tuple(heights),
            floor=tuple(spots),
        )
    raise RuntimeError(
        f"build_spec_v3: no accepted instance for seed={seed} stratum={stratum} "
        f"within {_MAX_RESEED_V3} reseeds (loosen the stratum bands)"
    )


def _rgba(height: float) -> tuple[float, float, float, float]:
    """Cosmetic colour: reddish if tall-only (> short cutoff), greenish if short-eligible."""
    return (0.6, 0.2, 0.2, 1.0) if height > F.SHORT_CUTOFF else (0.1, 0.5, 0.1, 1.0)


def v3_config() -> Restock3DEnvConfig:
    """The v3 env config: the re-balanced (0.27, 0.22) section partition."""
    return Restock3DEnvConfig(section_clearances=F.SECTION_CLEARANCES)


def make_spec_fn(stratum: int):
    """A ``seed -> list[ObjectSpec]`` closure for :class:`ObjectCentricRestock3DEnvV3`."""

    def spec_fn(seed: int) -> list[ObjectSpec]:
        spec = build_spec_v3(int(seed), stratum)
        return [
            (
                spec.names[i],
                (spec.widths[i] / 2.0, _DEPTH_HALF, spec.heights[i] / 2.0),
                _rgba(spec.heights[i]),
            )
            for i in range(spec.n)
        ]

    return spec_fn


def make_pose_fn(stratum: int) -> PoseFn:
    """A ``seed -> {name: (x, y)}`` closure for the env's floor layout."""

    def pose_fn(seed: int) -> dict[str, tuple[float, float]]:
        spec = build_spec_v3(int(seed), stratum)
        return {spec.names[i]: spec.floor[i] for i in range(spec.n)}

    return pose_fn


def stratum_env_args_v3(stratum: int, config: Restock3DEnvConfig | None = None):
    """The ``(spec_fn, pose_fn, section_infos, config)`` tuple for a v3 stratum env."""
    if config is None:
        config = v3_config()
    return (
        make_spec_fn(stratum),
        make_pose_fn(stratum),
        compute_section_infos(config),
        config,
    )
