"""Offline nesting solver, the packing-margin radius η, and the label rule (spec §7).

The nester is a depth-first bottom-left placer over the NFP/IFP arrangement of chunk 1
(:mod:`.geometry`) — the standard exact-in-practice 2D irregular-packing recipe (§7.2).
It is *not* a heuristic packer: on the infeasible side it **exhausts** its discretized
search space (intensified mode: fine rotation grid, high node caps) so that an
``INFEASIBLE`` return certifies "no nest exists at this discretization", which is what
the bimodal label rule (§7.3) and η require. All feasibility geometry is shared with the
sampler (chunk 4) and the checker-in-loop baseline, and every op is counted (§5.3).

Core quantities (spec §2.8):

* ``N(S, r)`` (:func:`packs`) — 1 iff the r-inflated shapes of ``S`` admit an
  interior-disjoint placement inside the raw tray ``T`` (free rotation + translation).
* ``η(S)`` (:func:`packing_margin_radius`) — ``sup {r ≥ 0 : N(S, r) = 1}``, by bisection.

Honesty note (§7.2): the search is complete only w.r.t. its discretization; a razor-thin
missed nest is a one-sided error, harmless to the stream-level bimodality C2 actually
operationalizes because such a nest is unreachable by the sampling refiner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import geometry
from .counters import OpCounter
from .params import DifficultyDials

_AREA_EPS = 1e-7  # slack for numerical interior-disjoint / containment guards (cm²)


class NestStatus(Enum):
    """Outcome of a :func:`nest` call (spec §7.2)."""

    FOUND = "found"
    INFEASIBLE = "infeasible"  # search space exhausted, no nest at this discretization
    TIMEOUT = "timeout"  # node cap hit before exhaustion — indeterminate


@dataclass(frozen=True)
class Placement:
    """One placed shape: input index + reference-point translation + rotation."""

    shape_index: int
    x: float
    y: float
    theta_rad: float


@dataclass(frozen=True)
class NestResult:
    """A nester outcome with the placement certificate when ``FOUND`` (spec §7.2)."""

    status: NestStatus
    radius: float
    nodes: int
    placements: list[Placement] | None = None


@dataclass(frozen=True)
class NesterConfig:
    """Search discretization and budget for :func:`nest` (spec §7.2)."""

    rot_grid_deg: float = 5.0
    n_restarts: int = 3
    node_cap: int = 20_000
    include_midpoints: bool = True
    include_intersections: bool = False
    seed: int = 0

    @classmethod
    def exact(cls, *, seed: int = 0) -> "NesterConfig":
        """Generous exact-mode config for labels/generation (spec §7.2)."""
        return cls(rot_grid_deg=5.0, n_restarts=3, node_cap=20_000, seed=seed)

    @classmethod
    def anytime(cls, *, node_cap: int = 2_000, seed: int = 0) -> "NesterConfig":
        """Budget-capped best-effort config (checker-in-loop / triage; spec §7.2)."""
        return cls(rot_grid_deg=5.0, n_restarts=1, node_cap=node_cap, seed=seed)

    @classmethod
    def intensified(cls, *, seed: int = 0) -> "NesterConfig":
        """Fine-grid, high-cap config for infeasible certification (spec §7.2)."""
        return cls(
            rot_grid_deg=1.0,
            n_restarts=2,
            node_cap=200_000,
            include_intersections=True,
            seed=seed,
        )


class _NodeCapExceeded(Exception):
    """Internal: unwinds the DFS when the node cap is hit (→ TIMEOUT)."""


def _rotations(grid_deg: float, offset_deg: float) -> list[float]:
    """Rotation grid over [0, 360) in degrees, shifted by ``offset_deg``."""
    steps = max(1, round(360.0 / grid_deg))
    return [offset_deg + k * grid_deg for k in range(steps)]


def _candidate_positions(free: BaseGeometry, cfg: NesterConfig) -> geometry.Vertices:
    """Bottom-left-sorted reference-point candidates on the free-region boundary.

    The free region ``IFP(shape(θ), T) ∖ ⋃ NFP(shape(θ), placed_j)`` already carries the
    NFP–NFP arrangement vertices that lie on its boundary (shapely computes the exact
    arrangement), so its boundary vertices + edge midpoints are the touching placements
    (§7.2). Sorted by (y, then x) for the compaction/bottom-left preference.
    """
    verts = geometry.region_vertices(
        [free],
        include_midpoints=cfg.include_midpoints,
        include_intersections=cfg.include_intersections,
    )
    if len(verts) == 0:
        return verts
    order = np.lexsort((verts[:, 0], verts[:, 1]))  # primary y, secondary x
    return verts[order]


def _valid_placement(
    placed: Polygon, container: Polygon, others: Sequence[Polygon]
) -> bool:
    """Guard: ``placed`` is inside ``container`` and interior-disjoint from others."""
    if placed.difference(container).area > _AREA_EPS:
        return False
    for other in others:
        if placed.intersection(other).area > _AREA_EPS:
            return False
    return True


def nest(
    polys: Sequence[Polygon],
    container: Polygon,
    r: float,
    cfg: NesterConfig = NesterConfig(),
    *,
    counter: OpCounter | None = None,
) -> NestResult:
    """Place the r-inflated ``polys`` interior-disjoint inside ``container`` (spec
    §7.2).

    Returns ``FOUND`` with a placement certificate; ``INFEASIBLE`` if every restart
    exhausts its search space without a nest; ``TIMEOUT`` if the node cap was hit before
    exhaustion (and no restart found a nest). Each shape's reference point is its local
    origin; rotation is about that origin, then round-inflation by ``r``.
    """
    n = len(polys)
    rng = np.random.default_rng(cfg.seed)
    infl_cache: dict[tuple[int, float], Polygon] = {}
    ifp_cache: dict[tuple[int, float], BaseGeometry] = {}
    # NFP between two rotated-inflated shapes at the origin, keyed by rotation only —
    # translation-invariant (NFP(translate(A,t), B) = translate(NFP(A,B), t)), so each
    # is built once and translated at use (the §5.3 NFP-caching win, ~70x).
    nfp_cache: dict[tuple[int, float, int, float], BaseGeometry] = {}

    def rotated_inflated(i: int, theta_deg: float) -> Polygon:
        key = (i, round(theta_deg % 360.0, 6))
        cached = infl_cache.get(key)
        if cached is None:
            rotated = rotate(polys[i], theta_deg, origin=(0.0, 0.0))
            cached = geometry.inflate(rotated, r, counter=counter)
            infl_cache[key] = cached
        return cached

    def ifp_of(i: int, theta_deg: float) -> BaseGeometry:
        key = (i, round(theta_deg % 360.0, 6))
        cached = ifp_cache.get(key)
        if cached is None:
            cached = geometry.ifp(
                rotated_inflated(i, theta_deg), container, counter=counter
            )
            ifp_cache[key] = cached
        return cached

    def nfp_base(j: int, theta_j: float, i: int, theta_i: float) -> BaseGeometry:
        key = (j, round(theta_j % 360.0, 6), i, round(theta_i % 360.0, 6))
        cached = nfp_cache.get(key)
        if cached is None:
            cached = geometry.nfp(
                rotated_inflated(j, theta_j),
                rotated_inflated(i, theta_i),
                counter=counter,
            )
            nfp_cache[key] = cached
        return cached

    area_order = sorted(range(n), key=lambda i: -polys[i].area)
    nodes = [0]

    def dfs(
        depth: int,
        placed: list[tuple[int, float, float, float, Polygon]],
        order: list[int],
        offset_deg: float,
    ) -> list[Placement] | None:
        if depth == n:
            return []
        i = order[depth]
        for theta_deg in _rotations(cfg.rot_grid_deg, offset_deg):
            shape = rotated_inflated(i, theta_deg)
            free: BaseGeometry = ifp_of(i, theta_deg)
            for j, theta_j, tx_j, ty_j, _ in placed:
                if free.is_empty:
                    break
                base = nfp_base(j, theta_j, i, theta_deg)
                free = free.difference(translate(base, xoff=tx_j, yoff=ty_j))
            if free.is_empty or free.area < 0.0:
                continue
            others = [rec[4] for rec in placed]
            for tx, ty in _candidate_positions(free, cfg):
                nodes[0] += 1
                if nodes[0] > cfg.node_cap:
                    raise _NodeCapExceeded()
                placed_poly = translate(shape, xoff=float(tx), yoff=float(ty))
                if not _valid_placement(placed_poly, container, others):
                    continue
                rec = (i, theta_deg, float(tx), float(ty), placed_poly)
                suffix = dfs(depth + 1, placed + [rec], order, offset_deg)
                if suffix is not None:
                    theta_rad = math.radians(theta_deg % 360.0)
                    return [Placement(i, float(tx), float(ty), theta_rad)] + suffix
        return None

    any_timeout = False
    for restart in range(cfg.n_restarts):
        if restart == 0:
            order, offset = area_order, 0.0
        else:
            order = [int(v) for v in rng.permutation(n)]
            offset = float(rng.uniform(0.0, cfg.rot_grid_deg))
        try:
            placements = dfs(0, [], order, offset)
        except _NodeCapExceeded:
            any_timeout = True
            continue
        if placements is not None:
            return NestResult(NestStatus.FOUND, r, nodes[0], placements)
    status = NestStatus.TIMEOUT if any_timeout else NestStatus.INFEASIBLE
    return NestResult(status, r, nodes[0], None)


def packs(
    polys: Sequence[Polygon],
    container: Polygon,
    r: float,
    cfg: NesterConfig = NesterConfig(),
    *,
    counter: OpCounter | None = None,
) -> bool:
    """``N(S, r)`` as a boolean oracle: ``FOUND`` → True, else False (spec §2.8).

    ``TIMEOUT`` maps to False (conservative); inspect :func:`nest` directly to
    distinguish an exhausted-infeasible result from a capped one.
    """
    return nest(polys, container, r, cfg, counter=counter).status is NestStatus.FOUND


def packing_margin_radius(
    polys: Sequence[Polygon],
    container: Polygon,
    cfg: NesterConfig = NesterConfig(),
    *,
    r_hi: float,
    tol: float = 0.02,
    counter: OpCounter | None = None,
) -> tuple[float, bool]:
    """Compute η(S) by bisection on the non-increasing ``N(S, r)`` (spec §2.8, §10.0).

    Returns ``(eta, any_timeout)``. ``eta`` is ``-inf`` if the shapes cannot pack even
    at r = 0, ``r_hi`` if they still pack at ``r_hi`` (η ≥ r_hi), else the bisection
    estimate to within ``tol``. ``any_timeout`` flags that some probe hit the node cap
    (η may be under-estimated).
    """
    timed_out = [False]

    def n_of(r: float) -> bool:
        result = nest(polys, container, r, cfg, counter=counter)
        if result.status is NestStatus.TIMEOUT:
            timed_out[0] = True
        return result.status is NestStatus.FOUND

    if not n_of(0.0):
        return (float("-inf"), timed_out[0])
    if n_of(r_hi):
        return (r_hi, timed_out[0])
    lo, hi = 0.0, r_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if n_of(mid):
            lo = mid
        else:
            hi = mid
    return (lo, timed_out[0])


def verify_placements(
    polys: Sequence[Polygon],
    container: Polygon,
    r: float,
    placements: Sequence[Placement],
) -> bool:
    """True iff the r-inflated placed shapes are inside ``container`` and disjoint."""
    built: list[Polygon] = []
    for pl in placements:
        rotated = rotate(
            polys[pl.shape_index], pl.theta_rad, origin=(0.0, 0.0), use_radians=True
        )
        inflated = geometry.inflate(rotated, r)
        built.append(translate(inflated, xoff=pl.x, yoff=pl.y))
    if unary_union([p.difference(container) for p in built]).area > _AREA_EPS:
        return False
    return geometry.interior_disjoint(built)


# --------------------------------------------------------------------------------- #
# Label rule (spec §7.3)
# --------------------------------------------------------------------------------- #
class Label(Enum):
    """Feasibility label for a candidate subset (spec §7.3)."""

    FEASIBLE = "feasible"  # η ≥ r_f
    INFEASIBLE = "infeasible"  # η < r_i
    MARGINAL = "marginal"  # r_i ≤ η < r_f — dropped from the candidate set
    INDETERMINATE = "indeterminate"  # nester timed out; caller must escalate/regenerate


@dataclass(frozen=True)
class LabelResult:
    """A candidate label plus the feasible-side nest certificate (spec §7.3)."""

    label: Label
    certificate: NestResult | None
    timed_out: bool


def label_candidate(
    polys: Sequence[Polygon],
    container: Polygon,
    dials: DifficultyDials,
    *,
    feasible_cfg: NesterConfig | None = None,
    infeasible_cfg: NesterConfig | None = None,
    counter: OpCounter | None = None,
) -> LabelResult:
    """Classify a candidate subset feasible / infeasible / marginal (spec §7.3).

    ``feasible(S) ⇔ N(S, r_f)`` (constructive, exact mode); otherwise
    ``infeasible(S) ⇔ ¬N(S, r_i)`` certified by intensified exhaustion, with a nest found
    at ``r_i`` meaning ``marginal`` (``r_i ≤ η < r_f``). If the r_i certification times
    out, the label is ``INDETERMINATE`` and the caller escalates caps or regenerates.
    """
    feas_cfg = feasible_cfg if feasible_cfg is not None else NesterConfig.exact()
    infeas_cfg = (
        infeasible_cfg if infeasible_cfg is not None else NesterConfig.intensified()
    )
    feas = nest(polys, container, dials.r_f_cm, feas_cfg, counter=counter)
    if feas.status is NestStatus.FOUND:
        return LabelResult(Label.FEASIBLE, feas, timed_out=False)
    reach = nest(polys, container, dials.r_i_cm, infeas_cfg, counter=counter)
    if reach.status is NestStatus.FOUND:
        return LabelResult(Label.MARGINAL, reach, timed_out=False)
    if reach.status is NestStatus.INFEASIBLE:
        return LabelResult(Label.INFEASIBLE, None, timed_out=False)
    return LabelResult(Label.INDETERMINATE, None, timed_out=True)
