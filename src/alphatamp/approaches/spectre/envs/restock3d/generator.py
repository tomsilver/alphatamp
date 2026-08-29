"""Per-seed parametric problem generator for the **kinematic** Restock3D (Config B two shelves).

Lays a **tall shelf** (high ceiling — a tall block fits) and a **short shelf** (low ceiling — a
tall block collides with the board above), each holding single-object front-strip regions, and
stages small cubes + tall blocks (+ F1 clutter) on the floor to be stored. Region capacity and
cell height are invisible to the planner (``Place`` has no ``Clear`` precond), so a
height-/capacity-blind A* enumerates goal-reaching skeletons that over-assign a region (F2) or send
a tall block to a short shelf (F3); those genuinely fail refinement by real PyBullet collision, an
oracle avoids them. Four strata of increasing tightness; the difficulty statistic
``d = (sigma_tall, sigma_short)`` = per-shelf region-slot slack, oracle-computable and stored in
provenance.

This module owns only the abstract per-seed layout (region ys, floor spots, clutter). The concrete
shelf/object geometry lives in :mod:`kinematic_env` (``Restock3DEnvConfig``); regions are computed
from that config in :mod:`region_geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_REGION_PITCH = 0.10  # lateral spacing between region centres (world x)

#: Stratum recipes: (n_small, n_tall, n_tall_regions, n_short_regions). Tuned so difficulty rises;
#: r0 slack (~0 FP), r1 short-cell capacity pressure (F2), r2 tall-slot competition + height (F3),
#: r3 both cells tight (F2 + F3 compose).
STRATA: dict[int, tuple[int, int, int, int]] = {
    0: (3, 0, 2, 5),  # sigma_tall=2, sigma_short=4 — slack, ~0 FP
    1: (5, 0, 1, 4),  # sigma_short=0 — short-cell capacity pressure (F2)
    2: (3, 1, 2, 4),  # sigma_tall=1, sigma_short=2 — tall present (F3), solvable
    3: (4, 2, 3, 5),  # sigma_tall=1, sigma_short=2 — F2 + F3 compose (hard tail)
}


class _Rng:
    """Tiny deterministic RNG (mirrors envs/shelf3d/generator._Rng)."""

    def __init__(self, seed: int) -> None:
        self._s = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF

    def _next(self) -> int:
        self._s = (1103515245 * self._s + 12345) & 0x7FFFFFFF
        return self._s

    def uniform(self) -> float:
        return self._next() / 0x7FFFFFFF


@dataclass(frozen=True)
class RestockSpec:
    """One generated Restock3D problem."""

    stratum: int
    n_small: int
    n_tall: int
    tall_region_ys: list[float]
    short_region_ys: list[float]
    small_floor: list[tuple[float, float]] = field(default_factory=list)
    tall_floor: list[tuple[float, float]] = field(default_factory=list)
    clutter_floor: list[tuple[float, float]] = field(default_factory=list)

    @property
    def sigma_tall(self) -> int:
        return len(self.tall_region_ys) - self.n_tall

    @property
    def sigma_short(self) -> int:
        # smalls fit short regions plus any tall regions the talls do not need.
        return (
            len(self.short_region_ys)
            + (len(self.tall_region_ys) - self.n_tall)
            - self.n_small
        )


def _row_ys(n: int, jitter: float) -> list[float]:
    """``n`` region centres (world y) evenly pitched about 0, within the reliable
    window."""
    span = (n - 1) * _REGION_PITCH
    y0 = -span / 2 + jitter
    return [round(y0 + i * _REGION_PITCH, 4) for i in range(n)]


# Object sampling region (fully-lateral layout, decisions/07 2026-08-16): a ~0.6x0.6 m band to the
# -x (LEFT) of the shelf. The base picks every object from ~0.72 m SOUTH (y <= ~0.55), so objects sit
# north of the base corridor (y >= 0.60) and the base never crosses the field. GATE B keeps the old
# fixed 2x3 grid, just remapped into this band, to isolate the layout change; Gate D replaces it with
# region rejection sampling. Spots are >= 0.30 m apart so no floor object obstructs another's grasp.
_OBJECT_REGION_X = (
    -0.80,
    -0.20,
)  # object band x-range (Gate D sampler); right edge clears the base
_OBJECT_REGION_Y = (0.60, 1.20)  # placing footprint at the leftmost shelf region
# Region rejection sampling (decisions/07 2026-08-16), replacing the fixed grid: objects are sampled
# uniformly in the object region, each claiming an exclusion radius no other object may enter; object
# types (cube vs tall block) are assigned in random order. Only xy is sampled -- objects stay
# axis-aligned. Reach-over feasibility (a north object obstructed by a nearer south one) is handled by
# the pick ORDER (south-to-north in the oracle/eager), NOT by the sampler, so the exclusion radius is
# just the front-grasp lateral clearance.
_EXCLUSION_RADIUS = 0.12  # min centre-to-centre between goal objects
_SAMPLE_BUDGET = (
    200  # rejection attempts per object before the whole layout is reseeded
)
_MAX_RESEED = 64  # distinct RNG seeds tried before build_spec raises


def _rng_shuffle(seq: list, rng: _Rng) -> None:
    """In-place Fisher-Yates shuffle using the deterministic ``_Rng``."""
    for i in range(len(seq) - 1, 0, -1):
        j = min(int(rng.uniform() * (i + 1)), i)
        seq[i], seq[j] = seq[j], seq[i]


def _sample_positions(
    n: int,
    rng: _Rng,
    region_x: tuple[float, float],
    region_y: tuple[float, float],
    radius: float,
    budget: int,
) -> list[tuple[float, float]] | None:
    """``n`` axis-aligned positions in ``region_x`` x ``region_y`` with pairwise centre
    distance >= ``radius``, by rejection sampling.

    Returns None if any object cannot be placed within
    ``budget`` attempts (the caller reseeds the whole layout).
    """
    spots: list[tuple[float, float]] = []
    r2 = radius * radius
    for _ in range(n):
        for _attempt in range(budget):
            x = region_x[0] + rng.uniform() * (region_x[1] - region_x[0])
            y = region_y[0] + rng.uniform() * (region_y[1] - region_y[0])
            if all((x - px) ** 2 + (y - py) ** 2 >= r2 for px, py in spots):
                spots.append((round(x, 4), round(y, 4)))
                break
        else:
            return None
    return spots


# Clutter blocks per stratum -- **RETIRED, all 0** (decisions/07 2026-08-16): the unified front grasp is
# not obstructed by a floor neighbour at the grasp config (sweep-verified), so F1 clutter cannot be
# realised; the depth reach-over among goals is the difficulty instead. ``_sample_blockers`` below is
# kept inert (one flag away). MUST match kinematic_env.CLUTTER_PER_STRATUM.
_CLUTTER_PER_STRATUM: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

# v2 SPECTRE strata: the (n_tall x n_short) configs the v2 collection uses. Committed here
# -- NOT runtime-injected -- so config_hash + git_sha pin the recipe (the collection carries
# the stratum key in ``model_kwargs``). Keys 10-15 avoid clobbering r0-r3, which the oracle /
# kmax / sweep still use. Recipe tuple = (n_small, n_tall, n_tall_regions, n_short_regions);
# n_small = n_short (only n_small/n_tall are load-bearing for v2's continuous packing).
# 10-13 are the pilot's symmetric 1x1..4x4; 14/15 are the full collection's asymmetric
# 3x4 (n_tall=3,n_short=4) and 4x3 (n_tall=4,n_short=3).
STRATA_V2_PILOT: dict[int, tuple[int, int, int, int]] = {
    10: (1, 1, 1, 1),
    11: (2, 2, 2, 2),
    12: (3, 3, 3, 3),
    13: (4, 4, 4, 4),
    14: (4, 3, 3, 4),  # 3x4: n_tall=3, n_short=4
    15: (3, 4, 4, 3),  # 4x3: n_tall=4, n_short=3
}
STRATA.update(STRATA_V2_PILOT)
_CLUTTER_PER_STRATUM.update({k: 0 for k in STRATA_V2_PILOT})

# Clutter is placed this far (centre-to-centre) off a target cube's +/-x face -- the front-grasp
# obstruction zone. CALIBRATION PENDING (Gate D ±x sweep); a placeholder that puts the clutter close
# enough to touch the target's exclusion radius without physically overlapping it.
_CLUTTER_GAP = 0.09


def _sample_blockers(
    small_floor: list[tuple[float, float]],
    occupied: list[tuple[float, float]],
    n_clutter: int,
    rng: _Rng,
) -> list[tuple[float, float]] | None:
    """Sample ``n_clutter`` clutter cubes, each adjacent to a distinct cube goal's +/-x
    face (inside the target's exclusion radius -- the point -- but clear of every OTHER
    object).

    Returns positions, or None if a clutter cannot be placed (caller reseeds). The
    blocker is verified to actually obstruct the front grasp by ``verify_spec`` (sample-
    and-verify); this only places it geometrically.
    """
    if n_clutter == 0:
        return []
    clutter: list[tuple[float, float]] = []
    placed = list(occupied)
    for i in range(min(n_clutter, len(small_floor))):
        cx, cy = small_floor[i]
        side = -1.0 if rng.uniform() < 0.5 else 1.0
        bx, by = round(cx + side * _CLUTTER_GAP, 4), round(cy, 4)
        if not _OBJECT_REGION_X[0] <= bx <= _OBJECT_REGION_X[1]:
            side = -side  # try the other face if the first runs off the region edge
            bx = round(cx + side * _CLUTTER_GAP, 4)
        if not _OBJECT_REGION_X[0] <= bx <= _OBJECT_REGION_X[1]:
            return None
        # Clear of every object except its own target (no physical overlap; two half-widths = 0.05).
        others = [p for p in placed if p != (cx, cy)]
        if any((bx - ox) ** 2 + (by - oy) ** 2 < 0.06**2 for ox, oy in others):
            return None
        clutter.append((bx, by))
        placed.append((bx, by))
    return clutter


def build_spec(seed: int, stratum: int) -> RestockSpec:
    """Deterministic-in-seed problem for a stratum via region rejection sampling.

    Region ys (shelf) are stratum-deterministic; the floor layout is seed-dependent.
    Goal objects are sampled in the object region with an exclusion radius (no overlap),
    object types assigned in random order, and -- for strata with blockers -- clutter
    cubes sampled adjacent to a target's +/-x face. On a sampling failure (region too
    full to fit an object or blocker) the whole layout is reseeded with a fresh
    deterministic RNG; after ``_MAX_RESEED`` tries it raises.
    """
    n_small, n_tall, n_tall_reg, n_short_reg = STRATA[stratum]
    tall_ys = _row_ys(n_tall_reg, 0.0)
    short_ys = _row_ys(n_short_reg, 0.0)
    n_total = n_small + n_tall
    n_clutter = _CLUTTER_PER_STRATUM[stratum]

    for attempt in range(_MAX_RESEED):
        rng = _Rng(seed * 97 + stratum + attempt * 100003)
        spots = _sample_positions(
            n_total,
            rng,
            _OBJECT_REGION_X,
            _OBJECT_REGION_Y,
            _EXCLUSION_RADIUS,
            _SAMPLE_BUDGET,
        )
        if spots is None:
            continue
        # Assign types in random order: n_tall of the sampled spots are tall blocks, the rest cubes.
        idx = list(range(n_total))
        _rng_shuffle(idx, rng)
        tall_set = set(idx[:n_tall])
        small_floor = [spots[i] for i in range(n_total) if i not in tall_set]
        tall_floor = [spots[i] for i in range(n_total) if i in tall_set]
        clutter_floor = _sample_blockers(small_floor, spots, n_clutter, rng)
        if clutter_floor is None:
            continue
        return RestockSpec(
            stratum=stratum,
            n_small=n_small,
            n_tall=n_tall,
            tall_region_ys=tall_ys,
            short_region_ys=short_ys,
            small_floor=small_floor,
            tall_floor=tall_floor,
            clutter_floor=clutter_floor,
        )
    raise RuntimeError(
        f"build_spec: could not sample stratum {stratum} seed {seed} in {_MAX_RESEED} tries"
    )


def goal_object_names(spec: RestockSpec) -> list[str]:
    """The names of the goal objects (small cubes + tall blocks) for ``spec``."""
    return [f"cube_goal{i}" for i in range(1, spec.n_small + 1)] + [
        f"block_goal{i}" for i in range(1, spec.n_tall + 1)
    ]
