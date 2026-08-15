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


# A well-spaced floor grid over the reachable staging zone. Spots are >= 0.30 m apart so NO floor
# object blocks another's grasp -- a tall block within ~0.1 m of a cube obstructs the cube's top-down
# pick (the descending arm hits the tall block), which would make feasibility depend on floor layout
# rather than the intended region assignment (F2/F3). Six spots fit the max object count (r3 = 6).
_FLOOR_XS = [0.3, 0.6]
_FLOOR_YS = [-0.25, 0.05, 0.35]


def _floor_spots(n: int, rng: _Rng) -> list[tuple[float, float]]:
    """``n`` distinct, well-spaced floor staging spots (world x, y), grid + small jitter."""
    grid = [(x, y) for y in _FLOOR_YS for x in _FLOOR_XS]
    spots: list[tuple[float, float]] = []
    for i in range(n):
        gx, gy = grid[i % len(grid)]
        spots.append(
            (
                round(gx + (rng.uniform() - 0.5) * 0.02, 4),
                round(gy + (rng.uniform() - 0.5) * 0.02, 4),
            )
        )
    return spots


# Clutter blocks per stratum: one movable clutter cube next to the first cube goal so a top-down grasp
# is obstructed (F1), relocated via a buffer to clear it. **r1 only** (Gate-3, decisions/07 2026-08-15):
# F1 composes with r1's F2 (the eager order surfaces the relocate-first feasible at index 0, oracle-
# solvable, plain order catastrophically censored = the intended difficulty), but NOT with r3's F3 --
# the F1+F3+relocation abstract search is unenumerable (plain censored past K=200, eager times out with
# 0 candidates), though the oracle certifies a feasible exists. r0 stays the ~0-FP floor; r2/r3 stay
# F2+F3. ``CLUTTER_PER_STRATUM`` in kinematic_env MUST match these counts (specs vs positions).
_CLUTTER_PER_STRATUM: dict[int, int] = {0: 0, 1: 1, 2: 0, 3: 0}

# Clutter is placed at this world-frame offset from the blocked cube: +y (toward the shelf) at the
# Gate-1-calibrated gap. A cube's top-down grasp is obstructed for a +y clutter at gap 0.05-0.10 m
# (blocks reliably, clutter itself pickable, no deadlock cycle); +x/-x never block a top-down grasp.
_CLUTTER_DX, _CLUTTER_DY = 0.0, 0.07


def build_spec(seed: int, stratum: int) -> RestockSpec:
    """Deterministic-in-seed problem for a stratum.

    Region ys are **stratum-deterministic** (no per-seed jitter) so the models factory and the
    collection env agree on region geometry without threading the problem id; the floor layout is
    seed-dependent. Clutter blocks (F1) ring the first small goal cube.
    """
    n_small, n_tall, n_tall_reg, n_short_reg = STRATA[stratum]
    rng = _Rng(seed * 97 + stratum)
    tall_ys = _row_ys(n_tall_reg, 0.0)
    short_ys = _row_ys(n_short_reg, 0.0)
    # Allocate cubes AND blocks from ONE well-spaced grid so no two floor objects collide or block
    # each other's grasp. Blocks take the last spots (kept furthest apart from the cube cluster).
    all_floor = _floor_spots(n_small + n_tall, rng)
    small_floor = all_floor[:n_small]
    tall_floor = all_floor[n_small : n_small + n_tall]

    # One clutter cube per blocked goal, placed +y of the first k cube goals so their top-down grasp
    # is obstructed (F1); the clutter is itself pickable (Gate-1 sweep) and relocated to a buffer.
    clutter_floor: list[tuple[float, float]] = []
    n_clutter = _CLUTTER_PER_STRATUM[stratum]
    for i in range(min(n_clutter, len(small_floor))):
        cx, cy = small_floor[i]
        clutter_floor.append((round(cx + _CLUTTER_DX, 4), round(cy + _CLUTTER_DY, 4)))

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


def goal_object_names(spec: RestockSpec) -> list[str]:
    """The names of the goal objects (small cubes + tall blocks) for ``spec``."""
    return [f"cube_goal{i}" for i in range(1, spec.n_small + 1)] + [
        f"block_goal{i}" for i in range(1, spec.n_tall + 1)
    ]
