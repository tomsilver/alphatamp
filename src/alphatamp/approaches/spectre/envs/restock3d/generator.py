"""Per-seed parametric problem generator for Restock3D (Config B two-level cupboard).

Lays a **tall cell** (shelf 0, clearance ~0.495) and a **short cell** (shelf 1, clearance ~0.241)
of single-object front-strip regions, and stages small cubes + tall blocks on the floor to be
stored. Region capacity and cell height are invisible to the planner (no ``Clear`` precond, DD-3),
so a height-/capacity-blind A* enumerates goal-reaching skeletons that over-assign a region (F2)
or send a tall block to a short cell (F3); those fail the geometric gate (``refine``), an oracle
avoids them. Four strata of increasing tightness; the difficulty statistic ``d = (sigma_tall,
sigma_short)`` = per-cell slot slack, oracle-computable and stored in provenance.

Surfaces/clearances are the *measured* Config B values (shelf 0 surface z=0.017, shelf 1 z=0.537;
see the ADRs). F1 (grasp obstruction / clutter relocation) is deferred, so there is no floor
clutter in v1 — difficulty is carried by F2 + F3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Config B geometry (measured — DD-6/DD-8).
_TALL_SHELF, _SHORT_SHELF = 0, 1
_TALL_SURFACE_Z, _SHORT_SURFACE_Z = 0.017, 0.537
_TALL_CLEARANCE, _SHORT_CLEARANCE = 0.495, 0.241
_SHELF_HEIGHTS = [0.508, 0.254]

# Front-strip depth band (reliable place/grasp window) and region footprint half-width.
_FRONT_LY = (0.085, 0.105)
_REGION_HALF_LX = 0.03
_REGION_PITCH = 0.10  # lateral spacing between region centres (world y)

# Object specs.
_SMALL_HALF = 0.02  # 0.04 cube
_TALL_HALF_XY, _TALL_HALF_Z = 0.025, 0.145  # 0.05 x 0.05 x 0.29 block

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


def _floor_spots(n: int, rng: _Rng) -> list[tuple[float, float]]:
    """``n`` distinct floor staging spots (world x, y)."""
    spots: list[tuple[float, float]] = []
    for i in range(n):
        col, row = i % 3, i // 3
        x = 0.48 + 0.09 * row + (rng.uniform() - 0.5) * 0.02
        y = -0.22 + 0.20 * col + (rng.uniform() - 0.5) * 0.02
        spots.append((round(x, 4), round(y, 4)))
    return spots


def build_spec(seed: int, stratum: int) -> RestockSpec:
    """Deterministic-in-seed problem for a stratum (a fixed row jitter + floor
    layout)."""
    n_small, n_tall, n_tall_reg, n_short_reg = STRATA[stratum]
    rng = _Rng(seed * 97 + stratum)
    tall_ys = _row_ys(n_tall_reg, (rng.uniform() - 0.5) * 0.03)
    short_ys = _row_ys(n_short_reg, (rng.uniform() - 0.5) * 0.03)
    small_floor = _floor_spots(n_small, rng)
    tall_floor = _floor_spots(n_tall, _Rng(seed * 131 + stratum + 7))
    # keep the tall-block floor spots clear of the small ones
    tall_floor = [(x, y + 0.06) for x, y in tall_floor]
    return RestockSpec(
        stratum=stratum,
        n_small=n_small,
        n_tall=n_tall,
        tall_region_ys=tall_ys,
        short_region_ys=short_ys,
        small_floor=small_floor,
        tall_floor=tall_floor,
    )


def _region(
    shelf: int, cy: float, surface_z: float, clearance: float
) -> tuple[dict, dict]:
    """A region JSON entry + its region_meta entry (local box on ``shelf`` at world-y
    ``cy``)."""
    ly0, ly1 = _FRONT_LY
    box = [cy - _REGION_HALF_LX, ly0, 0.0, cy + _REGION_HALF_LX, ly1, 0.03]
    entry = {
        "target": "cupboard_1",
        "shelf": shelf,
        "ranges": [box],
        "rgba": [0.0, 1.0, 1.0, 0.3] if shelf == _SHORT_SHELF else [1.0, 0.7, 0.0, 0.3],
        "yaw_ranges": [[0, 0]],
    }
    meta = {"cell_clearance": clearance, "surface_z": surface_z}
    return entry, meta


def build_task_config(spec: RestockSpec) -> dict:
    """A Restock3D task-config dict realising ``spec``."""
    cfg: dict = {
        "description": f"Restock3D r{spec.stratum} (sigma_tall={spec.sigma_tall}, "
        f"sigma_short={spec.sigma_short})",
        "robots": {"tidybot": {"robot": {}}},
        "scene": "lab2",
        "fixtures": {
            "cupboard": {
                "cupboard_1": {
                    "length": 0.60198,
                    "depth": 0.254,
                    "shelf_heights": _SHELF_HEIGHTS,
                    "shelf_partitions": [[] for _ in _SHELF_HEIGHTS],
                    "shelf_thickness": 0.0127,
                    "side_and_back_open": False,
                }
            }
        },
        "regions": {
            "ground_cupboard_init_region": {
                "target": "ground",
                "ranges": [[1.5, 0.0, 1.5, 0.0]],
                "yaw_ranges": [[90, 90]],
            },
            "robot_0_task_init_region": {
                "target": "ground",
                "ranges": [[-0.1, -0.1, 0.1, 0.1]],
                "yaw_ranges": [[0, 0]],
            },
        },
        "region_meta": {},
        "cameras": {
            "task_view": {
                "position": [-1, 1, 2],
                "lookat": [2, 0, 0],
                "fovy": 42,
                "resolution": [640, 480],
            }
        },
        "objects": {"cube": {}},
        "initial_state": [
            ["on", "cupboard_1", "ground_cupboard_init_region"],
            ["on", "robot", "robot_0_task_init_region"],
        ],
        "goal_objects": [],
        "goal_state": [],
    }
    regions = cfg["regions"]
    meta = cfg["region_meta"]
    cubes = cfg["objects"]["cube"]
    init = cfg["initial_state"]
    goals = cfg["goal_objects"]

    for i, cy in enumerate(spec.tall_region_ys, start=1):
        entry, m = _region(_TALL_SHELF, cy, _TALL_SURFACE_Z, _TALL_CLEARANCE)
        regions[f"region_0_{i}"] = entry
        meta[f"region_0_{i}"] = m
    for i, cy in enumerate(spec.short_region_ys, start=1):
        entry, m = _region(_SHORT_SHELF, cy, _SHORT_SURFACE_Z, _SHORT_CLEARANCE)
        regions[f"region_1_{i}"] = entry
        meta[f"region_1_{i}"] = m

    for i, (fx, fy) in enumerate(spec.small_floor, start=1):
        name = f"cube_goal{i}"
        cubes[name] = {"size": _SMALL_HALF, "rgba": [0.1, 0.5, 0.1, 1], "mass": 0.02}
        regions[f"{name}_init_region"] = {
            "target": "ground",
            "ranges": [[fx - 0.02, fy - 0.02, fx + 0.02, fy + 0.02]],
            "yaw_ranges": [[0, 0]],
        }
        init.append(["on", name, f"{name}_init_region"])
        goals.append(name)
    for i, (fx, fy) in enumerate(spec.tall_floor, start=1):
        name = f"block_goal{i}"
        cubes[name] = {
            "size": [_TALL_HALF_XY, _TALL_HALF_XY, _TALL_HALF_Z],
            "rgba": [0.6, 0.2, 0.2, 1],
            "mass": 0.05,
        }
        regions[f"{name}_init_region"] = {
            "target": "ground",
            "ranges": [[fx - 0.03, fy - 0.03, fx + 0.03, fy + 0.03]],
            "yaw_ranges": [[0, 0]],
        }
        init.append(["on", name, f"{name}_init_region"])
        goals.append(name)
    return cfg


def write_task(cfg: dict, path: str) -> str:
    """Write a task-config dict to ``path`` and return it."""
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return path
