"""Per-seed parametric problem generator for ShelfObstruct3D.

Lays a row of shelf regions along the reachable front band (``n_targets`` *target regions*, each
holding a blocker that obstructs a ground target, plus ``n_free`` *free regions* for relocation)
and optionally places **obstructor** cubes in ``n_obstructed`` free regions' culprit band —
farther than the abstractor's At-radius from the centre (so the region reads ``Clear``) yet
overlapping a placed cube's footprint (so the placement check names a culprit). ``build_spec``
is deterministic in ``seed``; the caller certifies each generated scene and resamples rejects.

**⚠️ Empirical limit (notebook / decisions 2026-08-13): the obstructor is physically INERT.**
The env's shelf stably holds only cubes ≤ 0.07 m wide (a wider one drops to the shelf below,
*measured*), and with a same-size obstructor the largest Clear-but-blocking overlap is
``collision(0.07) − At-radius(0.05) ≈ 0.03`` m — which the placement physics treats as a soft
squeeze, not a block (a certified obstructed candidate refined to SUCCESS). So the geometric
certification passes but the class-1 payoff does **not** materialize; ShelfObstruct3D leans
class-2 like SB2D. The generator is kept for the class-2 / reachability difficulty and as a
foundation if the obstruction geometry is later redesigned (e.g. a wider shelf, thinner cubes,
or a different obstruction axis). The tolerances below document the (inert) band for the record.

FP is intended to rise with ``n_obstructed`` and ``n_targets`` — the M3 difficulty knob — but see
the limit above before relying on the obstruction path for FP.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Optional

# Reachable front band (see the shelf-grasp notes): world-x ~1.40 at these local-y depths.
_FRONT_LY = (0.088, 0.10)
_BLOCKER_SIZE = 0.035  # half-extent; cube is 2x -> 0.07 wide
_TARGET_SIZE = 0.02  # small: the stock ground pick fails on a big target
# A blocker-sized obstructor (0.035) is spawn-stable at the front band; a wider one overhangs the
# thin shelf and drops. The culprit band is then narrow -- farther than the At-radius (0.04, see
# models._AT_XY_TOL) from a region centre yet within the collision distance (0.035+0.035+0.01 =
# 0.08). _BAND_OFFSET=0.055 sits in it (not-At by 0.015, overlap ~0.015). The generator CERTIFIES
# each seed lands in the band and rejects the rest, so the narrowness costs yield, not validity.
_OBSTRUCTOR_SIZE = 0.035
_BAND_OFFSET = 0.055
# Pitch must exceed _BAND_OFFSET (0.055) + collision distance (~0.08) so an obstructor sits in
# exactly one region's band, not its neighbour's; 0.15 clears that (0.095 > 0.08) and keeps every
# region's At disc disjoint (0.15 > 2*0.04). Kept as small as possible so the whole row stays in
# the reliable lateral grasp/place window (~[-0.18, 0.20]; the shelf grasp degrades at the edges).
_REGION_PITCH = 0.15


@dataclass(frozen=True)
class ShelfObstructSpec:
    """The lateral (world-y) layout of one generated problem."""

    target_region_y: list[float]
    free_region_y: list[float]
    obstructed_free: list[int]  # indices into free_region_y that carry an obstructor
    obstructor_y: list[float]


def _lx_range(center: float, half: float = 0.005) -> list[float]:
    """A tight local-x window centred on ``center`` -> a near-point spawn at world-y = center."""
    return [center - half, center + half]


def build_spec(
    seed: int, n_targets: int, n_free: int, n_obstructed: int
) -> ShelfObstructSpec:
    """Lay regions along the front band and choose which free regions are obstructed.

    Deterministic in ``seed`` (a fixed permutation of which free regions get an obstructor and a
    small jitter of the row centre), so a rejected seed genuinely differs from the next.
    """
    rng = _Rng(seed)
    n_regions = n_targets + n_free
    span = (n_regions - 1) * _REGION_PITCH
    y0 = -span / 2 + (rng.uniform() - 0.5) * 0.04  # small row jitter
    ys = [y0 + i * _REGION_PITCH for i in range(n_regions)]
    # Interleave targets and frees deterministically: targets first, then frees.
    target_y = ys[:n_targets]
    free_y = ys[n_targets:]
    obstructed = sorted(rng.sample(range(n_free), min(n_obstructed, n_free)))
    obstructor_y = [free_y[i] + _BAND_OFFSET for i in obstructed]
    return ShelfObstructSpec(target_y, free_y, obstructed, obstructor_y)


def build_task_config(spec: ShelfObstructSpec) -> dict:
    """A ShelfObstruct3D task-config dict realising ``spec`` (shelf 2 of a standard cupboard)."""
    cfg: dict = {
        "description": "ShelfObstruct3D generated obstruction problem",
        "variant_description": "",
        "variant_specific_description": "",
        "robots": {"tidybot": {"robot": {}}},
        "scene": "lab2",
        "fixtures": {
            "cupboard": {
                "cupboard_1": {
                    "length": 0.60198,
                    "depth": 0.254,
                    "shelf_heights": [0.254, 0.254, 0.254],
                    "shelf_partitions": [[], [], []],
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
        "goal_state": [],
    }
    regions = cfg["regions"]
    cubes = cfg["objects"]["cube"]
    init = cfg["initial_state"]
    ly0, ly1 = _FRONT_LY

    # Target regions + their blockers + the ground targets.
    for i, ty in enumerate(spec.target_region_y, start=1):
        regions[f"target_region_{i}"] = {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [[ty - 0.03, ly0, 0.0, ty + 0.03, ly1, 0.03]],
            "rgba": [1.0, 1.0, 0.0, 0.3],
            "yaw_ranges": [[0, 0]],
        }
        regions[f"blocker{i}_init_region"] = {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [
                _lx_range(ty)[:1] + [0.09, 0.0] + _lx_range(ty)[1:] + [0.098, 0.03]
            ],
            "rgba": [1.0, 0.0, 0.0, 0.3],
            "yaw_ranges": [[0, 0]],
        }
        cubes[f"cube_blocker{i}"] = {
            "size": _BLOCKER_SIZE,
            "rgba": [0.5, 0.1, 0.1, 1],
            "mass": 0.1,
        }
        init.append(["on", f"cube_blocker{i}", f"blocker{i}_init_region"])
        cubes[f"cube_target{i}"] = {
            "size": _TARGET_SIZE,
            "rgba": [0.1, 0.5, 0.1, 1],
            "mass": 0.1,
        }
        gy = -0.2 + 0.1 * (i - 1)
        regions[f"ground_{i}_object_init_region"] = {
            "target": "ground",
            "ranges": [[0.55, gy, 0.65, gy + 0.05]],
            "yaw_ranges": [[0, 0]],
        }
        init.append(["on", f"cube_target{i}", f"ground_{i}_object_init_region"])
        cfg["goal_state"].append(["on", f"cube_target{i}", f"target_region_{i}"])

    # Free regions.
    for j, fy in enumerate(spec.free_region_y, start=1):
        regions[f"free_region_{j}"] = {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [[fy - 0.03, ly0, 0.0, fy + 0.03, ly1, 0.03]],
            "rgba": [0.0, 1.0, 1.0, 0.3],
            "yaw_ranges": [[0, 0]],
        }

    # Obstructor cubes (At no region; land in the culprit band of their free region).
    for k, oy in enumerate(spec.obstructor_y, start=1):
        cubes[f"cube_obstructor{k}"] = {
            "size": _OBSTRUCTOR_SIZE,
            "rgba": [0.4, 0.4, 0.4, 1],
            "mass": 0.1,
        }
        # Blocker-sized: sits stably in the front band like the blockers (a wider cube overhangs
        # the thin shelf and drops to the shelf below).
        regions[f"obstructor{k}_init_region"] = {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [
                _lx_range(oy)[:1] + [0.09, 0.0] + _lx_range(oy)[1:] + [0.098, 0.03]
            ],
            "rgba": [0.4, 0.4, 0.4, 0.3],
            "yaw_ranges": [[0, 0]],
        }
        init.append(["on", f"cube_obstructor{k}", f"obstructor{k}_init_region"])
    return cfg


class _Rng:
    """A tiny deterministic RNG (avoids importing numpy just for the layout)."""

    def __init__(self, seed: int) -> None:
        self._s = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF

    def _next(self) -> int:
        self._s = (1103515245 * self._s + 12345) & 0x7FFFFFFF
        return self._s

    def uniform(self) -> float:
        return self._next() / 0x7FFFFFFF

    def sample(self, population: range, k: int) -> list[int]:
        items = list(population)
        for i in range(len(items) - 1, 0, -1):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]
        return items[:k]


def write_task(cfg: dict, path: str) -> str:
    """Write a task-config dict to ``path`` and return it."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return path


def n_free_for_level(n_targets: int) -> tuple[int, int]:
    """(n_free, n_obstructed) for a difficulty level: obstruct all but one free region."""
    n_free = n_targets + 2
    return n_free, n_free - 1
