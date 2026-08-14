"""Unit tests for the Restock3D geometric feasibility gate (F2 over-assign + F3 height).

Pure Python — no MuJoCo — so they run in CI. The MuJoCo integration (pool enumeration,
physics pick + geometric place, the baseline↔oracle gap) is exercised by
``experiments/spectre/restock3d_difficulty.py`` and the collector, which need the env.
"""

from __future__ import annotations

from alphatamp.approaches.spectre.envs.restock3d.geometry import (
    height_ok,
    place_gate,
    region_capacity,
)
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import RegionInfo

# A short cell (holds smalls only) and a tall cell (holds talls), single-object footprints.
_SHORT = RegionInfo(
    "region_1_1", 1, (1.4, 0.0), (0.01, 0.03), cell_clearance=0.241, surface_z=0.54
)
_TALL = RegionInfo(
    "region_0_1", 0, (1.4, 0.0), (0.01, 0.03), cell_clearance=0.495, surface_z=0.02
)

_SMALL_H, _TALL_H = 0.04, 0.29
_HALF = 0.02


def test_height_ok_small_fits_both() -> None:
    assert height_ok(_SMALL_H, _SHORT.cell_clearance)
    assert height_ok(_SMALL_H, _TALL.cell_clearance)


def test_height_ok_tall_only_tall_cell() -> None:
    assert not height_ok(_TALL_H, _SHORT.cell_clearance)  # 0.29 > 0.241
    assert height_ok(_TALL_H, _TALL.cell_clearance)  # 0.29 < 0.495


def test_region_capacity_single_object() -> None:
    # A 0.06 m strip (half 0.03) holds one 0.04 cube + margins.
    assert region_capacity(_SHORT, _HALF) == 1


def test_place_gate_f3_tall_in_short() -> None:
    family, culprits = place_gate(_SHORT, _TALL_H, 0.025, residents=())
    assert family == "F3"
    assert culprits == ()


def test_place_gate_f3_takes_precedence_over_f2() -> None:
    # A tall object into an occupied short cell is F3 (height checked first).
    family, _ = place_gate(_SHORT, _TALL_H, 0.025, residents=("cube_goal1",))
    assert family == "F3"


def test_place_gate_f2_over_assignment() -> None:
    family, culprits = place_gate(_SHORT, _SMALL_H, _HALF, residents=("cube_goal1",))
    assert family == "F2"
    assert culprits == ("cube_goal1",)


def test_place_gate_feasible() -> None:
    family, culprits = place_gate(_SHORT, _SMALL_H, _HALF, residents=())
    assert family is None
    assert culprits == ()


def test_place_gate_tall_into_empty_tall_cell_feasible() -> None:
    family, _ = place_gate(_TALL, _TALL_H, 0.025, residents=())
    assert family is None
