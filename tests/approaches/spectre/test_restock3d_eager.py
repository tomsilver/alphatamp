"""V0 unit tests for the Restock3D eager-validity tables + penalty + feasibility
classifier.

Fast (no PyBullet): region geometry comes from ``stratum_env_args`` (pure metadata) and
ground operators are built directly. Covers table classification, penalty polarity per
term, and the no-refinement ``is_feasible_skeleton`` classifier. See
``envs/restock3d/eager_tables.py``.
"""

from __future__ import annotations

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)

from alphatamp.approaches.spectre.envs.restock3d.eager_tables import (
    EagerWeights,
    build_tables,
    is_feasible_skeleton,
    make_penalty,
)
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import stratum_env_args
from alphatamp.approaches.spectre.envs.restock3d.models import (
    CubeType,
    HandEmpty,
    Holding,
    InRegion,
    RobotType,
    Stored,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import RegionType

_ROBOT = Object("robot", RobotType)
_R = Variable("?robot", RobotType)
_T = Variable("?target", CubeType)
_RG = Variable("?region", RegionType)

_PLACE = LiftedOperator(
    "place",
    [_R, _T, _RG],
    preconditions={LiftedAtom(Holding, [_R, _T])},
    add_effects={
        LiftedAtom(HandEmpty, [_R]),
        LiftedAtom(InRegion, [_T, _RG]),
        LiftedAtom(Stored, [_T]),
    },
    delete_effects={LiftedAtom(Holding, [_R, _T])},
)
_PICK = LiftedOperator(
    "pick",
    [_R, _T],
    preconditions={LiftedAtom(HandEmpty, [_R])},
    add_effects={LiftedAtom(Holding, [_R, _T])},
    delete_effects={LiftedAtom(HandEmpty, [_R])},
)


def _place(obj: str, region: str) -> GroundOperator:
    return _PLACE.ground((_ROBOT, Object(obj, CubeType), Object(region, RegionType)))


def _pick(obj: str) -> GroundOperator:
    return _PICK.ground((_ROBOT, Object(obj, CubeType)))


def _state(inregion: dict[str, str]) -> RelationalAbstractState:
    """Build a pre-state with ``{obj: region}`` residents (each also Stored)."""
    atoms: set[GroundAtom] = set()
    objects: set[Object] = {_ROBOT}
    for obj, region in inregion.items():
        o = Object(obj, CubeType)
        rg = Object(region, RegionType)
        atoms.add(GroundAtom(InRegion, [o, rg]))
        atoms.add(GroundAtom(Stored, [o]))
        objects |= {o, rg}
    return RelationalAbstractState(atoms, objects)


def _tables_r(stratum: int):
    _, _, region_infos, _ = stratum_env_args(stratum)
    goal_names = ["cube_goal1", "cube_goal2", "cube_goal3"]
    if stratum >= 2:
        goal_names.append("block_goal1")
    if stratum >= 3:
        goal_names = [
            "cube_goal1",
            "cube_goal2",
            "cube_goal3",
            "cube_goal4",
            "block_goal1",
            "block_goal2",
        ]
    return build_tables(region_infos, goal_names), region_infos


def test_build_tables_classifies_sections_and_tall_goal() -> None:
    tables, region_infos = _tables_r(2)
    # r2 has 2 tall regions (region_0_*) and 4 short (region_1_*).
    assert tables.tall_regions == {"region_0_1", "region_0_2"}
    assert tables.short_regions == {
        "region_1_1",
        "region_1_2",
        "region_1_3",
        "region_1_4",
    }
    assert tables.tall_goal == {"block_goal1"}
    # fits: cube anywhere; tall block only in a tall region.
    assert tables.fits("cube_goal1", "region_1_1")
    assert tables.fits("cube_goal1", "region_0_1")
    assert tables.fits("block_goal1", "region_0_1")
    assert not tables.fits("block_goal1", "region_1_1")
    assert set(region_infos) == tables.tall_regions | tables.short_regions


def test_penalty_clean_feasible_place_is_zero() -> None:
    tables, _ = _tables_r(2)
    penalty = make_penalty(tables, EagerWeights())
    # cube into an empty short region, block still has a free tall region: nothing fires.
    assert penalty(_place("cube_goal1", "region_1_1"), _state({})) == 0.0
    # tall block into an empty tall region: fits, not occupied, not a cube-squat.
    assert penalty(_place("block_goal1", "region_0_1"), _state({})) == 0.0


def test_penalty_tall_into_short_is_lambda_h() -> None:
    w = EagerWeights()
    tables, _ = _tables_r(2)
    penalty = make_penalty(tables, w)
    assert penalty(_place("block_goal1", "region_1_1"), _state({})) == w.h


def test_penalty_occupied_region_is_lambda_c() -> None:
    w = EagerWeights()
    tables, _ = _tables_r(2)
    penalty = make_penalty(tables, w)
    # region_1_1 already holds cube_goal2 -> over-assignment.
    assert (
        penalty(
            _place("cube_goal1", "region_1_1"), _state({"cube_goal2": "region_1_1"})
        )
        == w.c
    )


def test_penalty_cube_squats_needed_tall_region_is_lambda_r() -> None:
    w = EagerWeights()
    tables, _ = _tables_r(2)  # 2 tall regions, 1 tall block (block_goal1)
    penalty = make_penalty(tables, w)
    # Both tall regions empty, block unstored: free_tall=2, demand=1 -> no squat.
    assert penalty(_place("cube_goal1", "region_0_1"), _state({})) == 0.0
    # One tall region taken by a cube: free_tall=1, demand=1, 1-1=0 < 1 -> squat fires.
    p = penalty(
        _place("cube_goal1", "region_0_2"), _state({"cube_goal2": "region_0_1"})
    )
    assert p == w.r


def test_penalty_terms_add() -> None:
    w = EagerWeights()
    tables, _ = _tables_r(2)
    penalty = make_penalty(tables, w)
    # tall block into an OCCUPIED short region: T1 (tall->short) + T2 (occupied).
    p = penalty(
        _place("block_goal1", "region_1_2"), _state({"cube_goal3": "region_1_2"})
    )
    assert p == w.h + w.c


def test_pick_is_free() -> None:
    tables, _ = _tables_r(2)
    penalty = make_penalty(tables, EagerWeights())
    assert penalty(_pick("cube_goal1"), _state({})) == 0.0


def test_is_feasible_skeleton() -> None:
    tables, _ = _tables_r(2)
    # Feasible: block -> tall, cubes -> distinct short regions.
    feasible = [
        _pick("block_goal1"),
        _place("block_goal1", "region_0_1"),
        _pick("cube_goal1"),
        _place("cube_goal1", "region_1_1"),
        _pick("cube_goal2"),
        _place("cube_goal2", "region_1_2"),
        _pick("cube_goal3"),
        _place("cube_goal3", "region_1_3"),
    ]
    assert is_feasible_skeleton(feasible, tables)
    # Infeasible: block -> short region (F3).
    bad_height = [_pick("block_goal1"), _place("block_goal1", "region_1_1")]
    assert not is_feasible_skeleton(bad_height, tables)
    # Infeasible: two cubes into the same region (F2 over-assignment).
    bad_crowd = [
        _pick("cube_goal1"),
        _place("cube_goal1", "region_1_1"),
        _pick("cube_goal2"),
        _place("cube_goal2", "region_1_1"),
    ]
    assert not is_feasible_skeleton(bad_crowd, tables)
