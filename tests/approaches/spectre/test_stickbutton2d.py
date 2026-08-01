"""Tests for the StickButton2D adapters.

Covers the two things that would silently produce a bad dataset if they broke: the reach
classification (which buttons need the stick) and the heuristic built on top of it.

See ``src/alphatamp/approaches/spectre/docs/kinder_stickbutton2d_map.md``.
"""

from __future__ import annotations

import dataclasses

import pytest
from bilevel_planning.structs import RelationalAbstractGoal, RelationalAbstractState
from kinder.envs.kinematic2d.object_types import CircleType, CRVRobotType, RectangleType
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig
from relational_structs import GroundAtom, Object, Predicate

from alphatamp.approaches.spectre.env_registry import get_type_aug_policy
from alphatamp.approaches.spectre.envs.stickbutton2d import (
    button_count_heuristic,
    robot_reach_max_y,
)
from alphatamp.approaches.spectre.envs.stickbutton2d.geometry import ButtonReach
from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (
    _COUNT_WEIGHT,
    _distance_to_nearest,
)

_PRESSED = Predicate("Pressed", [CircleType])
_HAND_EMPTY = Predicate("HandEmpty", [CRVRobotType])
_GRASPED = Predicate("Grasped", [CRVRobotType, RectangleType])

_ROBOT = Object("robot", CRVRobotType)
_STICK = Object("stick", RectangleType)


def _button(i: int) -> Object:
    return Object(f"button{i}", CircleType)


def _state(atoms: set[GroundAtom]) -> RelationalAbstractState:
    return RelationalAbstractState(atoms, {_ROBOT, _STICK})


def _goal(n: int) -> RelationalAbstractGoal:
    return RelationalAbstractGoal(
        {GroundAtom(_PRESSED, [_button(i)]) for i in range(n)}, lambda x: x
    )


def _reach(needs_stick: set[str], robot_only: set[str] | None = None) -> ButtonReach:
    return ButtonReach(
        needs_stick=frozenset(needs_stick),
        robot_only=frozenset(robot_only or set()),
        reach_max_y=robot_reach_max_y(),
    )


def test_reach_limit_matches_config_derivation() -> None:
    """The reach bound is derived from the env config, not hardcoded.

    Pins the documented figure (1.405) so a config change surfaces here rather than as
    a mysterious drop in collection yield.
    """
    cfg = StickButton2DEnvConfig()
    expected = (
        cfg.table_pose.y
        - cfg.robot_base_radius
        + cfg.robot_arm_length
        + cfg.robot_gripper_width / 2.0
        + cfg.button_radius
    )
    assert robot_reach_max_y() == pytest.approx(expected)
    assert robot_reach_max_y() == pytest.approx(1.405)


def test_reach_limit_tracks_a_changed_config() -> None:
    """A taller table moves the bound — i.e. the value really is computed."""
    cfg = dataclasses.replace(
        StickButton2DEnvConfig(), robot_arm_length=0.5, robot_base_radius=0.1
    )
    assert robot_reach_max_y(cfg) == pytest.approx(robot_reach_max_y() + 0.3)


def test_heuristic_is_zero_at_the_goal() -> None:
    """No unpressed buttons ⇒ no remaining cost, whatever the gripper holds."""
    heuristic = button_count_heuristic(frozenset({"button0"}), _reach({"button0"}))
    goal_state = _state(
        {GroundAtom(_PRESSED, [_button(0)]), GroundAtom(_HAND_EMPTY, [_ROBOT])}
    )
    assert heuristic(goal_state) == 0.0


def test_heuristic_counts_unpressed_buttons_when_all_are_reachable() -> None:
    """With no table buttons the extra terms are inert and h == |unpressed|.

    This is the case where the heuristic must reduce to plain unpressed-counting; it is
    why a scene with no stick-needing buttons reproduces hff's ordering exactly.
    """
    heuristic = button_count_heuristic(
        frozenset({"button0", "button1", "button2"}), _reach(set())
    )
    state = _state(
        {GroundAtom(_PRESSED, [_button(0)]), GroundAtom(_HAND_EMPTY, [_ROBOT])}
    )
    assert heuristic(state) == pytest.approx(2 * _COUNT_WEIGHT)


def test_heuristic_charges_for_an_unavoidable_stick_pickup() -> None:
    """A remaining table button + empty hand ⇒ a PickStick is coming; count it now."""
    heuristic = button_count_heuristic(
        frozenset({"button0", "button1"}), _reach({"button1"})
    )
    empty_hand = _state({GroundAtom(_HAND_EMPTY, [_ROBOT])})
    # two presses (weighted) plus exactly one unweighted pickup
    assert heuristic(empty_hand) == pytest.approx(2 * _COUNT_WEIGHT + 1.0)


def test_heuristic_does_not_charge_a_pickup_while_already_holding_the_stick() -> None:
    """Once the stick is held the pickup is spent, so the surcharge must drop off."""
    heuristic = button_count_heuristic(
        frozenset({"button0", "button1"}), _reach({"button1"})
    )
    holding = _state({GroundAtom(_GRASPED, [_ROBOT, _STICK])})
    assert holding is not None
    assert heuristic(holding) == pytest.approx(2 * _COUNT_WEIGHT)


def test_heuristic_ignores_a_pressed_table_button() -> None:
    """The surcharge keys on *remaining* stick-needing buttons, not all of them."""
    heuristic = button_count_heuristic(
        frozenset({"button0", "button1"}), _reach({"button1"})
    )
    state = _state(
        {GroundAtom(_PRESSED, [_button(1)]), GroundAtom(_HAND_EMPTY, [_ROBOT])}
    )
    # button0 remains and is robot-reachable, so no pickup surcharge
    assert heuristic(state) == pytest.approx(1 * _COUNT_WEIGHT)


def test_heuristic_charges_for_an_unavoidable_place_when_holding() -> None:
    """Symmetric term: a robot-only button while holding the stick implies PlaceStick."""
    heuristic = button_count_heuristic(
        frozenset({"button0"}), _reach(set(), robot_only={"button0"})
    )
    holding = _state({GroundAtom(_GRASPED, [_ROBOT, _STICK])})
    assert heuristic(holding) == pytest.approx(1 * _COUNT_WEIGHT + 1.0)


@pytest.mark.parametrize("variant", [1, 2, 3, 5, 10])
def test_every_variant_has_an_explicit_aug_policy(variant: int) -> None:
    """All three StickButton2D types are registered augmentable, explicitly."""
    policy = get_type_aug_policy(f"stickbutton2d_b{variant}")
    assert policy == {"crv_robot": True, "rectangle": True, "circle": True}


def test_count_weight_is_above_one() -> None:
    """The count weight must exceed 1, or the search cannot descend.

    Each press adds 1 to ``g`` and removes 1 from ``|unpressed|``, so at weight exactly 1
    the A* score is depth-invariant and the search plateaus over shallow states. Measured
    consequence at b10: an empty pool after 30 s, versus 200 plans in 1.4 s at 1.05.
    """
    assert _COUNT_WEIGHT > 1.0


def test_count_weight_stays_small_enough_to_preserve_opening_diversity() -> None:
    """...but only just, or the 200 candidates all share an opening move.

    Refinement failures happen at step 0-1, so a pool whose members share a prefix fails
    as a block. Measured on b5 (distinct first press / distinct first three, of 200):
    1.05 -> 5/32 (same as weight 1.0), 1.5 -> 2/7, 2.0 -> 1/2.
    """
    assert _COUNT_WEIGHT <= 1.1


def test_distance_term_can_never_outweigh_an_action() -> None:
    """The distance term is a tie-break, not a cost: it must stay within [0, 1].

    The bound is attained only by the two opposite world corners, which no button can
    occupy (``button_init_position_bounds`` insets them by one radius), so in any real
    scene the term is strictly below 1.
    """
    cfg = StickButton2DEnvConfig()
    corner_to_corner = _distance_to_nearest(
        (cfg.world_min_x, cfg.world_min_y), [(cfg.world_max_x, cfg.world_max_y)]
    )
    assert corner_to_corner == pytest.approx(1.0)

    (lo_x, lo_y), (hi_x, hi_y) = cfg.button_init_position_bounds
    assert _distance_to_nearest((lo_x, lo_y), [(hi_x, hi_y)]) < 1.0

    assert _distance_to_nearest((1.0, 1.0), []) == 0.0
    assert _distance_to_nearest((1.0, 1.0), [(1.0, 1.0)]) == 0.0


def test_distance_term_prefers_the_nearer_button() -> None:
    """The ordering signal itself: closer remaining button => lower h."""
    near = _distance_to_nearest((0.0, 0.0), [(0.2, 0.0), (3.0, 2.0)])
    far = _distance_to_nearest((0.0, 0.0), [(3.0, 2.0)])
    assert near < far
