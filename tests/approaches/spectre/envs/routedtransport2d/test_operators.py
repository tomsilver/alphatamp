"""Operator vocabulary + STRIPS effects tests (spec §8.3 #3-4)."""

from __future__ import annotations

import pytest
from relational_structs import Object

from alphatamp.approaches.spectre.envs.routedtransport2d import operators as ops
from alphatamp.approaches.spectre.trajectory import apply_operator


def test_eight_lifted_operators_with_expected_names() -> None:
    names = {op.name for op in ops.ALL_OPERATORS}
    assert names == {
        "PickItemTop",
        "PickItemSide",
        "PlaceItemTop",
        "PlaceItemSide",
        "TraverseEmpty",
        "TraverseLoadedColorA",
        "TraverseLoadedColorB",
        "TraverseLoadedColorC",
    }
    assert len(ops.ALL_OPERATORS) == 8


def test_pick_then_place_top_restores_handempty_and_at() -> None:
    """Spec §8.3 #4: pick + place sequence restores HandEmpty (and ItemAt at dst)."""
    robot = Object("robot_0", ops.RobotType)
    item = Object("item_0", ops.ItemType)
    z_src = Object("L1", ops.ZoneType)
    z_dst = Object("L2", ops.ZoneType)

    pick = ops.PickItemTop.ground((robot, item, z_src))
    place = ops.PlaceItemTop.ground((robot, item, z_dst))

    from bilevel_planning.structs import RelationalAbstractState

    s0 = RelationalAbstractState(
        atoms={
            ops.At([robot, z_src]),
            ops.ItemAt([item, z_src]),
            ops.HandEmpty([robot]),
        },
        objects={robot, item, z_src, z_dst},
    )
    s1 = apply_operator(s0, pick)
    # After pick: Holding + HeldGraspTop, no HandEmpty, no ItemAt(item, src).
    assert ops.Holding([robot, item]) in s1.atoms
    assert ops.HeldGraspTop([robot, item]) in s1.atoms
    assert ops.HandEmpty([robot]) not in s1.atoms
    assert ops.ItemAt([item, z_src]) not in s1.atoms

    # Add At(robot, z_dst) so place's preconditions are met (place needs robot at dst).
    s2_atoms = (s1.atoms - {ops.At([robot, z_src])}) | {ops.At([robot, z_dst])}
    s2 = RelationalAbstractState(atoms=s2_atoms, objects=s1.objects)
    s3 = apply_operator(s2, place)
    assert ops.HandEmpty([robot]) in s3.atoms
    assert ops.ItemAt([item, z_dst]) in s3.atoms
    assert ops.Holding([robot, item]) not in s3.atoms
    assert ops.HeldGraspTop([robot, item]) not in s3.atoms


def test_pick_top_then_place_side_blocked_by_grasp_precondition() -> None:
    """Mixed-grasp pick→place must fail to satisfy place's HeldGrasp precondition."""
    robot = Object("robot_0", ops.RobotType)
    item = Object("item_0", ops.ItemType)
    z = Object("L1", ops.ZoneType)
    pick_top = ops.PickItemTop.ground((robot, item, z))
    place_side = ops.PlaceItemSide.ground((robot, item, z))
    # PickItemTop adds HeldGraspTop. PlaceItemSide requires HeldGraspSide.
    assert ops.HeldGraspSide([robot, item]) not in pick_top.add_effects


@pytest.mark.parametrize("color", ["A", "B", "C"])
def test_loaded_traverse_carries_item_param(color: str) -> None:
    op = ops.loaded_op_for_color(color)
    # Loaded traverse signature: (robot, passage, src, dst, item).
    assert len(op.parameters) == 5
    assert op.parameters[0].type is ops.RobotType
    assert op.parameters[4].type is ops.ItemType


def test_traverse_empty_signature() -> None:
    # (robot, passage, src, dst); no item arg.
    assert len(ops.TraverseEmpty.parameters) == 4
    assert ops.TraverseEmpty.parameters[0].type is ops.RobotType
    assert ops.TraverseEmpty.parameters[1].type is ops.PassageType


def test_passage_subtype_dispatch() -> None:
    assert ops.passage_subtype_for_color("A") is ops.PassageColorAType
    assert ops.passage_subtype_for_color("B") is ops.PassageColorBType
    assert ops.passage_subtype_for_color("C") is ops.PassageColorCType
    # Subtype hierarchy holds.
    assert ops.PassageColorAType.parent is ops.PassageType


def test_static_predicates_not_in_operator_effects() -> None:
    """``PassageWidth`` and ``ItemSize`` must not appear in any add/delete effects."""
    static_preds = {ops.PassageWidth, ops.ItemSize}
    for op in ops.ALL_OPERATORS:
        for atom_set in (op.preconditions, op.add_effects, op.delete_effects):
            preds_used = {a.predicate for a in atom_set}
            assert not (
                preds_used & static_preds
            ), f"{op.name} references static predicate in pre/effects"
