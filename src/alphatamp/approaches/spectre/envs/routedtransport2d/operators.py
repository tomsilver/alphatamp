"""Types, predicates, and lifted operators for RoutedTransport2D (spec §2.2-2.4).

Eight lifted operators total:

- PickItemTop, PickItemSide       — pick splits by grasp mode (spec §2.4)
- PlaceItemTop, PlaceItemSide     — place splits by grasp mode
- TraverseEmpty                   — generic empty traversal (any passage subtype)
- TraverseLoadedColorA/B/C        — loaded traversal split by passage color

Passage subtyping is load-bearing: typed-local-id renumbering (per
``docs/archive/SPECTRE_METHOD_SPEC.md`` §4.1.4) groups objects by type, so the
canonical key
for ``TraverseLoadedColorA(...)`` is distinct from ``TraverseLoadedColorB(...)``.
This is what gives B4 access to the color-family signal (spec lines 99-103).

Width/size compatibility is intentionally **not** in any operator precondition
— it is checked at refinement time only (spec lines 38-43, 124-126). That
asymmetry is the mechanism by which per-problem tags confound B4's pair table.
"""

from __future__ import annotations

from typing import Final

from relational_structs import LiftedAtom, LiftedOperator, Predicate, Type, Variable

# ---- Types ----------------------------------------------------------------

# Top-level (no parent). Robot, Item, Zone are all distinct sources of
# typed-local-id namespaces; each will be renumbered independently by
# canonicalize.canonicalize_episode (SPEC §4.1.4).
RobotType: Final[Type] = Type("robot")
ItemType: Final[Type] = Type("item")
ZoneType: Final[Type] = Type("zone")

# Passage hierarchy: parent ``passage`` plus three color subtypes.
# TraverseEmpty takes parent type so it accepts any color; TraverseLoadedColorA
# takes the corresponding subtype so the canonical key encodes color.
PassageType: Final[Type] = Type("passage")
PassageColorAType: Final[Type] = Type("passage_color_a", parent=PassageType)
PassageColorBType: Final[Type] = Type("passage_color_b", parent=PassageType)
PassageColorCType: Final[Type] = Type("passage_color_c", parent=PassageType)

# Level constants are typed objects too — they appear as arguments to the
# static atoms PassageWidth and ItemSize. SesameModels.types is a flat set;
# Object.is_instance walks the parent chain.
WidthLevelType: Final[Type] = Type("width_level")
SizeLevelType: Final[Type] = Type("size_level")

ALL_TYPES: Final[set[Type]] = {
    RobotType,
    ItemType,
    ZoneType,
    PassageType,
    PassageColorAType,
    PassageColorBType,
    PassageColorCType,
    WidthLevelType,
    SizeLevelType,
}


# ---- Predicates -----------------------------------------------------------

# Dynamic (change over plan; in operator effects).
At = Predicate("At", [RobotType, ZoneType])
ItemAt = Predicate("ItemAt", [ItemType, ZoneType])
HandEmpty = Predicate("HandEmpty", [RobotType])
Holding = Predicate("Holding", [RobotType, ItemType])
HeldGraspTop = Predicate("HeldGraspTop", [RobotType, ItemType])
HeldGraspSide = Predicate("HeldGraspSide", [RobotType, ItemType])

# Static (in s_0 only; never in operator effects).
Connects = Predicate("Connects", [PassageType, ZoneType, ZoneType])
PassageWidth = Predicate("PassageWidth", [PassageType, WidthLevelType])
ItemSize = Predicate("ItemSize", [ItemType, SizeLevelType])

ALL_PREDICATES: Final[set[Predicate]] = {
    At,
    ItemAt,
    HandEmpty,
    Holding,
    HeldGraspTop,
    HeldGraspSide,
    Connects,
    PassageWidth,
    ItemSize,
}


# ---- Lifted operators -----------------------------------------------------


def _build_pick(name: str, grasp_pred: Predicate) -> LiftedOperator:
    robot = Variable("?robot", RobotType)
    item = Variable("?item", ItemType)
    zone = Variable("?zone", ZoneType)
    return LiftedOperator(
        name=name,
        parameters=[robot, item, zone],
        preconditions={
            LiftedAtom(At, [robot, zone]),
            LiftedAtom(ItemAt, [item, zone]),
            LiftedAtom(HandEmpty, [robot]),
        },
        add_effects={
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(grasp_pred, [robot, item]),
        },
        delete_effects={
            LiftedAtom(ItemAt, [item, zone]),
            LiftedAtom(HandEmpty, [robot]),
        },
    )


def _build_place(name: str, grasp_pred: Predicate) -> LiftedOperator:
    robot = Variable("?robot", RobotType)
    item = Variable("?item", ItemType)
    zone = Variable("?zone", ZoneType)
    return LiftedOperator(
        name=name,
        parameters=[robot, item, zone],
        preconditions={
            LiftedAtom(At, [robot, zone]),
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(grasp_pred, [robot, item]),
        },
        add_effects={
            LiftedAtom(ItemAt, [item, zone]),
            LiftedAtom(HandEmpty, [robot]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(grasp_pred, [robot, item]),
        },
    )


def _build_traverse_empty() -> LiftedOperator:
    robot = Variable("?robot", RobotType)
    passage = Variable("?passage", PassageType)
    src = Variable("?src", ZoneType)
    dst = Variable("?dst", ZoneType)
    return LiftedOperator(
        name="TraverseEmpty",
        parameters=[robot, passage, src, dst],
        preconditions={
            LiftedAtom(At, [robot, src]),
            LiftedAtom(Connects, [passage, src, dst]),
            LiftedAtom(HandEmpty, [robot]),
        },
        add_effects={LiftedAtom(At, [robot, dst])},
        delete_effects={LiftedAtom(At, [robot, src])},
    )


def _build_traverse_loaded(name: str, passage_subtype: Type) -> LiftedOperator:
    robot = Variable("?robot", RobotType)
    passage = Variable("?passage", passage_subtype)
    src = Variable("?src", ZoneType)
    dst = Variable("?dst", ZoneType)
    item = Variable("?item", ItemType)
    return LiftedOperator(
        name=name,
        parameters=[robot, passage, src, dst, item],
        preconditions={
            LiftedAtom(At, [robot, src]),
            LiftedAtom(Connects, [passage, src, dst]),
            LiftedAtom(Holding, [robot, item]),
        },
        add_effects={LiftedAtom(At, [robot, dst])},
        delete_effects={LiftedAtom(At, [robot, src])},
    )


PickItemTop: Final[LiftedOperator] = _build_pick("PickItemTop", HeldGraspTop)
PickItemSide: Final[LiftedOperator] = _build_pick("PickItemSide", HeldGraspSide)
PlaceItemTop: Final[LiftedOperator] = _build_place("PlaceItemTop", HeldGraspTop)
PlaceItemSide: Final[LiftedOperator] = _build_place("PlaceItemSide", HeldGraspSide)
TraverseEmpty: Final[LiftedOperator] = _build_traverse_empty()
TraverseLoadedColorA: Final[LiftedOperator] = _build_traverse_loaded(
    "TraverseLoadedColorA", PassageColorAType
)
TraverseLoadedColorB: Final[LiftedOperator] = _build_traverse_loaded(
    "TraverseLoadedColorB", PassageColorBType
)
TraverseLoadedColorC: Final[LiftedOperator] = _build_traverse_loaded(
    "TraverseLoadedColorC", PassageColorCType
)

ALL_OPERATORS: Final[set[LiftedOperator]] = {
    PickItemTop,
    PickItemSide,
    PlaceItemTop,
    PlaceItemSide,
    TraverseEmpty,
    TraverseLoadedColorA,
    TraverseLoadedColorB,
    TraverseLoadedColorC,
}


def passage_subtype_for_color(color: str) -> Type:
    """Return the subtyped passage Type for a given color letter."""
    return {
        "A": PassageColorAType,
        "B": PassageColorBType,
        "C": PassageColorCType,
    }[color]


def loaded_op_for_color(color: str) -> LiftedOperator:
    """Return the TraverseLoaded operator for a given color letter."""
    return {
        "A": TraverseLoadedColorA,
        "B": TraverseLoadedColorB,
        "C": TraverseLoadedColorC,
    }[color]


def pick_op_for_grasp(grasp: str) -> LiftedOperator:
    """Return the Pick operator for a given grasp mode ('top'/'side')."""
    return PickItemTop if grasp == "top" else PickItemSide


def place_op_for_grasp(grasp: str) -> LiftedOperator:
    """Return the Place operator for a given grasp mode ('top'/'side')."""
    return PlaceItemTop if grasp == "top" else PlaceItemSide
