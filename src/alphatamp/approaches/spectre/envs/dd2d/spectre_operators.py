"""Substrate types / predicates / lifted operators for the DD2D drawer domain.

This is the ``relational_structs`` / ``bilevel_planning`` view of
``domain/drawer_declutter.pddl``, built so the SPECTRE pipeline (vocab
extraction, STRIPS reconstruction, canonicalization, training) can consume
DD2D episodes, for the small, geometry-blind drawer STRIPS domain.

The domain has a single object type (``item``), six predicates, and three
operators. The collision / packing structure that actually determines
feasibility is deliberately *not* modelled here — it lives only in DD2D's
geometric refiner (spec ``docs/dd2d_spec.md`` §6.1). Consequently the shortest
optimistic plan is literally ``retrieve(target)``, which fails geometrically
when the target is blocked; longer plans stage a subset of blockers onto the
buffer first.

These module-level singletons are load-bearing: the converter grounds them
per-object via :meth:`LiftedOperator.ground`, which sets ``GroundOperator.parent``
— required by ``canonicalize.py``, which re-grounds each stored operator through
its parent during training-time object-renumbering augmentation.
"""

from __future__ import annotations

from typing import Final

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Predicate,
    Type,
    Variable,
)

# ---- Types ----------------------------------------------------------------

# Single flat type. Every DD2D object (the target and every blocker) is an
# ``item``; the target is distinguished by the unary ``target`` predicate, not
# by its type — so item local-ids stay fully permutation-augmentable.
ItemType: Final[Type] = Type("item")

ALL_TYPES: Final[set[Type]] = {ItemType}

# ---- Predicates -----------------------------------------------------------

InDrawer: Final[Predicate] = Predicate("in-drawer", [ItemType])
OnBuffer: Final[Predicate] = Predicate("on-buffer", [ItemType])
Holding: Final[Predicate] = Predicate("holding", [ItemType])
Target: Final[Predicate] = Predicate("target", [ItemType])
Extracted: Final[Predicate] = Predicate("extracted", [ItemType])
# Nullary: the gripper-free fluent. arity == 0, no entities.
HandEmpty: Final[Predicate] = Predicate("handempty", [])

ALL_PREDICATES: Final[set[Predicate]] = {
    InDrawer,
    OnBuffer,
    Holding,
    Target,
    Extracted,
    HandEmpty,
}

# name -> Predicate, for the converter's literal parsing.
PREDICATE_BY_NAME: Final[dict[str, Predicate]] = {p.name: p for p in ALL_PREDICATES}

# ---- Lifted operators -----------------------------------------------------


def _build_pick() -> LiftedOperator:
    o = Variable("?o", ItemType)
    return LiftedOperator(
        name="pick",
        parameters=[o],
        preconditions={LiftedAtom(InDrawer, [o]), LiftedAtom(HandEmpty, [])},
        add_effects={LiftedAtom(Holding, [o])},
        delete_effects={LiftedAtom(InDrawer, [o]), LiftedAtom(HandEmpty, [])},
    )


def _build_place_buffer() -> LiftedOperator:
    o = Variable("?o", ItemType)
    return LiftedOperator(
        name="place-buffer",
        parameters=[o],
        preconditions={LiftedAtom(Holding, [o])},
        add_effects={LiftedAtom(OnBuffer, [o]), LiftedAtom(HandEmpty, [])},
        delete_effects={LiftedAtom(Holding, [o])},
    )


def _build_retrieve() -> LiftedOperator:
    o = Variable("?o", ItemType)
    return LiftedOperator(
        name="retrieve",
        parameters=[o],
        preconditions={
            LiftedAtom(HandEmpty, []),
            LiftedAtom(Target, [o]),
            LiftedAtom(InDrawer, [o]),
        },
        add_effects={LiftedAtom(Extracted, [o])},
        delete_effects={LiftedAtom(InDrawer, [o])},
    )


Pick: Final[LiftedOperator] = _build_pick()
PlaceBuffer: Final[LiftedOperator] = _build_place_buffer()
Retrieve: Final[LiftedOperator] = _build_retrieve()

ALL_OPERATORS: Final[set[LiftedOperator]] = {Pick, PlaceBuffer, Retrieve}

# name -> LiftedOperator, for grounding ``task_plan`` steps in the converter.
OPERATOR_BY_NAME: Final[dict[str, LiftedOperator]] = {
    op.name: op for op in ALL_OPERATORS
}
