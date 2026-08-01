"""Geometry-aware A* heuristic and plan generator for StickButton2D.

**Why this exists.** ``RelationalHeuristicSearchAbstractPlanGenerator`` accepts a
``heuristic_name`` argument, stores it, and then never uses it — line 198 of
``bilevel_planning/abstract_plan_generators/heuristic_search_plan_generator.py``
hardcodes ``create_pyperplan_heuristic("hff", ...)``. So the only way to supply a
domain-specific heuristic is to subclass the *base* generator and pass our own
``heuristic_factory``. That is what :func:`make_plan_generator` does.

**Why hff is not good enough here.** hff scores the delete-relaxed *symbolic* model, in
which ``RobotPressButton*`` applies to every button. It has no way to know that a button
deep on the table is out of the robot's reach (``geometry.robot_reach_max_y``), so the
stick-free plans — which are symbolically shortest — dominate the front of the pool and
are all physically unrefinable. Measured on b3/seed0: first refinable skeleton moves from
index 29 (hff) to 16 with this heuristic. On a scene with no table buttons the extra
terms are inert and this reduces to counting unpressed buttons, matching hff's ordering
exactly (b3/seed1: index 14 both ways).

See ``docs/kinder_stickbutton2d_map.md`` §7 for the measurements.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Sequence

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    HeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Goal,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.utils import RelationalAbstractSuccessorGenerator
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig
from relational_structs.utils import all_ground_operators

from alphatamp.approaches.spectre.envs.stickbutton2d.geometry import (
    ButtonReach,
    button_positions,
    classify_buttons,
    robot_start_xy,
)

_PRESSED = "Pressed"
_HAND_EMPTY = "HandEmpty"
_GRASPED = "Grasped"
_ROBOT_ABOVE = "RobotAboveButton"
_STICK_ABOVE = "StickAboveButton"


def _world_diagonal() -> float:
    """Longest possible distance between two points in the world, from the env config.

    Used only to normalise the distance term into [0, 1) so it can never outweigh a whole
    action. Derived rather than hardcoded for the same reason as the reach limit — and
    because it is genuinely easy to get wrong: a hardcoded 4.3 is *below* the true
    diagonal of 4.3012 and lets the term reach 1.0002.
    """
    cfg = StickButton2DEnvConfig()
    return math.hypot(
        float(cfg.world_max_x) - float(cfg.world_min_x),
        float(cfg.world_max_y) - float(cfg.world_min_y),
    )


_WORLD_DIAGONAL = _world_diagonal()

# Weight on the remaining-button count. Must be strictly above 1, for a structural
# reason:
# each press adds 1 to g and removes 1 from the count, so at weight exactly 1 the score
# g + h is *depth-invariant* and the search has no reason to go deeper — it plateaus over
# shallow states. b5 survives that; b10 does not (measured: empty pool after 30 s, versus
# 200 plans in 1.4 s at 1.05).
#
# It is deliberately only *just* above 1. Larger values buy nothing and cost the thing
# that
# actually matters, namely how much the 200 candidates differ in their *opening* moves:
# refinement failures happen at step 0-1, so a pool whose members share a prefix all fail
# together. Measured on b5 over 200 candidates (distinct first press / first three):
# 1.05 -> 5/32, same as weight 1.0; 1.5 -> 2/7; 2.0 -> 1/2.
_COUNT_WEIGHT = 1.05


def _distance_to_nearest(
    here: tuple[float, float], remaining: list[tuple[float, float]]
) -> float:
    """Distance from ``here`` to the closest remaining button, normalised to [0, 1)."""
    if not remaining:
        return 0.0
    return min(math.dist(here, p) for p in remaining) / _WORLD_DIAGONAL


def _robot_xy(
    state: RelationalAbstractState,
    positions: dict[str, tuple[float, float]],
    start_xy: tuple[float, float],
) -> tuple[float, float]:
    """Where the robot is, read off the abstract state.

    ``RobotAboveButton`` / ``StickAboveButton`` pin it to a button; otherwise it has not
    moved from its initial pose.
    """
    for atom in state.atoms:
        if atom.predicate.name in (_ROBOT_ABOVE, _STICK_ABOVE):
            name = str(atom.objects[-1].name)
            if name in positions:
                return positions[name]
    return start_xy


def _goal_button_names(goal: Goal) -> frozenset[str]:
    """The buttons the goal requires pressed.

    StickButton2D's goal is a static conjunction of unary ``Pressed`` atoms, so the
    argument of each atom names one required button.
    """
    atoms = getattr(goal, "atoms", ())
    return frozenset(
        str(atom.objects[0].name) for atom in atoms if atom.predicate.name == _PRESSED
    )


def button_count_heuristic(
    goal_buttons: frozenset[str],
    reach: ButtonReach,
    positions: dict[str, tuple[float, float]] | None = None,
    start_xy: tuple[float, float] | None = None,
) -> Callable[[RelationalAbstractState], float]:
    """Build ``h(s)`` for a fixed problem: *do the obvious work, in the obvious order.*

    ``h(s) = 1.05 * |unpressed|
             + 1 if some unpressed button needs the stick and the hand is empty
             + 1 if some unpressed button needs the bare robot and the stick is held
             + (distance from the robot to the nearest unpressed button) / diagonal``

    **Count the unpressed buttons.** Every unpressed button costs at least one press
    action, and no operator ever deletes ``Pressed``, so this term alone is admissible.

    **Add one for a pickup you can already see coming.** If any remaining button is a
    table button (out of the robot's reach) and the hand is empty, a ``PickStick`` must
    happen somewhere in the future — so charge for it now rather than discovering it at
    depth. Without this, stick-free plans look cheaper right up until they fail
    refinement.

    **Symmetrically for putting it down.** If a remaining button requires the bare robot
    while the stick is held, a ``PlaceStick`` is coming. On stock StickButton2D
    ``reach.robot_only`` is empty so this never fires; it is written out so the heuristic
    is complete for a variant where it would.

    **Then prefer being close to the next button.** This is the term that matters at 5+
    buttons, and the reason is specific to this env: *the robot presses any button
    it drives over*. A plan therefore fails when the robot crosses a button it had not
    gotten to yet — the world runs ahead of the plan, and the refiner's exact state check
    rejects it. But if you always go to the **nearest** remaining button, nothing
    unpressed
    can lie on the way, because anything on that segment would have been nearer and would
    have been your target instead. So "always walk to the nearest one" and "never press a
    button out of order" are the same preference.

    This term is load-bearing because every press ordering has the *same plan length*:
    without it A* sees all of them as tied — 120 orderings at 5 buttons, ~3.6M at 10 —
    and
    enumerates essentially arbitrarily.

    **Why distance-to-nearest, not remaining-tour-length.** Both sound reasonable; only
    one works. Each press adds 1 to ``g`` and removes 1 from ``|unpressed|``, so the
    counting part is *constant along every path* and the distance term is the sole
    discriminator — which is what lets a term normalised below 1 still steer the search.
    Ranking by *remaining* tour then backfires: clearing a far outlier early shrinks what
    is left, so A* rates far-first plans best, the exact opposite of what we want.
    Distance-to-the-next-button has no such inversion — going far leaves you far from the
    rest — and it makes A* descend greedily nearest-first. Measured: remaining-tour
    reached
    a first success only at candidate 145 on b5/seed5; see
    ``docs/autonomous_stickbutton_session.md``.

    Normalised by the world diagonal, so the term stays in [0, 1] and cannot outweigh a
    whole action. Omit ``positions`` to recover the pure counting heuristic.
    """
    use_walk = positions is not None and start_xy is not None

    def heuristic(state: RelationalAbstractState) -> float:
        pressed = {
            str(atom.objects[0].name)
            for atom in state.atoms
            if atom.predicate.name == _PRESSED
        }
        unpressed = goal_buttons - pressed
        if not unpressed:
            return 0.0

        hand_empty = any(atom.predicate.name == _HAND_EMPTY for atom in state.atoms)
        holding_stick = any(atom.predicate.name == _GRASPED for atom in state.atoms)

        value = _COUNT_WEIGHT * len(unpressed)
        if hand_empty and (unpressed & reach.needs_stick):
            value += 1.0
        if holding_stick and (unpressed & reach.robot_only):
            value += 1.0

        if use_walk:
            assert positions is not None and start_xy is not None
            here = _robot_xy(state, positions, start_xy)
            todo = [positions[b] for b in unpressed if b in positions]
            # Normalised by a bound on any tour (each of the <=len(todo) hops is at most
            # one world diagonal), so the term stays strictly below 1.
            value += _distance_to_nearest(here, todo)
        return value

    return heuristic


#: How many *raw* generator draws to consume while filling a filtered pool. The filter
#: below discards most of what the generator yields on the small variants, so it needs a
#: stop rule of its own or it would spin until the abstract-planning timeout. Measured
#: raw draws to reach 200 acyclic plans: b5 200, b3 ~640. b1 and b2 never get there --
#: their acyclic sets are genuinely finite (1-2 and 6-34) -- so those runs terminate on
#: this cap instead, which is what sizes it.
#:
#: 5000 is ~8x the largest observed requirement and costs 1-4 s at b1, where the padded
#: plans grow to hundreds of operators and each successive draw is slower. The pools are
#: identical at 2000, 5000 and 20000; only the wasted time differs (20000 spent 20-61 s
#: at b1 producing nothing extra).
_RAW_CAP = 5_000


def _is_acyclic(state_plan: Sequence[RelationalAbstractState]) -> bool:
    """Whether a skeleton ever returns to an abstract state it has already been in.

    Identity is the atom set: the object universe is constant within a problem, so two
    states are the same state exactly when they carry the same atoms.
    """
    seen: set[frozenset] = set()
    for state in state_plan:
        key = frozenset(state.atoms)
        if key in seen:
            return False
        seen.add(key)
    return True


class AcyclicPlanGenerator(HeuristicSearchAbstractPlanGenerator):
    """A* pool generator that drops skeletons containing a loop.

    Upstream's search deliberately allows revisiting abstract states -- "that's important
    because we need to generate multiple abstract plans"
    (``heuristic_search_plan_generator.py``) -- and on StickButton2D that licenses padding
    any plan with ``PickStickFromNothing`` / ``PlaceStick`` pairs, which return to ``s_0``
    exactly. The result is pools that look full and are not: at b1 all 200 candidates are
    the same plan with 0-199 pickup/putdown cycles prepended, running to 400 operators.

    Measured acyclic fraction of a 200-draw pool: b1 **1-2**, b2 6-34, b3 73-101, b5
    193-200. So the filter is near-inert exactly where the ranking problem is real, and
    removes a degeneracy where it is not.

    **This is a benchmark-definition choice, not a free simplification.** A padded plan
    can be *genuinely* more refinable than its acyclic core, because ``PlaceStick`` puts
    the stick down somewhere new and re-picking it changes the geometry. What is asserted
    here is that a pool of near-duplicates is the wrong ranking problem, not that the
    dropped plans are infeasible. See
    ``decisions/07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1``.
    """

    def __init__(self, *args: Any, raw_cap: int = _RAW_CAP, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._raw_cap = raw_cap

    def __call__(  # type: ignore[override]
        self,
        x0: Any,
        s0: Any,
        goal: Any,
        timeout: float,
        bpg: BilevelPlanningGraph,
    ) -> Iterator[tuple[list[Any], list[Any]]]:
        drawn = 0
        for state_plan, action_plan in super().__call__(x0, s0, goal, timeout, bpg):
            if drawn >= self._raw_cap:
                return
            drawn += 1
            if _is_acyclic(state_plan):
                yield state_plan, action_plan


def make_plan_generator(
    env_models: SesameModels,
    x0: object,
    seed: int,
    *,
    use_walk_order: bool = True,
    prune_unreachable: bool = True,
    acyclic_only: bool = True,
    raw_cap: int = _RAW_CAP,
) -> HeuristicSearchAbstractPlanGenerator:
    """Build the geometry-aware A* skeleton generator for one problem.

    ``x0`` is the concrete initial state; it is read **once**, for button positions and
    reach. The heuristic is a function of the abstract state thereafter, so no continuous
    state leaks into the search beyond that fixed per-problem table.

    Two geometry-informed pieces, both keyed on the same reach fact:

    - ``use_walk_order`` adds the distance-to-nearest-button term (see
      :func:`button_count_heuristic`).
    - ``prune_unreachable`` refuses to *ground* ``RobotPressButton*`` on a button the
      bare robot cannot reach. Reach belongs in applicability, not in a cost: as a
      heuristic surcharge it is a constant, and pressing an out-of-reach button still
      lowers ``|unpressed|`` by one, so A* keeps rating those plans optimal.

    ``acyclic_only`` (default on) additionally drops looping skeletons — see
    :class:`AcyclicPlanGenerator` for what that buys and what it costs. It is *not*
    geometry-informed: it reads only the abstract state sequence, so it would apply
    unchanged to any environment whose generator revisits states.

    The returned object matches ``RelationalHeuristicSearchAbstractPlanGenerator``'s call
    signature — ``gen(x0, s0, goal, timeout, bpg)`` yielding
    ``(state_plan, action_plan)`` — so it is a drop-in for ``collect.py``.
    """
    reach = classify_buttons(x0)
    positions = button_positions(x0) if use_walk_order else None
    start_xy = robot_start_xy(x0) if use_walk_order else None

    ground_operators = env_models.ground_operators
    if prune_unreachable and reach.needs_stick:
        s0 = env_models.state_abstractor(x0)
        ground_operators = {
            g
            for g in all_ground_operators(env_models.operators, s0.objects)
            if not (
                g.name.startswith("RobotPressButton")
                and str(g.parameters[1].name) in reach.needs_stick
            )
        }

    def heuristic_factory(
        initial_abstract_state: RelationalAbstractState, goal: Goal
    ) -> Callable[[RelationalAbstractState], float]:
        del initial_abstract_state  # the goal names the buttons; s0 adds nothing
        return button_count_heuristic(
            _goal_button_names(goal), reach, positions, start_xy
        )

    successors = RelationalAbstractSuccessorGenerator(
        env_models.operators, ground_operators
    )
    if acyclic_only:
        return AcyclicPlanGenerator(
            heuristic_factory, successors, seed=seed, raw_cap=raw_cap
        )
    return HeuristicSearchAbstractPlanGenerator(
        heuristic_factory, successors, seed=seed
    )
