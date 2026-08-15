"""Eager-validity A* skeleton generator for Restock3D.

A thin subclass of the substrate's ``RelationalHeuristicSearchAbstractPlanGenerator``. It
adds an **eager-validity penalty** to the per-step g-cost, so the informed A* surfaces
feasible skeletons early. The substrate hardcodes a unit action cost
(``bilevel_planning/.../heuristic_search_plan_generator.py:139``) with no cost hook, so
the search loop is copied verbatim and the single cost line changes to
``+ 1.0 + penalty(a, pre_state)``.

Honest note (guide §5): hff ``h`` is on the unit-cost relaxation and does not see the
penalty, so with penalized ``g`` this is a **soft re-ranking** (``f`` is no longer an
admissible lower bound on penalized cost). That is what is wanted -- the objective is the
enumeration *order*, not optimality -- and the search stays lazy/complete, deduping
already-emitted plans. Penalties are large-but-finite, so provably-doomed (tall->short)
skeletons stay in the pool as F3 evidence; the penalty never prunes.
"""

from __future__ import annotations

import heapq as hq
import time
from typing import Callable, Iterator

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
    _Node,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Goal, RelationalAbstractState
from relational_structs import (
    GroundOperator,
    LiftedOperator,
    Predicate,
    Type,
)

_PenaltyFn = Callable[[GroundOperator, RelationalAbstractState], float]


class EagerValidityPlanGenerator(RelationalHeuristicSearchAbstractPlanGenerator):
    """A* over relational abstractions with an eager-validity penalty added to the
    g-cost."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        operators: set[LiftedOperator],
        heuristic_name: str,
        seed: int,
        penalty_fn: _PenaltyFn,
        precomputed_ground_operators: set[GroundOperator] | None = None,
    ) -> None:
        super().__init__(
            types,
            predicates,
            operators,
            heuristic_name,
            seed,
            precomputed_ground_operators,
        )
        self._penalty_fn = penalty_fn

    def __call__(  # type: ignore[override]
        self,
        x0: object,
        s0: RelationalAbstractState,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph,
    ) -> Iterator[tuple[list[RelationalAbstractState], list[GroundOperator]]]:
        # Copy of the parent A* loop; the one change is the penalized child cost (below).
        start_time = time.perf_counter()
        assert goal.check_abstract_state is not None

        heuristic = self._heuristic_factory(s0, goal)  # type: ignore[attr-defined]

        queue: list[tuple[float, float, _Node]] = []
        root: _Node = _Node(
            s0, last_abstract_action=None, parent=None, cumulative_cost=0.0
        )
        hq.heappush(queue, (heuristic(s0), self._rng.uniform(), root))

        visited_abstract_plans: set[tuple[GroundOperator, ...]] = set()
        visited_abstract_plans.add(root.abstract_action_plan)

        while queue and (time.perf_counter() - start_time < timeout):
            _, _, node = hq.heappop(queue)

            if goal.check_abstract_state(node.abstract_state):
                yield list(node.abstract_state_plan), list(node.abstract_action_plan)
                continue

            for a, ns in self._abstract_successor_function(node.abstract_state):
                bpg.add_abstract_state_node(ns)
                bpg.add_abstract_action_edge(node.abstract_state, a, ns)

                abstract_plan = node.abstract_action_plan + (a,)
                if abstract_plan in visited_abstract_plans:
                    continue

                # --- CHANGE vs the substrate: penalize the g-cost by the eager-validity
                # penalty, keyed on the PRE-state ``node.abstract_state`` (so "region
                # already occupied" reads the state before this Place's own effects). The
                # substrate had ``+ 1.0`` only.
                child_node: _Node = _Node(
                    ns,
                    last_abstract_action=a,
                    parent=node,
                    cumulative_cost=node.cumulative_cost
                    + 1.0
                    + self._penalty_fn(a, node.abstract_state),
                )

                priority = child_node.cumulative_cost + heuristic(ns)
                hq.heappush(queue, (priority, self._rng.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break
