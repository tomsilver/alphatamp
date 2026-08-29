"""Restock3D **v2** geometry-informed A* plan generator (nearest-first pick prior).

Subclasses the stock hff plan generator
(:class:`RelationalHeuristicSearchAbstractPlanGenerator`) and replaces the hardcoded unit operator
cost with a **state-dependent pick cost**::

    c(pick(o), s) = 1 + lam * |{ o' unpicked (OnFloor) in s : d(o') < d(o) }|

``d(o)`` is a per-episode object "distance from the park pose" — here the **northward reach** to the
object, which equals its y-coordinate. The holonomic base slides laterally (in x) for free along a
clear southern corridor to line up directly south of each object, so the only costly,
reach-over-relevant axis is y (``_blocks_reach``: a southern object blocks reach to a northern one).
All other operators keep unit cost and the hff heuristic is unchanged, so the **total extra penalty of
a complete plan is exactly its Kendall-tau inversion count relative to the nearest-first
(south-to-north = v2 oracle) pick order** (``oracle_v2.build_skeleton_v2``).

Because a 0-inversion plan is strictly cheaper than any inverted plan and the hff heuristic is 0 at
every goal, A* yields the 0-inversion (oracle) pick order — every section variant of it — before any
inverted order. It is a deliberately *weak* prior: it ranks skeletons only by how far their pick order
deviates from nearest-first, and does nothing about the tall/short section choice.
"""

from __future__ import annotations

import heapq as hq
import time
from typing import Iterator, Optional

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

_PICK_OP = "pick"
_FLOOR_PRED = "OnFloor"


def pick_distance_from_state(x0, goal_names: list[str]) -> dict[str, float]:
    """Per-episode ``d(o)``: northward reach to each goal object = its y-coordinate (read once).

    Read from the *initial* scene ``x0`` (object floor poses are fixed at reset). Equivalent to the
    key :func:`oracle_v2.build_skeleton_v2` sorts on, so nearest-first == the oracle south-to-north
    order.
    """
    return {n: float(x0.get_object_pose(n).position[1]) for n in goal_names}


class GeometryGuidedRestockPlanGenerator(
    RelationalHeuristicSearchAbstractPlanGenerator
):
    """Hff A* with a nearest-first pick cost (see the module docstring)."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        operators: set[LiftedOperator],
        seed: int,
        *,
        pick_distance: dict[str, float],
        lam: float = 1.0,
        precomputed_ground_operators: Optional[set[GroundOperator]] = None,
    ) -> None:
        super().__init__(
            types,
            predicates,
            operators,
            heuristic_name="hff",
            seed=seed,
            precomputed_ground_operators=precomputed_ground_operators,
        )
        self._pick_distance = pick_distance
        self._lam = float(lam)

    def _edge_cost(self, s: RelationalAbstractState, a: GroundOperator) -> float:
        """Unit cost for every op except ``pick(o)``, which pays ``1 + lam * (# nearer
        OnFloor)``.

        In the state ``s`` where ``pick(o)`` is applied, ``o`` is still ``OnFloor`` (its
        precondition) and every not-yet-picked goal object is ``OnFloor``, so the count
        below is exactly the number of nearer objects skipped over.
        """
        if a.name != _PICK_OP:
            return 1.0
        target = a.parameters[1].name
        d_o = self._pick_distance[target]
        n_nearer = sum(
            1
            for atom in s.atoms
            if atom.predicate.name == _FLOOR_PRED
            and atom.objects[0].name != target
            and self._pick_distance.get(atom.objects[0].name, d_o) < d_o
        )
        return 1.0 + self._lam * n_nearer

    def __call__(
        self,
        x0,
        s0: RelationalAbstractState,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph,
    ) -> Iterator[tuple[list, list]]:
        # Faithful copy of the base A* enumeration loop (states are revisited so multiple abstract
        # plans can be generated); the ONLY change is the child edge cost -- self._edge_cost(...)
        # instead of the hardcoded +1.0.
        start_time = time.perf_counter()
        assert goal.check_abstract_state is not None

        heuristic = self._heuristic_factory(s0, goal)

        queue: list[tuple[float, float, _Node]] = []
        root: _Node = _Node(
            s0, last_abstract_action=None, parent=None, cumulative_cost=0.0
        )
        hq.heappush(queue, (heuristic(s0), self._rng.uniform(), root))

        visited_abstract_plans: set[tuple] = set()
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

                child_node: _Node = _Node(
                    ns,
                    last_abstract_action=a,
                    parent=node,
                    cumulative_cost=node.cumulative_cost
                    + self._edge_cost(node.abstract_state, a),
                )
                priority = child_node.cumulative_cost + heuristic(ns)
                hq.heappush(queue, (priority, self._rng.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break
