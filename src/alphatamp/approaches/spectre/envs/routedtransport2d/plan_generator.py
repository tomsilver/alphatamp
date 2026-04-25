"""Closed-form skeleton enumeration (spec §5.1).

For each ``(item_order, color_pair, grasp_mode)`` triple, build the unique
ground-operator skeleton in canonical form. Family-balanced cap to ``k_cap``
(default 30 for N=3, N=4; uncapped for N=2).

The class :class:`ClosedFormSkeletonGenerator` is duck-compatible with
``bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator
.RelationalHeuristicSearchAbstractPlanGenerator`` so collect.py's existing
plumbing (``pool_iter = plan_generator(x0, s0, goal, timeout, bpg)``) can call
either implementation interchangeably.

Why an iterator (not a list): the existing collect.py wraps the generator in
``itertools.islice(pool_iter, K_max)`` (collect.py:140). Returning an iterator
preserves that idiom even though we have a static pool.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterator, Sequence

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundOperator, Object

from alphatamp.approaches.spectre.envs.routedtransport2d import operators as ops
from alphatamp.approaches.spectre.envs.routedtransport2d import topology as topo
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    ProblemInstance,
)
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

# ---- Skeleton construction ------------------------------------------------


def _by_name(items: Sequence[Object]) -> dict[str, Object]:
    return {o.name: o for o in items}


def _build_action_plan(
    problem: ProblemInstance,
    item_order: tuple[str, ...],
    color_pair: tuple[str, ...],
    grasp: str,
) -> list[GroundOperator]:
    """Construct the sequence of GroundOperators for one (order, pair, grasp)."""
    by_name = _by_name(problem.objects)
    robot = by_name["robot_0"]
    pick_op = ops.pick_op_for_grasp(grasp)
    place_op = ops.place_op_for_grasp(grasp)

    out: list[GroundOperator] = []
    cur_zone = problem.robot_home

    for item_name in item_order:
        item_obj = by_name[item_name]
        src = problem.item_sources[item_name]
        dst = problem.item_targets[item_name]

        # Empty hop(s) cur_zone -> src, all within color_pair subgraph.
        for p_name, hop_src, hop_dst in topo.bfs_color_pair_path(
            cur_zone, src, color_pair
        ):
            out.append(
                ops.TraverseEmpty.ground(
                    (robot, by_name[p_name], by_name[hop_src], by_name[hop_dst])
                )
            )

        # Pick at src.
        out.append(pick_op.ground((robot, item_obj, by_name[src])))

        # Loaded hops src -> dst (always 2 hops, same-side).
        for p_name, hop_src, hop_dst in topo.bfs_color_pair_path(src, dst, color_pair):
            color = topo.color_of_passage(p_name)
            loaded_op = ops.loaded_op_for_color(color)
            out.append(
                loaded_op.ground(
                    (
                        robot,
                        by_name[p_name],
                        by_name[hop_src],
                        by_name[hop_dst],
                        item_obj,
                    )
                )
            )

        # Place at dst.
        out.append(place_op.ground((robot, item_obj, by_name[dst])))
        cur_zone = dst

    return out


# ---- Family classification ------------------------------------------------


def _skeleton_family(
    action_plan: Sequence[GroundOperator],
) -> tuple[frozenset[str], str]:
    """Return ``(loaded_color_pair, grasp_mode)`` for the skeleton."""
    loaded_colors: set[str] = set()
    grasp: str | None = None
    for op in action_plan:
        if op.name.startswith("TraverseLoadedColor"):
            loaded_colors.add(op.name[-1])
        elif op.name in ("PickItemTop", "PlaceItemTop"):
            grasp = "top" if grasp is None else grasp
        elif op.name in ("PickItemSide", "PlaceItemSide"):
            grasp = "side" if grasp is None else grasp
    assert grasp is not None, "skeleton has no Pick/Place op"
    assert len(loaded_colors) == 2, f"expected 2 loaded colors, got {loaded_colors}"
    return frozenset(loaded_colors), grasp


# ---- Pool generation + capping --------------------------------------------


def _enumerate_raw_skeletons(
    problem: ProblemInstance,
) -> list[tuple[list[GroundOperator], tuple[frozenset[str], str]]]:
    """Build the raw 36-element pool (uncapped) in canonical iteration order.

    Iteration order: ``(item_order, color_pair, grasp_mode)`` lex-sorted.
    Returned alongside each is the family classification so the cap step can
    group efficiently.
    """
    item_names = tuple(f"item_{i}" for i in range(problem.num_items))
    raw: list[tuple[list[GroundOperator], tuple[frozenset[str], str]]] = []
    for item_order in permutations(item_names):
        for color_pair_set in topo.color_pairs():
            color_pair = tuple(sorted(color_pair_set))
            for grasp in ("side", "top"):
                action_plan = _build_action_plan(problem, item_order, color_pair, grasp)
                family = _skeleton_family(action_plan)
                raw.append((action_plan, family))
    return raw


def _family_balanced_cap(
    raw: list[tuple[list[GroundOperator], tuple[frozenset[str], str]]],
    k_cap: int,
) -> list[list[GroundOperator]]:
    """Take the first ``ceil(k_cap / num_families)`` skeletons per family in canonical
    order.

    Returns ``len <= k_cap`` skeletons preserving family balance.
    """
    if len(raw) <= k_cap:
        return [a for a, _f in raw]
    # Build family buckets in first-seen order.
    families: dict[tuple[frozenset[str], str], list[list[GroundOperator]]] = {}
    family_order: list[tuple[frozenset[str], str]] = []
    for action_plan, family in raw:
        if family not in families:
            families[family] = []
            family_order.append(family)
        families[family].append(action_plan)
    per_family = max(1, k_cap // len(families))
    out: list[list[GroundOperator]] = []
    for family in family_order:
        out.extend(families[family][:per_family])
    return out[:k_cap]


# ---- Public generator class -----------------------------------------------


class ClosedFormSkeletonGenerator:
    """Closed-form pool generator, duck-compatible with the kinder interface.

    Bound to a single ``ProblemInstance``. The ``__call__`` signature mirrors
    ``RelationalHeuristicSearchAbstractPlanGenerator.__call__`` so the
    dispatcher in collect.py treats both interchangeably.
    """

    def __init__(
        self,
        problem: ProblemInstance,
        *,
        seed: int = 0,
        k_cap: int = 30,
    ) -> None:
        del seed  # closed-form enumeration is deterministic; no RNG needed
        self._problem = problem
        self._k_cap = k_cap

    def __call__(
        self,
        x0: object,
        s0: RelationalAbstractState,
        goal: object,
        timeout_s: float,
        bpg: object,
    ) -> Iterator[tuple[list[RelationalAbstractState], list[GroundOperator]]]:
        """Yield ``(state_plan, action_plan)`` pairs in canonical order.

        ``x0``, ``goal``, ``timeout_s``, ``bpg`` are accepted for interface
        compatibility but unused — closed-form enumeration depends only on
        the bound :class:`ProblemInstance`.
        """
        del x0, goal, timeout_s, bpg
        raw = _enumerate_raw_skeletons(self._problem)
        capped = _family_balanced_cap(raw, self._k_cap)
        for action_plan in capped:
            state_plan = reconstruct_trajectory(s0, action_plan)
            yield state_plan, action_plan
