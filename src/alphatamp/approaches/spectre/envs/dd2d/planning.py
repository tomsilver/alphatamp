"""Diverse task-plan (skeleton) enumeration -- a stand-in for the PIGINet paper's
FORBID-SEARCH (paper lines 142-148): repeatedly find distinct plan skeletons that
achieve the goal, which are later scored/refined in feasibility order.

Primary backend :class:`SymKPlanner` uses SymK's *native top-k* symbolic search
(via unified-planning + up-symk) -- this is literally forbid-based diverse
planning. Fallback :class:`ForbidLoopPlanner` uses pure-Python pyperplan with an
explicit forbid-by-operator-removal loop, for environments without the SymK wheel.

Both return ``list[Skeleton]`` (deduped, plan-length order = the paper's Baseline
refinement order).
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod

from .problem import BlocksWorldProblem
from .skeleton import Action, Skeleton

_DOMAIN_PDDL = os.path.join(os.path.dirname(__file__), "domain", "blocksworld.pddl")


def _dedupe_keep_order(skeletons: list[Skeleton]) -> list[Skeleton]:
    seen: set = set()
    out: list[Skeleton] = []
    for s in skeletons:
        k = s.key()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _sort_baseline_order(skeletons: list[Skeleton]) -> list[Skeleton]:
    """Ascending plan length -- the order the non-learning Baseline refines in."""
    return sorted(skeletons, key=lambda s: (s.length, s.key()))


class DiversePlanner(ABC):
    """Interface: enumerate up to ``k`` distinct skeletons achieving the goal."""

    name: str = "abstract"

    @abstractmethod
    def plan(
        self, problem: BlocksWorldProblem, k: int
    ) -> list[Skeleton]:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Primary backend: SymK native top-k
# --------------------------------------------------------------------------- #
class SymKPlanner(DiversePlanner):
    name = "symk-topk"

    def __init__(self, timeout: float | None = 30.0) -> None:
        self.timeout = timeout

    def plan(self, problem: BlocksWorldProblem, k: int) -> list[Skeleton]:
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import AnytimePlanner

        with tempfile.NamedTemporaryFile("w", suffix=".pddl", delete=False) as f:
            f.write(problem.to_pddl_problem())
            prob_path = f.name
        # The SymK/FastDownward driver writes fixed-name files (e.g. output.sas)
        # into the *current working directory*. Running several planners in the
        # same CWD (the parallel sweep) makes them clobber each other -> corrupted
        # output -> driver crash -> UP "Cannot interpret" parse errors. Isolate
        # each planning call in its own temp CWD so concurrent calls never collide.
        skeletons: list[Skeleton] = []
        cwd = os.getcwd()
        try:
            domain_pddl = getattr(problem, "domain_pddl_path", _DOMAIN_PDDL)
            up_problem = PDDLReader().parse_problem(domain_pddl, prob_path)
            with tempfile.TemporaryDirectory(prefix="symk_") as workdir:
                os.chdir(workdir)
                try:
                    with AnytimePlanner(
                        name="symk", params={"number_of_plans": k}
                    ) as planner:
                        for res in planner.get_solutions(
                            up_problem, timeout=self.timeout
                        ):
                            if res.plan is not None:
                                skeletons.append(_skeleton_from_up_plan(res.plan))
                            if len(_dedupe_keep_order(skeletons)) >= k:
                                break
                except Exception:
                    # a single malformed plan line shouldn't void the whole call;
                    # keep whatever distinct plans were already collected.
                    pass
                finally:
                    os.chdir(cwd)
        finally:
            os.unlink(prob_path)
        return _sort_baseline_order(_dedupe_keep_order(skeletons))[:k]


def _skeleton_from_up_plan(up_plan) -> Skeleton:
    actions: list[Action] = []
    for ai in up_plan.actions:
        op = ai.action.name
        args = tuple(str(p) for p in ai.actual_parameters)
        actions.append(Action(op, args))
    return Skeleton(tuple(actions))


# --------------------------------------------------------------------------- #
# Fallback backend: pyperplan + explicit forbid loop
# --------------------------------------------------------------------------- #
class ForbidLoopPlanner(DiversePlanner):
    """Pure-Python diverse-planning fallback (no SymK).

    True FORBID-SEARCH (paper line 144) forbids whole previously-found *plans* by
    task reformulation; SymK does this natively. Off-the-shelf pyperplan *search*
    is deterministic (every heuristic returns the same single plan), and the cheap
    trick of forbidding individual grounded operators is unsafe here: each block
    has a single pick operator (from its only current table), so removing one
    strands that block.

    Instead we enumerate diverse plans directly with a bounded breadth-first
    search over pyperplan's grounded task, collecting *distinct* goal-reaching
    action sequences (different orderings and short detours) up to a length bound.
    This is genuine diversity; the caps keep it tractable on small fallback
    instances. SymK remains the preferred backend for larger problems.

    ``length_slack`` bounds plan length to ``shortest + slack``. Pass
    ``length_slack=None`` for an **unbounded, k-driven** enumeration: BFS returns the
    ``k`` globally-shortest diverse plans across all lengths (bounded only by
    ``max_expansions``). This is the "fair baseline" mode DD2D uses -- a geometry-blind
    standard planner must enumerate deep enough (ascending length) to reach the plans a
    geometry-informed planner finds directly, so a small slack unfairly caps it below the
    feasible plan length (docs/dd2d.md "Fair baselines").
    """

    name = "pyperplan-bfs-diverse"

    def __init__(
        self,
        length_slack: int | None = 2,
        max_expansions: int = 200_000,
        search: str = "bfs",
        heuristic_factory=None,
        heuristic_tag: str = "",
        **_ignored,
    ) -> None:
        # ``length_slack=None`` -> no depth cap (k-driven). ``**_ignored`` swallows kwargs
        # meant for other planners (e.g. SymK's ``timeout``) so make_planner can thread them.
        self.length_slack = length_slack  # allow plans up to (shortest + slack) longer
        self.max_expansions = max_expansions
        # ``search`` selects the frontier: "bfs" (blind, ascending-length -- the established
        # baseline, unchanged) or "gbf"/"astar" (best-first ordered by a heuristic). The latter
        # need a ``heuristic_factory(task, problem) -> (node -> float)``.
        self.search = search
        self.heuristic_factory = heuristic_factory
        # diagnostics filled in by the last plan() call (read by the experiment harness).
        self.last_expansions = 0
        self.last_starved = False
        if search != "bfs":
            self.name = f"pyperplan-{search}" + (
                f"-{heuristic_tag}" if heuristic_tag else ""
            )

    def plan(self, problem: BlocksWorldProblem, k: int) -> list[Skeleton]:
        from pyperplan.grounding import ground
        from pyperplan.pddl.parser import Parser

        with tempfile.NamedTemporaryFile("w", suffix=".pddl", delete=False) as f:
            f.write(problem.to_pddl_problem())
            prob_path = f.name
        try:
            parser = Parser(
                getattr(problem, "domain_pddl_path", _DOMAIN_PDDL), prob_path
            )
            dom = parser.parse_domain()
            task = ground(parser.parse_problem(dom))
            if self.search == "bfs":
                plans = _bfs_diverse_plans(
                    task, k, self.length_slack, self.max_expansions
                )
                self.last_expansions, self.last_starved = 0, False
                skeletons = [_skeleton_from_op_names(names) for names in plans]
                # blind BFS: canonical ascending-length Baseline order.
                return _sort_baseline_order(_dedupe_keep_order(skeletons))[:k]
            # best-first (gbf/astar): the frontier *discovery order* IS the ranking under study,
            # so we must NOT re-sort by length -- preserve it, dedupe keeping first occurrence.
            if self.heuristic_factory is None:
                raise ValueError(f"search={self.search!r} requires a heuristic_factory")
            heuristic_fn = self.heuristic_factory(task, problem)
            plans, stats = _bestfirst_diverse_plans(
                task,
                k,
                self.length_slack,
                self.max_expansions,
                self.search,
                heuristic_fn,
            )
            self.last_expansions = stats["expansions"]
            self.last_starved = (
                stats["expansions"] >= self.max_expansions and len(plans) < k
            )
            skeletons = [_skeleton_from_op_names(names) for names in plans]
            return _dedupe_keep_order(skeletons)[:k]
        finally:
            os.unlink(prob_path)


def _bfs_diverse_plans(task, k, length_slack, max_expansions):
    """Bounded BFS collecting up to ~k distinct goal-reaching operator-name paths."""
    from collections import deque

    start = task.initial_state
    queue: deque = deque([(start, ())])
    found: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    max_len: int | None = None
    expansions = 0

    while queue and expansions < max_expansions and len(found) < k * 4:
        state, path = queue.popleft()
        if max_len is not None and len(path) >= max_len:
            continue
        for op in task.operators:
            if not op.applicable(state):
                continue
            expansions += 1
            succ = op.apply(state)
            new_path = path + (op.name,)
            if task.goal_reached(succ):
                key = new_path
                if key not in seen_paths:
                    seen_paths.add(key)
                    found.append(key)
                    # first solution fixes the length budget -- unless slack is None
                    # (unbounded/k-driven: enumerate the k shortest across all lengths)
                    if max_len is None and length_slack is not None:
                        max_len = len(new_path) + length_slack
                continue
            if max_len is None or len(new_path) < max_len:
                queue.append((succ, new_path))
    return found


def _bestfirst_diverse_plans(
    task, k, length_slack, max_expansions, search, heuristic_fn
):
    """Best-first diverse enumeration (a heuristic-guided stand-in for FORBID-SEARCH).

    Generalizes :func:`_bfs_diverse_plans` to an arbitrary node priority so the diverse-plan
    *ranking* can be studied: ``search="gbf"`` orders by ``h(node)`` alone, ``"astar"`` by
    ``g + h(node)``. ``heuristic_fn(node)`` reads ``node.state`` (a pyperplan SearchNode).

    Like the blind BFS this keeps **no closed set** -- revisiting a state via different action
    orderings is exactly what surfaces the distinct plans that make up the diverse set -- and
    relies on ``max_expansions`` + a ``k*4`` collection cap to terminate. Goals are recorded when
    popped, so the returned order is the frontier's discovery order (the ranking under study); the
    caller must NOT re-sort it by length. Returns ``(paths, stats)`` where ``paths`` is a list of
    op-name tuples and ``stats["expansions"]`` supports gbf-starvation reporting.

    Caveat: pure ``gbf`` (priority ``h`` only, no ``g`` pressure, no closed set) can churn among
    low-``h`` states and hit ``max_expansions`` without collecting ``k`` goals -- the caller flags
    that as starvation rather than hiding it.
    """
    import heapq

    from pyperplan.search import searchspace

    def priority(node):
        h = heuristic_fn(node)
        return h if search == "gbf" else node.g + h

    root = searchspace.make_root_node(task.initial_state)
    counter = 0
    heap = [(priority(root), counter, root)]
    counter += 1
    found: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    max_len: int | None = None
    expansions = 0

    while heap and expansions < max_expansions and len(found) < k * 4:
        _, _, node = heapq.heappop(heap)
        if task.goal_reached(node.state):
            key = tuple(node.extract_solution())  # op-name path root->goal
            if key not in seen_paths:
                seen_paths.add(key)
                found.append(key)
                if max_len is None and length_slack is not None:
                    max_len = (
                        node.g + length_slack
                    )  # first solution fixes the length budget
            continue  # goal is terminal; do not expand past it
        if max_len is not None and node.g >= max_len:
            continue
        for op in task.operators:
            if not op.applicable(node.state):
                continue
            expansions += 1
            child = searchspace.make_child_node(node, op.name, op.apply(node.state))
            if max_len is None or child.g <= max_len:
                heapq.heappush(heap, (priority(child), counter, child))
                counter += 1
    return found, {"expansions": expansions}


def _skeleton_from_op_names(op_names) -> Skeleton:
    # operator names look like "(pick green_block0 blue_table)"
    actions: list[Action] = []
    for name in op_names:
        toks = name.strip("()").split()
        actions.append(Action(toks[0], tuple(toks[1:])))
    return Skeleton(tuple(actions))


# --------------------------------------------------------------------------- #
# Auto-selection
# --------------------------------------------------------------------------- #
def make_planner(prefer: str = "symk", **kwargs) -> DiversePlanner:
    """Return the best available planner.

    ``prefer`` in {"symk", "pyperplan"}.
    """
    if prefer == "symk" and _symk_available():
        return SymKPlanner(**kwargs)
    return ForbidLoopPlanner(**kwargs)


def pyperplan_heuristic_factory(name: str):
    """Factory ``(task, problem) -> (node -> float)`` wrapping a pyperplan delete-
    relaxation heuristic (``hff``/``hadd``).

    Env-agnostic -- ignores ``problem`` geometry -- so it plugs
    into :class:`ForbidLoopPlanner`'s ``gbf``/``astar`` frontier for the off-the-shelf-heuristic
    arms of the DD2D heuristic experiment.
    """
    key = name.lower()

    def factory(task, _problem):
        from pyperplan.heuristics.relaxation import hAddHeuristic, hFFHeuristic

        if key == "hff":
            return hFFHeuristic(task)
        if key == "hadd":
            return hAddHeuristic(task)
        raise ValueError(
            f"unknown pyperplan heuristic {name!r}; choose 'hff' or 'hadd'"
        )

    return factory


def _symk_available() -> bool:
    try:
        import unified_planning  # noqa: F401
        import up_symk  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def enumerate_skeletons(
    problem: BlocksWorldProblem, k: int = 8, planner: DiversePlanner | None = None
) -> list[Skeleton]:
    """Convenience entry point used by the demo and tests."""
    planner = planner or make_planner()
    return planner.plan(problem, k)


# --------------------------------------------------------------------------- #
# symbolic validation (independent of any planner -- used by demo/tests)
# --------------------------------------------------------------------------- #
def symbolically_valid(problem: BlocksWorldProblem, skeleton: Skeleton) -> bool:
    """Apply the skeleton's symbolic effects to ℐ and check it reaches 𝒢.

    Reimplements the four STRIPS operators of ``domain/blocksworld.pddl`` so a plan can
    be validated without invoking a planner. Returns False on any inapplicable
    action or unmet goal.
    """
    on_table: dict[str, str] = {}
    on_block: dict[str, str] = {}
    clear: set[str] = set()
    holding: str | None = None
    for fact in problem.init_facts:
        if fact[0] == "on-table":
            on_table[fact[1]] = fact[2]
        elif fact[0] == "on-block":
            on_block[fact[1]] = fact[2]
        elif fact[0] == "clear":
            clear.add(fact[1])

    for a in skeleton.actions:
        op = a.name
        if op == "pick":
            b, t = a.args
            if holding is not None or on_table.get(b) != t or b not in clear:
                return False
            del on_table[b]
            clear.discard(b)
            holding = b
        elif op == "place":
            b, t = a.args
            if holding != b:
                return False
            on_table[b] = t
            clear.add(b)
            holding = None
        elif op == "unstack":
            b, lb = a.args
            if holding is not None or on_block.get(b) != lb or b not in clear:
                return False
            del on_block[b]
            clear.add(lb)
            clear.discard(b)
            holding = b
        elif op == "stack":
            b, lb = a.args
            if holding != b or lb not in clear:
                return False
            on_block[b] = lb
            clear.add(b)
            clear.discard(lb)
            holding = None
        else:
            return False

    if holding is not None:
        return False
    for fact in problem.goal_facts:
        if fact[0] == "on-table" and on_table.get(fact[1]) != fact[2]:
            return False
        if fact[0] == "on-block" and on_block.get(fact[1]) != fact[2]:
            return False
    return True
