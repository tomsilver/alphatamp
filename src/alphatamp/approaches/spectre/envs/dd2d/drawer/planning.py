"""DD2D diverse task-plan enumeration -- candidate orderings (spec Section 7 / 10.2).

Why DD2D needs its own enumerator (not generic SymK/pyperplan top-k). The clearing
decision is *geometric* -- which items block the target's grasp fingers -- and that
constraint is deliberately dropped from the symbolic model (spec Section 6.1); the
shortest optimistic plan is literally ``retrieve(target)`` (spec Section 6.2), which
fails geometrically, after which a goal-directed planner just grows plans that stage
*arbitrary* items. The blocking subsets never surface within k. So, exactly as PIGINet
obtains obstacle-moving skeletons by replanning-with-obstacle-removal, we enumerate the
clearing subsets up front (:mod:`blocks_tamp.dd2d.enumerate`) and turn each into a
staging skeleton ``[pick(o); place-buffer(o) ...] ++ retrieve(target)``.

:class:`DD2DPlanner` differs only in the candidate **ordering/selection policy** -- the
Tier-2 variable under study (spec Section 10.2): ``published`` (ascending |S|, the weak
size heuristic), ``random`` (the uninformed floor), ``slack`` (ascending buffer slack
ratio, the strongest cheap ordering), ``oracle`` (label-first upper bracket). Conforms to
``blocks_tamp.planning.DiversePlanner`` so the demo/record/refiner reuse is unchanged.
"""

from __future__ import annotations

import random

from alphatamp.approaches.spectre.envs.dd2d.planning import DiversePlanner
from alphatamp.approaches.spectre.envs.dd2d.skeleton import Action, Skeleton

ORDERS = ("published", "random", "slack", "oracle")


def staging_skeleton(target: str, members: list[str]) -> Skeleton:
    """``[pick(o); place-buffer(o) for o in members] ++ retrieve(target)``."""
    acts: list[Action] = []
    for o in members:
        acts.append(Action("pick", (o,)))
        acts.append(Action("place-buffer", (o,)))
    acts.append(Action("retrieve", (target,)))
    return Skeleton(tuple(acts))


class DD2DPlanner(DiversePlanner):
    name = "dd2d-candidates"

    def __init__(self, order: str = "published") -> None:
        if order not in ORDERS:
            raise ValueError(f"unknown order {order!r}; choose from {ORDERS}")
        self.order = order

    def plan(self, problem, k: int) -> list[Skeleton]:
        cands = list(problem.candidates)
        ordered = self._reorder(cands, problem)
        skeletons: list[Skeleton] = []
        seen: set = set()
        for c in ordered:
            sk = staging_skeleton(problem.target, c.members)
            key = sk.key()
            if key not in seen:
                seen.add(key)
                skeletons.append(sk)
            if len(skeletons) >= k:
                break
        return skeletons

    def _reorder(self, cands: list, problem) -> list:
        if self.order == "published":
            return cands  # already ascending |S| with seeded ties (spec Section 7)
        if self.order == "random":
            rng = random.Random((problem.seed * 6364136223846793005 + 1) & 0xFFFFFFFF)
            out = list(cands)
            rng.shuffle(out)
            return out
        if self.order == "slack":
            buf_area = problem.scene.buffer.area
            return sorted(cands, key=lambda c: (_slack(problem, c) / buf_area, c.size))
        if self.order == "oracle":
            # confidently-feasible candidates first (the upper bracket), then published
            return sorted(
                cands, key=lambda c: (0 if c.meta.get("label") == "feasible" else 1,)
            )
        return cands  # pragma: no cover


def _slack(problem, cand) -> float:
    return sum(problem.scene.items[n].shape.area for n in cand.subset)


def _resolve_heuristic(heuristic):
    """Map a heuristic name -> (factory, tag) for the pyperplan gbf/astar frontier.

    ``hff``/``hadd`` are off-the-shelf pyperplan delete-relaxation heuristics (geometry-blind);
    ``dist`` (and the ``dist-avg`` / ``dist-radius`` variants) are the hand-written DD2D geometric
    distance prior (:mod:`blocks_tamp.dd2d.heuristics`).
    """
    from alphatamp.approaches.spectre.envs.dd2d.planning import (
        pyperplan_heuristic_factory,
    )

    h = (heuristic or "").lower()
    if h in ("hff", "hadd"):
        return pyperplan_heuristic_factory(h), h
    if h in ("dist", "distance", "dist-inv"):
        from .heuristics import distance_heuristic_factory

        return distance_heuristic_factory("inv"), "dist"
    if h == "dist-avg":
        from .heuristics import distance_heuristic_factory

        return distance_heuristic_factory("avg"), "distavg"
    if h == "dist-radius":
        from .heuristics import distance_heuristic_factory

        return distance_heuristic_factory("radius"), "distrad"
    raise ValueError(
        f"search != 'bfs' needs heuristic in {{hff, hadd, dist, dist-avg, dist-radius}}, got {heuristic!r}"
    )


def make_dd2d_planner(
    prefer: str = "candidates",
    order: str = "published",
    search: str = "bfs",
    heuristic=None,
    **kwargs,
) -> DiversePlanner:
    """``prefer='candidates'`` -> the DD2D candidate enumerator (recommended), ordered
    by ``order``; ``'symk'``/``'pyperplan'`` -> the generic, geometry-blind diverse
    planner over the DD2D domain -- the **fair research baseline** (docs/dd2d.md "Fair
    baselines").

    For ``'pyperplan'`` we default ``length_slack=None`` (unbounded, k-driven), so it
    enumerates the ``k`` globally-shortest diverse plans ascending-length rather than being
    capped at ``shortest+2`` (which on DD2D is single-object stagings only). It must then
    refine deep enough to reach the feasible plans a geometry-informed planner finds
    directly -- so a subset-required instance needs a large ``k`` (~``n_blockers^subset``),
    and that cost gap vs. ``candidates`` is the intended baseline comparison. Pass an explicit
    ``length_slack=<int>`` to cap plan length.

    ``search``/``heuristic`` (pyperplan only) select the enumeration frontier for the DD2D
    *heuristic experiment*: ``search='bfs'`` (default) is the blind ascending-length baseline;
    ``'gbf'``/``'astar'`` order the diverse set by ``heuristic`` -- ``'hff'``/``'hadd'``
    (off-the-shelf, geometry-blind) or ``'dist'`` (hand-written geometric prior). See
    ``blocks_tamp/dd2d/heuristic_experiment.py``.
    """
    if prefer in ("symk", "pyperplan"):
        from alphatamp.approaches.spectre.envs.dd2d.planning import make_planner

        if prefer == "pyperplan":
            kwargs.setdefault("length_slack", None)  # unbounded k-driven fair baseline
            if search != "bfs":
                factory, tag = _resolve_heuristic(heuristic)
                kwargs.update(
                    search=search, heuristic_factory=factory, heuristic_tag=tag
                )
        return make_planner(prefer=prefer, **kwargs)
    return DD2DPlanner(order=order)
