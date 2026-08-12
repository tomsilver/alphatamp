"""DD2D geometric distance prior -- a *hand-written* search heuristic for the best-first
diverse enumerator (:func:`blocks_tamp.planning._bestfirst_diverse_plans`).

This is the deliberately-coarse "geometry" arm of the DD2D heuristic experiment
(see ``blocks_tamp/dd2d/heuristic_experiment.py``). It reads object *positions* -- so it
crosses the strict geometry-blind line the off-the-shelf ``hff``/``hadd`` arms respect -- but
it stays well short of a feasibility oracle: it knows nothing about grasp fingers, buffer
packing capacity, or the true minimum clearing-subset size. It is a soft spatial prior, a
weak stand-in for the geometric reasoning PIGINet learns.

Idea (spec / user framing): objects *near* the target are the likely blockers; objects far
away are likely distractors. A plan that has cleared the near objects is more promising. So we
score a symbolic state by the "proximity mass still in the drawer":

    h(state) = sum over non-target items still `(in-drawer o)` of  weight(dist(o, target))

with ``weight`` decreasing in distance, so **h falls as near/blocking items get staged** --
the natural minimize sign for gbf (priority ``h``) / astar (priority ``g + h``). We deliberately
do NOT use the precise grasp-blocker oracle in :mod:`blocks_tamp.dd2d.enumerate`
(``_blocker_sets`` / ``target_open_grasp``); that is the geometry oracle the ``candidates``
planner already uses, and the point of this arm is a crude prior, not the oracle.

State membership (verified against ``blocks_tamp/domain/drawer_declutter.pddl``): an item is
still in the drawer iff its ``(in-drawer <name>)`` fact is present (``pick`` deletes it); the
target itself stays in the drawer until ``retrieve``, so it is excluded from the sum. PDDL
object names equal ``scene.items`` keys / ``scene.target``, so we key geometry on the name
directly.
"""

from __future__ import annotations

import math

FORMS = ("inv", "avg", "radius")


def _in_drawer_nontarget(state, target: str) -> list[str]:
    """Names of non-target items still in the drawer, parsed from a pyperplan fact set
    (fact strings look like ``(in-drawer o3)``)."""
    out: list[str] = []
    for fact in state:
        toks = fact.strip("()").split()
        if len(toks) == 2 and toks[0] == "in-drawer" and toks[1] != target:
            out.append(toks[1])
    return out


def _target_distances(problem, use_edge: bool) -> dict[str, float]:
    """Per non-target item: distance to the target (centroid, or footprint-edge if ``use_edge``)."""
    scene = problem.scene
    target = problem.target
    tfp = scene.target_state().footprint()
    tx, ty = tfp.centroid.coords[0]
    dist: dict[str, float] = {}
    for name, st in scene.items.items():
        if name == target:
            continue
        if use_edge:
            dist[name] = st.footprint().distance(
                tfp
            )  # 0 if touching, as in enumerate._adjacent
        else:
            cx, cy = st.footprint().centroid.coords[0]
            dist[name] = math.hypot(cx - tx, cy - ty)
    return dist


def distance_heuristic_factory(
    form: str = "inv",
    eps: float = 1.0,
    radius_margin: float = 2.0,
    use_edge: bool = False,
):
    """Return a factory ``(task, problem) -> (node -> float)`` for the geometric
    distance prior.

    ``form`` selects the proximity weight (all monotone: clearing a near item lowers ``h``):

    * ``"inv"``   (default): ``h = Σ 1/(dist + eps)`` over remaining non-target items -- parameter-free
      (just ``eps``), the natural choice.
    * ``"avg"``   : ``h = -mean(dist)`` over remaining -- the user's original average-distance idea
      (higher average = the near items were removed, so negate to minimize).
    * ``"radius"``: ``h = Σ max(0, R - dist)`` with ``R = target.r_max + radius_margin`` -- closeness
      within a band around the target.

    ``use_edge`` swaps centroid distance for footprint edge distance (as ``enumerate._adjacent``).
    """
    if form not in FORMS:
        raise ValueError(f"unknown distance form {form!r}; choose from {FORMS}")

    def factory(task, problem):
        target = problem.target
        dist = _target_distances(problem, use_edge)
        if form == "inv":
            weight = {n: 1.0 / (d + eps) for n, d in dist.items()}
        elif form == "radius":
            R = problem.scene.target_state().shape.r_max + radius_margin
            weight = {n: max(0.0, R - d) for n, d in dist.items()}
        else:  # "avg" -- computed per-call from dist directly
            weight = None

        def h(node) -> float:
            remaining = _in_drawer_nontarget(node.state, target)
            if form == "avg":
                ds = [dist[n] for n in remaining if n in dist]
                return -sum(ds) / len(ds) if ds else 0.0
            return sum(weight[n] for n in remaining if n in weight)

        return h

    return factory
