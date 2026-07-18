"""DD2D instance generation -- docs/dd2d_spec.md.

Forward-generate-then-label (spec Section 8.1): synthesise a naturalistic drawer scene
(:mod:`blocks_tamp.dd2d.scene`), enumerate the geometric clearing candidates
(:mod:`blocks_tamp.dd2d.enumerate`), label every candidate under budget
(:mod:`blocks_tamp.dd2d.label`), then apply the three decision-relevance filters (F1 target
blocked, F2 a real choice of clearing subsets, F3 a solvability certificate). Instances
failing any filter are resampled. Optional generation-time certification runs the real
:class:`~blocks_tamp.dd2d.refine.DD2DRefiner`: the intended (first confidently-feasible)
staging skeleton must refine, and the degenerate ``retrieve(target)``-only skeleton must
NOT (the target is blocked) -- so the intended abstract difficulty is guaranteed by
construction, not hoped for.

An :class:`DD2DProblem` duck-types the surface ``blocks_tamp.record`` reads, so
``build_example`` / ``build_image_refs`` work unchanged. The difficulty here is *measured,
not installed*: the generator produces naturalistic scenes; the feasibility structure is a
property of the distribution (see notebook.md / docs/dd2d.md).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

from alphatamp.approaches.spectre.envs.dd2d.problem import ObjectInfo
from alphatamp.approaches.spectre.envs.dd2d.skeleton import Action, Skeleton

from .enumerate import Candidate, enumerate_candidates
from .label import decision_filters, label_all, min_feasible_subset_size
from .planning import staging_skeleton
from .refine import DD2DRefiner
from .world import DrawerScene  # re-exported

Literal = tuple

ITEM_HEIGHT = 6.0  # conceptual item height (spec P18); makes ObjectInfo.size a 3-tuple
_PALETTE = {"target": "tomato", "concave": "slateblue", "item": "silver"}


@dataclass
class DD2DProblem:
    """A drawer-decluttering TAMP instance (duck-types the record.py surface)."""

    problem_id: str
    objects: list[ObjectInfo]
    tables: list[str]  # region names: ["drawer", "buffer"]
    init_facts: list[Literal]
    goal_facts: list[Literal]
    scene: DrawerScene
    seed: int
    num_blocks: int  # = number of items (provenance parity with BlocksWorldProblem)
    num_blockers: int  # = number of non-target items
    target: str
    candidates: list[Candidate]
    problem_type: str = "drawer_declutter"
    lam: float = 1.0
    margin: float = 1.0
    crowd: int = 0
    diverse_crowd: bool = False  # collar drawn from all families, not just round ones
    min_feasible_subset: int | None = (
        None  # size of the smallest feasible clearing subset
    )
    intended: Skeleton | None = None

    @property
    def requires_subset(self) -> bool:
        """True iff no single-object removal is a feasible clearing plan (a 2+ blocker
        subset must be identified).

        See docs/dd2d.md 'Requiring a blocking subset'.
        """
        return self.min_feasible_subset is not None and self.min_feasible_subset >= 2

    # -- planner hook: use the DD2D domain, not blocksworld.pddl ----------------
    @property
    def domain_pddl_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "domain",
            "drawer_declutter.pddl",
        )

    # -- PDDL emission (geometry-blind; spec Section 6) ------------------------
    def to_pddl_problem(self, name: str | None = None) -> str:
        name = name or self.problem_id
        item_names = [o.name for o in self.objects]
        objs = "    " + " ".join(item_names) + " - item"

        def fact(lit: Literal) -> str:
            return f"({' '.join(str(a) for a in lit)})"

        init = "\n    ".join(fact(f) for f in self.init_facts)
        goal_parts = " ".join(fact(f) for f in self.goal_facts)
        return (
            f"(define (problem {name})\n"
            f"  (:domain drawer_declutter)\n"
            f"  (:objects\n{objs})\n"
            f"  (:init\n    {init})\n"
            f"  (:goal (and {goal_parts})))\n"
        )

    # -- certification / reference skeletons ----------------------------------
    def intended_skeleton(self) -> Skeleton:
        assert self.intended is not None
        return self.intended

    def retrieve_only_skeleton(self) -> Skeleton:
        """The degenerate "just grab it" plan (spec Section 6.2) -- always infeasible
        while the target is blocked (F1)."""
        return Skeleton((Action("retrieve", (self.target,)),))

    def a_infeasible_skeleton(self) -> Skeleton:
        return self.retrieve_only_skeleton()

    def feasible_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.meta.get("label") == "feasible"]


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _object_table(scene: DrawerScene) -> list[ObjectInfo]:
    out: list[ObjectInfo] = []
    for name, st in scene.items.items():
        w, h = st.shape.size
        if st.is_target:
            cat, color, blocker = "target", _PALETTE["target"], False
        else:
            cat, blocker = "item", True
            color = _PALETTE["concave"] if st.shape.concave else _PALETTE["item"]
        out.append(ObjectInfo(name, cat, color, (w, h, ITEM_HEIGHT), blocker, "drawer"))
    return out


def _facts(scene: DrawerScene) -> tuple[list[Literal], list[Literal]]:
    init: list[Literal] = [("handempty",), ("target", scene.target)]
    for name in scene.items:
        init.append(("in-drawer", name))
    goal: list[Literal] = [("extracted", scene.target)]
    return init, goal


def _build_problem(
    scene: DrawerScene,
    candidates: list[Candidate],
    seed: int,
    lam: float,
    crowd: int,
    diverse_crowd: bool = False,
) -> DD2DProblem:
    objects = _object_table(scene)
    init_facts, goal_facts = _facts(scene)
    n = len(scene.items)
    crowd_tag = (f"_c{crowd}" + ("dc" if diverse_crowd else "")) if crowd else ""
    cid = f"dd2d_n{n}_l{int(round(lam * 100))}" + crowd_tag + f"_s{seed}"
    return DD2DProblem(
        problem_id=cid,
        objects=objects,
        tables=["drawer", "buffer"],
        init_facts=init_facts,
        goal_facts=goal_facts,
        scene=scene,
        seed=seed,
        num_blocks=n,
        num_blockers=n - 1,
        target=scene.target,
        candidates=candidates,
        lam=lam,
        margin=scene.margin,
        crowd=crowd,
        diverse_crowd=diverse_crowd,
        min_feasible_subset=min_feasible_subset_size(candidates),
    )


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def generate_dd2d_problem(
    lam: float = 1.0,
    seed: int = 0,
    margin: float = 1.0,
    split: str = "train",
    n_items: int | None = None,
    crowd: int = 0,
    diverse_crowd: bool = False,
    require_subset: bool = False,
    min_subset: int = 2,
    unblocked_target: bool = False,
    certify: bool = True,
    budget: int | None = 300,
    retry_cap: int = 10,
    samples_per_step: int = 15,
    time_budget: float | None = None,
    max_resamples: int = 400,
) -> DD2DProblem:
    """Generate one filtered (and optionally certified) DD2D instance.

    Resamples scenes until F1 (target blocked) & F2 (>= 2 clearing subsets) & F3 (>= 1
    feasible candidate) all pass, then optionally certifies with the real refiner. Raises
    ``RuntimeError`` if nothing passes within ``max_resamples``. ``lam`` is the buffer
    scale (spec P4): smaller = tighter buffer = a stronger packing signal.

    ``crowd`` (default 0 = naturalistic) is the difficulty prior that pincers the target so
    that MANY problems require identifying a 2+ blocker **subset** rather than a single
    object (see docs/dd2d.md). ``diverse_crowd`` draws the collar from all families (not just
    round ones) so concave shapes join the pincer; it tends to lower the natural subset rate
    (looser ring), which ``require_subset`` compensates for by resampling. ``require_subset``
    additionally *guarantees* it: only keep instances whose smallest feasible clearing subset
    is >= ``min_subset`` (an optional F4 decision-relevance filter; off by default so the
    distribution stays a natural mix).

    The refiner budget knobs (``budget``/``retry_cap``/``samples_per_step``/``time_budget``,
    spec P13/P14/P15) are threaded into **generation-time certification** so every kept problem
    has >= 1 plan that refines *under the same budget you will collect with* (a tighter budget
    just means more resampling). See :class:`~blocks_tamp.dd2d.refine.DD2DRefiner`.
    """
    from .scene import generate_scene

    if n_items is not None and n_items + 0 > 14:
        warnings.warn(
            f"n_items={n_items} exceeds the spec's 9-14 range; SymK top-k may be slow.",
            stacklevel=2,
        )

    for attempt in range(max_resamples):
        scene_seed = (seed * 1_000_003 + attempt) & 0x7FFFFFFF
        scene = generate_scene(
            scene_seed,
            lam=lam,
            split=split,
            n_items=n_items,
            crowd=crowd,
            diverse_crowd=diverse_crowd,
        )
        scene.margin = margin
        candidates = enumerate_candidates(scene, seed=scene_seed)
        label_all(scene, candidates, seed=scene_seed)
        filt = decision_filters(scene, candidates)

        if (
            unblocked_target
        ):  # stratum 0: target directly graspable (retrieve-only feasible)
            if filt["F1"]:  # F1 True == target blocked -> reject; we want it OPEN
                continue
            problem = _build_problem(scene, candidates, seed, lam, crowd, diverse_crowd)
            problem.min_feasible_subset = 0  # no clearing needed
            problem.intended = problem.retrieve_only_skeleton()
            if certify:
                ref = DD2DRefiner(
                    budget=budget,
                    retry_cap=retry_cap,
                    samples_per_step=samples_per_step,
                    time_budget=time_budget,
                )
                if not ref.refine(
                    problem.retrieve_only_skeleton(), scene, seed=1
                ).feasible:
                    continue  # retrieve-only MUST succeed for stratum 0
            return problem

        if not (filt["F1"] and filt["F2"] and filt["F3"]):
            continue
        if require_subset:  # F4: no single-object removal is a feasible clearing plan
            mfs = min_feasible_subset_size(candidates)
            if mfs is None or mfs < min_subset:
                continue

        problem = _build_problem(scene, candidates, seed, lam, crowd, diverse_crowd)
        feas = problem.feasible_candidates()
        problem.intended = staging_skeleton(problem.target, feas[0].members)

        if certify:
            ref = DD2DRefiner(
                budget=budget,
                retry_cap=retry_cap,
                samples_per_step=samples_per_step,
                time_budget=time_budget,
            )
            if not ref.refine(problem.intended, scene, seed=1).feasible:
                continue  # intended must refine
            if ref.refine(problem.retrieve_only_skeleton(), scene, seed=2).feasible:
                continue  # "just grab it" must fail (target blocked)
        return problem

    raise RuntimeError(
        f"could not generate a DD2D instance for seed {seed} within {max_resamples} "
        f"resamples (lam={lam}, margin={margin}); loosen lam or the filters"
    )


def make_dd2d_problem(**kwargs) -> DD2DProblem:
    """Thin wrapper mirroring ``blocks_tamp.problem.make_problem`` dispatch."""
    return generate_dd2d_problem(**kwargs)
