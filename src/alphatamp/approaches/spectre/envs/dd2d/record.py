"""PIGINet training-example record schema + construction.

One :class:`PIGINetExample` is exactly the tuple the PIGINet paper feeds its
feasibility predictor f(I, pi, G) (paper lines 67-76, 197, 223):

    objects O, initial literals I, goal literals G, a task plan pi (the skeleton
    with continuous args omitted -- paper Table II), per-object segmented images,
    and a noisy feasibility label (positive iff refinement found a binding).

This is the artefact a re-implemented PIGINet will consume. Field-by-field mapping
to the original ``fastamp`` contract (``get_facts_goals_visuals`` and the ``PVT``
checker in ``pybullet_planning/pigi_tools/feasibility_checkers.py``) is documented
in ``docs/piginet_record_schema.md``.

Image *pixels* are deferred (see ``rendering.py``), but the image schema is fully
present: each entry names an object, a viewpoint, an optional file path, a
segmentation id, and a bounding box, so wiring real crops later is mechanical.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .problem import SortingProblem
from .refine import RefineResult
from .skeleton import Skeleton

SCHEMA_VERSION = "1.0"


@dataclass
class ImageRef:
    """A reference to one per-object segmented image (pixels optional)."""

    object: str
    view: str  # "topdown" | "oblique" | ...
    seg_id: int | None = None  # segmentation id within the rendered frame
    bbox: list[int] | None = None  # [row_min, col_min, row_max, col_max]
    path: str | None = None  # file path once pixels are rendered; None = deferred


@dataclass
class PIGINetExample:
    problem_id: str
    objects: list[dict]  # {name, category, color, size, is_blocker, start_table}
    init_literals: list[list[str]]  # I
    goal_literals: list[list[str]]  # G
    task_plan: list[list[str]]  # pi: [operator, *args] per step, no continuous args
    label: bool  # feasibility (positive iff refined within budget)
    label_source: str  # how the label was decided
    refine: dict  # refinement diagnostics
    images: list[dict] = field(default_factory=list)  # list[ImageRef-as-dict]
    provenance: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # -- serialisation -------------------------------------------------------
    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "PIGINetExample":
        return cls(**json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "PIGINetExample":
        with open(path) as f:
            return cls.from_json(f.read())


# --------------------------------------------------------------------------- #
# literal / object extraction
# --------------------------------------------------------------------------- #
def extract_init_literals(problem: SortingProblem) -> list[list[str]]:
    """Initial literals I as a list of [predicate, *args]."""
    return [list(fact) for fact in problem.init_facts]


def extract_goal_literals(problem: SortingProblem) -> list[list[str]]:
    """Goal literals G as a list of [predicate, *args]."""
    return [list(fact) for fact in problem.goal_facts]


def object_table(problem: SortingProblem) -> list[dict]:
    return [
        {
            "name": o.name,
            "category": o.category,
            "color": o.color,
            "size": list(o.size),
            "is_blocker": o.is_blocker,
            "start_table": o.start_table,
        }
        for o in problem.objects
    ]


def build_image_refs(
    problem: SortingProblem, render=None, views=("topdown",)
) -> list[ImageRef]:
    """One :class:`ImageRef` per object per view.

    With a RenderResult, fill in seg ids + bounding boxes; otherwise emit schema-only
    refs (deferred pixels).
    """
    import numpy as np

    refs: list[ImageRef] = []
    name_to_segid: dict[str, int] = {}
    bbox_by_id: dict[int, list[int]] = {}
    if render is not None:
        name_to_segid = {v: k for k, v in render.id_to_name.items()}
        for sid in render.segment_ids():
            ys, xs = np.where(render.seg == sid)
            if len(ys):
                bbox_by_id[sid] = [
                    int(ys.min()),
                    int(xs.min()),
                    int(ys.max()),
                    int(xs.max()),
                ]
    for view in views:
        for o in problem.objects:
            sid = name_to_segid.get(o.name)
            refs.append(
                ImageRef(
                    object=o.name,
                    view=view,
                    seg_id=sid,
                    bbox=bbox_by_id.get(sid) if sid is not None else None,
                    path=None,  # pixels deferred; rendering is confirmed doable
                )
            )
    return refs


# --------------------------------------------------------------------------- #
# example construction
# --------------------------------------------------------------------------- #
def _refine_dict(refine_result: RefineResult) -> dict[str, Any]:
    """The persisted refinement diagnostics.

    The first five keys are the pre-v3 payload, unchanged and in order, so a v3 record
    is a strict superset of a v2/v3-collection record and every existing reader keeps
    working. The rest is the v3 instrumentation (observation-only, see
    ``dd2d/refine.py``): ``elapsed`` closes the long-standing gap where DD2D could not
    report the proposal's wall-clock metric at all (``refinement_wall_clock_s`` was
    hardcoded 0.0 because this dict dropped it), and ``failures`` carries the typed
    observations the v3 adaptive pathway consumes.

    ``budget_exhausted`` is load-bearing rather than diagnostic: ``failure_action`` names
    the deepest step *reached*, which on a budget exit was never tested. A consumer that
    promotes such a failure to proof tier is unsound -- that is what made one dd2d_v2
    candidate demote 12 genuinely-feasible plans.
    """
    out: dict[str, Any] = {
        "status": refine_result.status,
        "steps_bound": refine_result.steps_bound,
        "plan_length": refine_result.plan_length,
        "n_attempts": refine_result.n_attempts,
        "failure_action": refine_result.failure_action,
    }
    out["elapsed"] = round(refine_result.elapsed, 6)
    out["n_backjumps"] = refine_result.n_backjumps
    out["budget_exhausted"] = refine_result.budget_exhausted
    out["failures"] = [
        {
            "step_index": f.step_index,
            "schema": f.schema,
            "args": list(f.args),
            "culprits": list(f.culprits),
            "unmoved": list(f.unmoved),
            "n_step": f.n_step,
            "exhausted": f.exhausted,
            "budget_exhausted": f.budget_exhausted,
        }
        for f in refine_result.failures
    ]
    return out


def build_example(
    problem: SortingProblem,
    skeleton: Skeleton,
    refine_result: RefineResult,
    planner_name: str,
    images: list[ImageRef] | None = None,
    extra_provenance: dict[str, Any] | None = None,
    label_source: str = "refine_timeout",
) -> PIGINetExample:
    provenance = {
        "planner": planner_name,
        "seed": problem.seed,
        "num_blocks": problem.num_blocks,
        "num_blockers": problem.num_blockers,
        "problem_type": getattr(problem, "problem_type", "sorting"),
        "generator": f"blocks_tamp.problem.make_problem({getattr(problem, 'problem_type', 'sorting')!r})",
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    return PIGINetExample(
        problem_id=problem.problem_id,
        objects=object_table(problem),
        init_literals=extract_init_literals(problem),
        goal_literals=extract_goal_literals(problem),
        task_plan=skeleton.to_tokens_as_lists(),
        label=refine_result.feasible,
        label_source=label_source,  # how feasibility was decided (refiner-specific)
        refine=_refine_dict(refine_result),
        images=[img.__dict__ for img in (images or [])],
        provenance=provenance,
    )
