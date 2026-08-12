"""The domain contract -- everything v3 is allowed to know about an environment.

v2.2 called itself domain-agnostic while reaching for DD2D operator names in eleven
places: ``place-buffer`` to decide which objects a candidate manipulates and how long
its plan is, ``retrieve`` to decide whether a failure licenses demotion. Each is a
small, reasonable-looking literal, and together they mean porting to a second
environment is a search-and-replace through the tensorizer and the rollout rather than a
declaration.

v3 replaces them with one :class:`DomainSpec` per environment, whose only *required*
content is a per-query-type axiom declaration -- about one bit each. Everything else has
a domain-independent default that reads the operator schema the planner already
consumes.

**Why axioms are specification, not inference.** Proof-demotion is sound in DD2D because
of two properties (``docs/SPECTRE_v3_proposal.md`` L6): **monotonicity** (a query that
failed against some occupancy fails against any larger occupancy -- universal in
collision-based feasibility) and **locality** (objects the prefix moved have left the
query's relevant region -- a property of the world layout, which fails in e.g. same-
surface declutter). Declaring these has the same epistemic status as writing the PDDL
domain file. Getting them *wrong* costs attempts but cannot lose a feasible plan,
because demotion only ever reorders (P-E). With an empty registry nothing is promoted to
proof tier, every failure flows through the learned pathway, and the ranker still works
-- "learning is the floor".

**Exactness is split deliberately.** :attr:`QueryAxioms.exact` says "when this query
runs, it runs to exhaustion" -- a statement about the query type. Whether it actually
ran is a property of the *observation* (the refiner's ``exhausted`` /
``budget_exhausted`` flags), checked per record at demotion time. Conflating the two is
what made v2.2 unsound: a budget exit still reported ``retrieve(target)`` as the failing
action even though the retrieve was never tested, and one such candidate demoted 12
genuinely-feasible plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from alphatamp.approaches.spectre.schema import EpisodeRecord, SkeletonRecord

__all__ = [
    "QueryAxioms",
    "DomainSpec",
    "goal_objects",
    "manipulated",
    "length_key",
    "unmoved",
    "failure_schema",
    "spec_for",
    "DOMAINS",
    "EMPTY_SPEC",
]


@dataclass(frozen=True)
class QueryAxioms:
    """What a domain declares about one continuous-query (operator) schema.

    All three default to ``False``: an undeclared query is hint-tier, which is the safe
    direction -- it costs a missed pruning opportunity, never a wrong one.
    """

    monotone: bool = False
    """Failure against occupancy ``O`` implies failure against any ``O' superset O``."""

    local: bool = False
    """Objects moved by the plan prefix leave the query's relevant region."""

    exact: bool = False
    """A *completed* run of this query is exhaustive, not a sampled approximation."""

    def proof_tier(self) -> bool:
        """Whether a *completed, exhausted* failure of this query licenses demotion.

        The instance-level guard (did the query actually run to exhaustion?) is applied
        separately, against the observation.
        """
        return self.monotone and self.local and self.exact


def goal_objects(episode: EpisodeRecord) -> frozenset[str]:
    """Objects named by the goal.

    Computable in any PDDL problem.
    """
    return frozenset(o.name for atom in episode.goal_atoms for o in atom.objects)


def manipulated(skeleton: SkeletonRecord, goal_objs: frozenset[str]) -> frozenset[str]:
    """Objects the skeleton acts on: every object argument, minus goal-role objects.

    Goal objects are excluded because they appear in *every* candidate (DD2D's target is
    retrieved by all of them), so including them would add a constant to every set --
    harmless for the subset relations, but it would make the necessity head spend a
    logit predicting a label that is always 1 and already stated by ``obj_is_goal``.

    Verified equal to DD2D's hand-written ``place-buffer`` filter on **120000/120000**
    dd2d_v3 skeletons, so replacing the literal is a proof, not a hope.
    """
    args = frozenset(p.name for op in skeleton.operator_seq for p in op.parameters)
    return args - goal_objs


def length_key(skeleton: SkeletonRecord) -> int:
    """Bucket key for the within-length PL loss: the plan's operator count.

    Universal (every TAMP plan has a length) and, on DD2D, exactly equivalent to the
    ``-removals/max`` column v2.2 bucketed on: ``len(operator_seq) == 2*|staged| + 1``
    on **120000/120000** skeletons, so the induced partition is identical. Using the raw
    count rather than a normalized float also removes the ``round(key*1000)`` collision
    hazard in the loss.
    """
    return len(skeleton.operator_seq)


def unmoved(
    skeleton: SkeletonRecord,
    step_index: int,
    all_objects: frozenset[str],
    goal_objs: frozenset[str],
) -> frozenset[str]:
    """``U(sigma, j)`` -- objects untouched by the prefix before ``step_index``.

    The certificate rule compares these sets across candidates. On DD2D this reduces
    exactly to v2.2's staged-subset rule: at the retrieve step the prefix has staged
    ``S``, so ``U = all - S``, and ``U' superset-eq U`` iff ``S' subset-eq S``.
    """
    prefix = frozenset(
        p.name for op in skeleton.operator_seq[:step_index] for p in op.parameters
    )
    return all_objects - (prefix - goal_objs)


def failure_schema(outcome) -> Optional[str]:
    """The operator schema of a failed refinement, from stored metadata.

    Reads ``failure_action`` (e.g. ``"pick(o3)"``) and returns the schema name.
    **Callers must not treat this as proof that the query ran**: on a budget exit the
    refiner still names the deepest step it reached. v3 records carry an explicit
    ``budget_exhausted`` marker for exactly this reason; this helper exists for pre-v3
    collections and for the hint pathway, where a noisy signal is acceptable.
    """
    action = str(
        (getattr(outcome, "refiner_metadata", None) or {}).get("failure_action", "")
    )
    if not action:
        return None
    return action.split("(", 1)[0] or None


@dataclass(frozen=True)
class DomainSpec:
    """One environment's contract.

    Only ``axioms`` is environment-specific.
    """

    axioms: Mapping[str, QueryAxioms] = field(default_factory=dict)
    goal_objects: Callable[[EpisodeRecord], frozenset[str]] = goal_objects
    manipulated: Callable[[SkeletonRecord, frozenset[str]], frozenset[str]] = (
        manipulated
    )
    length_key: Callable[[SkeletonRecord], int] = length_key

    min_calls_per_schema: Mapping[str, int] = field(default_factory=dict)
    """Minimum sampler calls one grounding of each operator schema can cost.

    Optional, and it buys one specific thing: **exactness evidence for collections that
    predate refiner instrumentation**. If a whole attempt reports exactly the minimum
    possible call count, then nothing was re-sampled, so every query it reports really
    did run -- a sound witness recoverable from stored metadata alone. Where the count
    exceeds the minimum, the attempt re-sampled and may have stopped on a budget, so the
    witness correctly declines to fire.

    Declaring it is a cost-model statement of the same epistemic class as the axioms,
    not an inference routine. Undeclared (the default) simply means pre-instrumentation
    records cannot reach proof tier in strict mode.
    """

    def min_calls(self, skeleton: SkeletonRecord) -> Optional[int]:
        """Minimum possible sampler calls for a straight-through grounding of a plan."""
        if not self.min_calls_per_schema:
            return None
        total = 0
        for op in skeleton.operator_seq:
            if op.name not in self.min_calls_per_schema:
                return None  # an undeclared step makes the total unknowable, not zero
            total += self.min_calls_per_schema[op.name]
        return total

    def axioms_for(self, schema: Optional[str]) -> QueryAxioms:
        """Declared axioms for a query schema; the safe all-``False`` default if the
        domain said nothing about it."""
        if schema is None:
            return QueryAxioms()
        return self.axioms.get(schema, QueryAxioms())

    def subsets(self, episode: EpisodeRecord) -> list[frozenset[str]]:
        """Per candidate, the set of objects it manipulates."""
        goal_objs = self.goal_objects(episode)
        return [self.manipulated(s, goal_objs) for s in episode.skeleton_pool]

    def licenses_demotion(self, outcome) -> bool:
        """Whether this *failed* candidate's failure licenses sound proof-demotion.

        Two conditions, and both are needed. The **domain** must declare the failing
        query monotone + local + exact; and the **observation** must show the query
        actually ran to exhaustion rather than the refiner stopping on its budget.

        Pre-v3 records carry no exhaustion evidence. They are accepted here (v2.2
        semantics) because rejecting them would silently disable demotion on every
        existing collection; ``proof_demotion_v3`` exposes a strict mode that requires
        positive evidence, and the two modes are what let the v3 rule be checked against
        v2.2's decisions candidate-for-candidate before it is allowed to differ.
        """
        if not self.axioms_for(failure_schema(outcome)).proof_tier():
            return False
        meta = getattr(outcome, "refiner_metadata", None) or {}
        return not bool(meta.get("budget_exhausted", False))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

#: DD2D. The entire environment-specific content of the v3 contract: three lines saying
#: which query yields a sound deduction, and no geometry whatsoever.
#:
#: ``retrieve`` is monotone (a target ungraspable amid some drawer contents stays
#: ungraspable amid a superset), local (staged objects are physically out of the drawer,
#: so they cannot affect the grasp), and exact (``has_grasp`` enumerates every grasp
#: cell rather than sampling). ``pick`` and ``place-buffer`` are sampled, so a failure
#: is evidence, not proof -- they stay hint-tier and flow through the learned pathway.
#: ``min_calls_per_schema`` reads off the refiner's own loop: ``pick`` and ``retrieve``
#: each run one grasp test; ``place-buffer`` costs one pose sample plus one
#: accessibility test. So a straight-through grounding of ``[pick, place-buffer] * n
#: ++ retrieve`` costs
#: exactly ``3n + 1`` calls -- measured to hold for 85.76% of dd2d_v3 retrieve failures,
#: which are therefore provably un-resampled.
_DD2D = DomainSpec(
    axioms={
        "retrieve": QueryAxioms(monotone=True, local=True, exact=True),
        "pick": QueryAxioms(),
        "place-buffer": QueryAxioms(),
    },
    min_calls_per_schema={"pick": 1, "place-buffer": 2, "retrieve": 1},
)

#: An environment that declares nothing. Everything is hint-tier and the ranker must
#: learn from evidence alone -- the "learning is the floor" control.
EMPTY_SPEC = DomainSpec()

DOMAINS: dict[str, DomainSpec] = {
    "dd2d_v2": _DD2D,
    "dd2d_v3": _DD2D,
    "dd2d_v4": _DD2D,
    # Held-out generalization sets (docs/decisions 2026-08-01): same DD2D domain, unseen
    # item counts / new shape figures. They share the dd2d_v4 operator/predicate/type
    # contract, so they resolve to the same spec (a shape family is geometry metadata,
    # not a new schema).
    "dd2d_v4gen_count": _DD2D,
    "dd2d_v4gen_shape": _DD2D,
    # Shape-only generalization set (docs/decisions 2026-08-04): new tee/cross figures
    # at the TRAINED 9-12 blocker count, isolating the shape variable from count.
    "dd2d_v4gen_shapeonly": _DD2D,
    # Shape-size sweep + inference-time geometry interventions (docs/decisions
    # 2026-08-06): the physically-shrunk tee/cross collection, and the input-rewrites of
    # the shape-only episodes (tee/cross area->hull, boundary->hull). All the same DD2D
    # domain -- a shape family / rescale / hull-rewrite is geometry metadata, not a new
    # schema, so they resolve to the same spec and reuse the dd2d_v4 vocab (no OOV).
    "dd2d_v4gen_shapeonly_sz07": _DD2D,
    "dd2d_v4gen_shapeonly_hullarea": _DD2D,
    "dd2d_v4gen_shapeonly_hullshape": _DD2D,
    # x0.7 boundary shrink, input-only (fixed labels)
    "dd2d_v4gen_shapeonly_scale07": _DD2D,
    # fresh un-shrunk control (collection-variance bound)
    "dd2d_v4gen_shapeonly_fresh": _DD2D,
}


def spec_for(env_variant: str) -> DomainSpec:
    """The domain contract for an env variant.

    Unknown variants get :data:`EMPTY_SPEC` rather than raising: a new environment must
    be *runnable* before it is declared, and the degraded mode is the honest default
    (hints only, no proofs).
    """
    return DOMAINS.get(env_variant, EMPTY_SPEC)
