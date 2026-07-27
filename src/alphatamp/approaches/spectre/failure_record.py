"""One canonical failure record, replacing v2.2's five bespoke fact types.

v2.2 harvested a per-environment fact vocabulary -- ``blocked-at-contents``,
``extraction-failed``, ``grasp-witness``, ``pack-exhausted``, ``pack-impossible`` -- each
with its own producer, its own tier, and its own place in the tensorizer. Adding an
environment meant adding fact types; two of the five were computed by *re-deriving DD2D
geometry* rather than observing the refiner.

v3 asks instead: when a refinement attempt fails, what is legally observable with **no
domain computation**? A refiner grounds steps of a plan until step *j* fails on some
continuous query. Instrumenting the computations that already ran yields exactly:

===================  ====================================================================
``step_index`` *j*   which step failed
``schema``/``args``  which query, on which objects
``unmoved``          ``U(sigma, j)``: objects the prefix had not moved when it failed
``culprits``         objects the failed samples actually collided with
``n_step``           sampler effort spent on this step
``exhausted``        the sampler ran out of its own retries (not the global budget)
``budget_exhausted`` the refiner stopped on time/call budget: **this proves nothing**
===================  ====================================================================

Every v2.2 fact type is a projection of this record (proposal §6.2): ``extraction-failed``
is (schema=pick, args=the blocker); ``pack-exhausted`` is (schema=place, exhausted);
``blocked-at-contents`` in its *observed* mode is (schema=retrieve, U=all∖staged);
``grasp-witness`` is the culprit slot. The two that are not projections --
``blocked-at-contents`` in *computed* mode and ``pack-impossible`` -- are counterfactual
analytic claims about attempts that never ran, and are deliberately dropped (R2, R3), not
casualties of the unification.

**Two provenances, one schema.** ``dd2d_v4`` carries instrumented records directly. Older
collections are **backfilled** from stored metadata, which recovers *j*, the query, the
args and *U* exactly, but cannot recover culprits or per-step effort -- so those fields are
absent rather than guessed, and ``effort_is_total`` marks that ``n_step`` is really the
whole-attempt stream count. A model trained on backfilled records has never seen the rich
fields populated, so the two must not be mixed within one checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.schema import EpisodeRecord

__all__ = ["FailureRecord", "records_for_episode", "records_for_candidate"]


@dataclass(frozen=True)
class FailureRecord:
    """One observed refinement failure."""

    candidate_idx: int
    step_index: int
    schema: str
    args: tuple[str, ...] = ()
    culprits: tuple[str, ...] = ()
    unmoved: frozenset[str] = frozenset()
    n_step: int = 0
    exhausted: bool = True
    budget_exhausted: bool = False
    effort_is_total: bool = False
    """``n_step`` is the whole-attempt stream count, not this step's (backfill only).

    Kept explicit because the instrumented collection redefines the same field: a model
    trained on one and deployed on the other would read a silently different scalar.
    """

    instrumented: bool = False
    """True when the refiner emitted this directly; False when backfilled from metadata."""

    def proves_failure(self) -> bool:
        """Whether this record witnesses a query that actually ran to exhaustion.

        A budget exit proves nothing: the refiner reports the deepest step it *reached*,
        which on a timeout was never tested. Trusting that is what made one dd2d_v2
        candidate demote 12 genuinely-feasible plans.
        """
        return self.exhausted and not self.budget_exhausted


def _from_instrumented(
    candidate_idx: int, obs: dict, all_objects: frozenset[str]
) -> FailureRecord:
    """Build from a v3-collection ``refiner_metadata['failures']`` entry."""
    unmoved = frozenset(obs.get("unmoved") or ())
    return FailureRecord(
        candidate_idx=candidate_idx,
        step_index=int(obs["step_index"]),
        schema=str(obs["schema"]),
        args=tuple(obs.get("args") or ()),
        culprits=tuple(obs.get("culprits") or ()),
        unmoved=unmoved or all_objects,
        n_step=int(obs.get("n_step") or 0),
        exhausted=bool(obs.get("exhausted", True)),
        budget_exhausted=bool(obs.get("budget_exhausted", False)),
        instrumented=True,
    )


def _backfilled(
    candidate_idx: int,
    outcome,
    skeleton,
    all_objects: frozenset[str],
    spec: DomainSpec,
    goal_objs: frozenset[str],
) -> Optional[FailureRecord]:
    """Reconstruct a record from a pre-v3 collection's stored metadata.

    ``steps_bound`` is the failing step index -- verified equal to the index of
    ``failure_action`` in the plan on 2528/2528 sampled candidates -- so *j*, the query,
    its args and ``U(sigma, j)`` all come back exactly. Culprits and per-step effort do
    not exist in the stored data and are left empty rather than approximated.
    """
    from alphatamp.approaches.spectre.domain import unmoved as unmoved_fn

    meta = outcome.refiner_metadata or {}
    action = str(meta.get("failure_action") or "")
    if not action:
        return None
    schema = action.split("(", 1)[0]
    inside = action[action.find("(") + 1 : action.rfind(")")] if "(" in action else ""
    args = tuple(a.strip() for a in inside.split(",") if a.strip())
    j = int(meta.get("steps_bound", 0))
    n_total = int(meta.get("n_attempts") or 0)
    # Exactness witness: if the whole attempt cost exactly the minimum possible number of
    # sampler calls, nothing was re-sampled, so every query it reports genuinely ran. That
    # is derivable from stored metadata and is sound; where the count exceeds the minimum
    # the attempt re-sampled and may have stopped on a budget, so we claim nothing. A
    # domain that declares no cost model gets `False` -- no evidence, not assumed evidence.
    floor = spec.min_calls(skeleton)
    return FailureRecord(
        candidate_idx=candidate_idx,
        step_index=j,
        schema=schema,
        args=args,
        culprits=(),  # never recorded pre-v3; absent, not guessed
        unmoved=unmoved_fn(skeleton, j, all_objects, goal_objs),
        n_step=n_total,
        exhausted=(floor is not None and n_total == floor),
        budget_exhausted=False,
        effort_is_total=True,
        instrumented=False,
    )


def records_for_candidate(
    episode: EpisodeRecord,
    candidate_idx: int,
    spec: Optional[DomainSpec] = None,
) -> list[FailureRecord]:
    """Every failure observed while refining one candidate (empty if it succeeded)."""
    spec = spec or spec_for(episode.provenance.env_variant)
    outcome = episode.outcomes[candidate_idx]
    if outcome.outcome != "fail":
        return []
    all_objects = frozenset(episode.object_registry)
    meta = outcome.refiner_metadata or {}
    # `refiner_metadata` is a free-form dict, so the instrumented payload is validated
    # rather than trusted: a malformed entry should be skipped, not crash a rollout.
    observations = meta.get("failures")
    if isinstance(observations, (list, tuple)) and observations:
        return [
            _from_instrumented(candidate_idx, o, all_objects)
            for o in observations
            if isinstance(o, dict) and "schema" in o and "step_index" in o
        ]
    rec = _backfilled(
        candidate_idx,
        outcome,
        episode.skeleton_pool[candidate_idx],
        all_objects,
        spec,
        spec.goal_objects(episode),
    )
    return [rec] if rec is not None else []


def records_for_episode(
    episode: EpisodeRecord,
    candidate_indices=None,
    spec: Optional[DomainSpec] = None,
) -> list[FailureRecord]:
    """Flatten the failure records of the given candidates (default: all failures)."""
    spec = spec or spec_for(episode.provenance.env_variant)
    if candidate_indices is None:
        candidate_indices = [
            i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"
        ]
    out: list[FailureRecord] = []
    for idx in candidate_indices:
        out.extend(records_for_candidate(episode, idx, spec))
    return out
