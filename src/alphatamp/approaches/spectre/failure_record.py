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
``state_delta``      ``s_j`` relative to ``s_0``: which atoms the prefix added / deleted
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

from dataclasses import dataclass, replace
from typing import Optional

from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.schema import EpisodeRecord, SkeletonRecord
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

__all__ = [
    "FailureRecord",
    "StateDelta",
    "records_for_episode",
    "records_for_candidate",
]

#: One atom, as ``(predicate name, argument names)``.
DeltaAtom = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class StateDelta:
    """``s_j`` relative to ``s_0``: which atoms the prefix added, which it deleted.

    The proposal's §6.1 field is the abstract state itself; what is carried is the
    *delta*, because ``s_0`` already reaches the scorer through the scene tokens and the
    delta is the part a record actually contributes. It is pure STRIPS progression over
    the candidate's own operator sequence -- no geometry, no domain computation.

    Atoms are ``(predicate, args)`` **name** pairs rather than ``GroundAtom``s, so a
    record stays cheap and its object names live in the same namespace as ``args`` /
    ``culprits`` / ``unmoved``. Both tuples are **sorted**, so any downstream truncation
    is deterministic rather than a function of set iteration order.
    """

    added: tuple[DeltaAtom, ...] = ()
    deleted: tuple[DeltaAtom, ...] = ()

    def is_empty(self) -> bool:
        """``s_j == s_0`` -- true exactly when the prefix is empty (``j == 0``)."""
        return not self.added and not self.deleted


def _atom_names(atoms) -> set[DeltaAtom]:
    return {(a.predicate.name, tuple(e.name for e in a.entities)) for a in atoms}


def _state_deltas(
    episode: EpisodeRecord, skeleton: SkeletonRecord, step_indices: set[int]
) -> dict[int, StateDelta]:
    """``s_j - s_0`` and ``s_0 - s_j`` for each requested prefix length ``j``.

    One progression per *candidate*, not per record: a candidate's records share a plan,
    so the trajectory is computed once and indexed. ``verify_preconditions=False`` is
    defensive rather than permissive -- every dd2d_v4 skeleton verifies, but a deployed
    rollout must not raise on a malformed one.
    """
    traj = reconstruct_trajectory(
        episode.initial_abstract_state,
        skeleton.operator_seq,
        verify_preconditions=False,
    )
    s0 = _atom_names(episode.initial_abstract_state.atoms)
    out: dict[int, StateDelta] = {}
    for j in step_indices:
        # A budget exit reports the deepest step *reached*, so clamp rather than trust
        # it.
        sj = _atom_names(traj[min(max(j, 0), len(traj) - 1)].atoms)
        out[j] = StateDelta(
            added=tuple(sorted(sj - s0)), deleted=tuple(sorted(s0 - sj))
        )
    return out


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

    dev_blame: tuple[str, ...] = ()
    """Objects the *collateral deviation* names, on environments with no class-1 channel.

    Separate from :attr:`culprits` on purpose. A culprit was named by a validity check the
    environment itself ran; this is inferred from the observed-vs-predicted trace, which is
    all StickButton2D affords (kinder's collision check returns a bool). Keeping them in
    one field would let a model trained where the signal is observed be deployed where it
    is inferred without anything saying so. Absent on every DD2D record, so the token path
    that falls back to it is unreachable there.
    """

    state_delta: Optional[StateDelta] = None
    """``s_j`` relative to ``s_0``, or ``None`` when it was not asked for.

    ``None`` and ``StateDelta()`` mean different things and the distinction is
    load-bearing: ``None`` is *not computed*, ``StateDelta()`` is *computed, and the
    prefix changed nothing* (``j == 0``, ~48% of aggregated dd2d_v4 tokens). Only the
    token path requests it -- ``records_for_candidate`` is called three times per
    candidate in ``build_v3_example`` and the progression is not free.
    """

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
        dev_blame=tuple(obs.get("dev_blame") or ()),
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
    # sampler calls, nothing was re-sampled, so every query it reports genuinely ran.
    # That is derivable from stored metadata and is sound; where the count exceeds the
    # minimum the attempt re-sampled and may have stopped on a budget, so we claim
    # nothing. A domain that declares no cost model gets `False` -- no evidence, not
    # assumed evidence.
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
    with_state_delta: bool = False,
) -> list[FailureRecord]:
    """Every failure observed while refining one candidate (empty if it succeeded).

    ``with_state_delta`` populates :attr:`FailureRecord.state_delta`. It is off by
    default because the progression costs a trajectory per candidate and only the
    learned token path consumes it.

    Object names in the returned records -- ``args``, ``culprits``, ``unmoved`` and the
    delta alike -- are in whatever namespace ``episode`` is in. Records are built on
    demand and never serialized, so a canonicalized episode yields canonical records and
    a raw one yields raw records; the delta inherits that for free because it is derived
    from ``initial_abstract_state`` and ``operator_seq``, both of which
    ``canonicalize_episode`` already remaps.
    """
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
        recs = [
            _from_instrumented(candidate_idx, o, all_objects)
            for o in observations
            if isinstance(o, dict) and "schema" in o and "step_index" in o
        ]
    else:
        rec = _backfilled(
            candidate_idx,
            outcome,
            episode.skeleton_pool[candidate_idx],
            all_objects,
            spec,
            spec.goal_objects(episode),
        )
        recs = [rec] if rec is not None else []
    if not with_state_delta or not recs:
        return recs
    deltas = _state_deltas(
        episode,
        episode.skeleton_pool[candidate_idx],
        {r.step_index for r in recs},
    )
    return [replace(r, state_delta=deltas[r.step_index]) for r in recs]


def records_for_episode(
    episode: EpisodeRecord,
    candidate_indices=None,
    spec: Optional[DomainSpec] = None,
    with_state_delta: bool = False,
) -> list[FailureRecord]:
    """Flatten the failure records of the given candidates (default: all failures)."""
    spec = spec or spec_for(episode.provenance.env_variant)
    if candidate_indices is None:
        candidate_indices = [
            i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"
        ]
    out: list[FailureRecord] = []
    for idx in candidate_indices:
        out.extend(records_for_candidate(episode, idx, spec, with_state_delta))
    return out
