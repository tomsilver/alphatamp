"""The certificate rule: sound deductions from observed failures, applied outside the net.

v2.2 demoted candidate ``S`` when ``S`` was a subset of some staged set whose refinement
reported ``retrieve``. That rule is correct on DD2D but written in DD2D's vocabulary. v3
states the same deduction generically:

    demote sigma' if it issues the **same query on the same arguments** at some step j'
    with ``U(sigma', j') superset-eq U(sigma, j)``, and the domain declares that query
    **monotone** and **local**, and the observation shows the query actually ran.

Read it as: *this exact query already failed with fewer objects out of the way, and moving
fewer things cannot help.* On DD2D it reduces exactly to the subset rule -- at the retrieve
step ``U = all objects - staged``, so ``U' superset-eq U`` iff ``staged' subset-eq staged``
-- which is what ``test_proof_demotion_v3`` pins before the rule is allowed to differ.

**Two modes, because the exactness evidence is not always present.**

``permissive``
    Missing exhaustion evidence is treated as exhausted -- v2.2's semantics. Required for
    the equivalence check against v2.2 on pre-v3 collections, which have no such flag.
``strict``
    A record must positively witness that its query ran to exhaustion. This is the
    deployment default on instrumented collections, and it is what closes v2.2's
    unsoundness: on a budget exit the refiner still names ``retrieve(target)`` as the
    failing action although the retrieve was never tested, and trusting that let one
    dd2d_v2 candidate demote 12 genuinely-feasible plans.

**Demotion reorders; it never removes** (P-E). A demoted candidate loses a finite offset,
so if everything is proven dead the pool is still attempted in order. A wrong axiom
therefore costs attempts -- it cannot lose the feasible plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.failure_record import FailureRecord
from alphatamp.approaches.spectre.schema import EpisodeRecord

__all__ = ["CandidateQuery", "ProofStateV3", "candidate_queries", "DemotionMode"]

DemotionMode = Literal["permissive", "strict"]


@dataclass(frozen=True)
class CandidateQuery:
    """One (query, args, unmoved-set) a candidate will issue if it gets that far."""

    step_index: int
    schema: str
    args: tuple[str, ...]
    unmoved: frozenset[str]


def candidate_queries(
    episode: EpisodeRecord, spec: Optional[DomainSpec] = None
) -> list[list[CandidateQuery]]:
    """Per candidate, the queries it would issue, with the objects unmoved at each.

    Only *proof-eligible* query types are enumerated: a query the domain has not declared
    monotone+local can never license a demotion, so materialising it would be wasted work
    and would invite someone to use it as if it could.
    """
    spec = spec or spec_for(episode.provenance.env_variant)
    from alphatamp.approaches.spectre.domain import unmoved as unmoved_fn

    goal_objs = spec.goal_objects(episode)
    all_objects = frozenset(episode.object_registry)
    out: list[list[CandidateQuery]] = []
    for skeleton in episode.skeleton_pool:
        queries: list[CandidateQuery] = []
        for j, op in enumerate(skeleton.operator_seq):
            if not spec.axioms_for(op.name).proof_tier():
                continue
            queries.append(
                CandidateQuery(
                    step_index=j,
                    schema=op.name,
                    args=tuple(p.name for p in op.parameters),
                    unmoved=unmoved_fn(skeleton, j, all_objects, goal_objs),
                )
            )
        out.append(queries)
    return out


@dataclass
class ProofStateV3:
    """Accumulates provably-dead candidates as failures are observed."""

    queries: list[list[CandidateQuery]]
    spec: DomainSpec
    mode: DemotionMode = "strict"
    dead: set[int] = field(default_factory=set)
    _witnessed: list[tuple[str, tuple[str, ...], frozenset[str]]] = field(
        default_factory=list
    )

    def observe(self, records: Iterable[FailureRecord]) -> None:
        """Fold in the failures observed from one attempt."""
        new = False
        for rec in records:
            if not self.spec.axioms_for(rec.schema).proof_tier():
                continue
            if self.mode == "strict" and not rec.proves_failure():
                continue
            if self.mode == "permissive" and rec.budget_exhausted:
                # Even permissively, a record that positively says "I never ran" is not
                # evidence. Permissive relaxes *absent* evidence, not contrary evidence.
                continue
            self._witnessed.append((rec.schema, rec.args, rec.unmoved))
            new = True
        if new:
            self._recompute()

    def _recompute(self) -> None:
        for i, queries in enumerate(self.queries):
            if i in self.dead:
                continue
            for q in queries:
                if any(
                    q.schema == schema and q.args == args and q.unmoved >= unmoved
                    for schema, args, unmoved in self._witnessed
                ):
                    self.dead.add(i)
                    break

    def is_dead(self, idx: int) -> bool:
        return idx in self.dead
