"""Unified culprits, coverage and waste — the definitions of
``docs/unified_culprits_coverage_waste.md``.

**Why this module exists.** SPECTRE's **earlier** ``coverage``/``waste`` were computed
against ``S(c) = args \\ goal_objects``, which encodes "discretionary work = touching
non-goal objects". That is true on DD2D and false wherever tools exist: on StickButton2D
every candidate has ``S(c) = {stick}``, so that coverage was identically 0 (the culprit
buttons are goal objects, structurally barred from ``S``) and that waste was identically
1 for every stick-using plan — including the plan that responds perfectly to the
evidence. Blind and anti-signed respectively. That formula has been removed; the
definitions below are the sole coverage/waste path.

The replacements here derive everything from the **operator schemas**: which objects a
failure's own explanation names, and which of a candidate's steps the abstraction's
causal chain cannot account for. Nothing below names a drawer, a button or a stick.

**Status: deployed.** These are the coverage/waste definitions ``dataset`` emits (the
deployed definition since 2026-07-31; the earlier ``S(c) = args \\ goal_objects``
formula has been removed). The module still
carries its own lightweight :class:`UnifiedRecord` rather than extending
:class:`failure_record.FailureRecord`, which is now a deliberate boundary rather than a
staging area: ``FailureRecord`` is what a collection *stored*, this is what the features
are *computed over*, and keeping them apart is what let the class-2 deviation channel be
added for StickButton2D without touching the stored schema DD2D depends on.

Section numbers in docstrings refer to the design document.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from relational_structs import GroundAtom, GroundOperator

__all__ = [
    "Deviation",
    "UnifiedRecord",
    "actionable_objects",
    "anchored",
    "blame",
    "collateral",
    "coverage",
    "coverage_and_waste",
    "covered",
    "culprit_pool",
    "matched_steps",
    "records_from_failure_records",
    "scene_filters",
    "predicted_states",
    "superfluous_steps",
    "touch",
    "universal_objects",
    "waste",
]


def _names(atom: GroundAtom) -> frozenset[str]:
    return frozenset(o.name for o in atom.objects)


@lru_cache(maxsize=512)
def scene_filters(
    lifted_operators: frozenset, objects: frozenset
) -> tuple[frozenset[str], frozenset[str]]:
    """``(universal, actionable)`` for one problem's grounded domain.

    Cached because both filters are properties of the *problem*, not of the candidate,
    and ``build_example`` is called many times per episode. The lifted operator set
    is recovered from the pool's own ``GroundOperator.parent``, so nothing here needs to
    know which environment it is looking at.
    """
    from relational_structs.utils import (  # pylint: disable=import-outside-toplevel
        all_ground_operators,
    )

    ground = list(all_ground_operators(set(lifted_operators), set(objects)))
    return universal_objects(ground), actionable_objects(ground)


@lru_cache(maxsize=64)
def _atom_table(
    lifted_operators: frozenset, objects: frozenset, extra: frozenset
) -> tuple[dict, dict]:
    """``(predicates_by_name, objects_by_name)`` for rebuilding serialized atoms.

    A class-2 deviation is stored in ``refiner_metadata`` as ``(predicate, args)`` name
    pairs — it has to be, or ``canonicalize_episode`` cannot rewrite the object names in
    it. Rebuilding real :class:`GroundAtom`s is therefore required, not an optimisation:
    every consumer downstream (:func:`collateral`, :func:`covered`) compares deviation
    atoms by identity against operator effects and progressed states.

    Predicates are harvested from the pool's own lifted operators plus whatever the
    initial state carries, which between them cover every predicate a plan can mention.
    """
    predicates = {}
    for op in lifted_operators:
        for atom in (
            set(op.preconditions) | set(op.add_effects) | set(op.delete_effects)
        ):
            predicates[atom.predicate.name] = atom.predicate
    for pred in extra:
        predicates[pred.name] = pred
    return predicates, {o.name: o for o in objects}


def _rebuild_atoms(pairs, predicates: dict, objects: dict) -> frozenset[GroundAtom]:
    """Serialized ``[[predicate, [arg, ...]], ...]`` back into ground atoms.

    A pair naming an unknown predicate or object is **dropped**, not guessed. That can
    only happen if a record outlived the vocabulary it was written against, and a
    silently mis-bound atom would corrupt the coverage test far more expensively than
    a missing one.
    """
    out = set()
    for pair in pairs or ():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        pred = predicates.get(str(pair[0]))
        if pred is None:
            continue
        args = [objects[str(n)] for n in pair[1] if str(n) in objects]
        if len(args) != len(pair[1]):
            continue
        out.add(GroundAtom(pred, args))
    return frozenset(out)


def records_from_failure_records(episode, context, spec) -> list[UnifiedRecord]:
    """Adapt one episode's stored failure observations into unified records.

    Both §2 classes come through here and **every** observed failure yields a record,
    including one whose channel is empty for this environment. Class 1 is the
    ``culprits`` list — the objects the refiner's own validity check named, which is all
    DD2D can produce. Class 2 is the ``dev_added``/``dev_deleted`` deviation between the
    predicted and achieved abstract state, which is all StickButton2D can produce
    (kinder's collision check returns a bool without naming anything).

    **Blameless records are kept.** A failure that names nobody is still an observation
    that this step failed, and the record-token stream reads it. It is provably inert
    for ``coverage``/``waste``: it contributes nothing to ``K``, :func:`covered` skips
    it for every object, and :func:`_justified` never consults it — with :func:`waste`
    abstaining on an empty pool so the one arithmetic edge case cannot leak. Which class
    an environment produces is therefore a property of its data, never a branch a
    consumer has to take.
    """
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.failure_record import records_for_candidate

    predicates, objects = _atom_table(
        frozenset(
            op.parent for skel in episode.skeleton_pool for op in skel.operator_seq
        ),
        frozenset(episode.initial_abstract_state.objects),
        frozenset(a.predicate for a in episode.initial_abstract_state.atoms),
    )
    out: list[UnifiedRecord] = []
    for idx in context:
        seq = episode.skeleton_pool[idx].operator_seq
        if not seq:
            continue
        meta = episode.outcomes[idx].refiner_metadata or {}
        raw = meta.get("failures")
        # Pair positionally with the records `records_for_candidate` actually built,
        # which means applying **its** validity filter here too. Without that, one
        # malformed entry shifts the alignment and every later deviation is attached
        # to the wrong record -- a corruption with no symptom, since both sides stay
        # well-formed.
        raw = (
            [
                o
                for o in raw
                if isinstance(o, dict) and "schema" in o and "step_index" in o
            ]
            if isinstance(raw, (list, tuple))
            else []
        )
        for pos, rec in enumerate(records_for_candidate(episode, idx, spec)):
            # A budget exit reports the deepest step *reached*; clamp rather than trust
            # it.
            step = seq[min(max(rec.step_index, 0), len(seq) - 1)]
            entry = raw[pos] if pos < len(raw) and isinstance(raw[pos], dict) else {}
            added, deleted = entry.get("dev_added"), entry.get("dev_deleted")
            deviation = (
                Deviation(
                    added=_rebuild_atoms(added, predicates, objects),
                    deleted=_rebuild_atoms(deleted, predicates, objects),
                )
                if added is not None or deleted is not None
                else None
            )
            out.append(
                UnifiedRecord(
                    failed_step=step,
                    deviation=deviation,
                    check_blame=tuple(rec.culprits),
                )
            )
    return out


# --------------------------------------------------------------------------- #
# §1 — the two filters
# --------------------------------------------------------------------------- #
def universal_objects(ground_ops: Iterable[GroundOperator]) -> frozenset[str]:
    """Objects appearing in the argument list of **every** ground operator instance.

    Behaviourally the robot. On DD2D this is empty **by construction** — ``pick(o1)``
    does not mention ``o2`` — which is why every exclusion keyed on it is provably a no-
    op there rather than an empirical one.
    """
    arg_sets = [frozenset(p.name for p in op.parameters) for op in ground_ops]
    if not arg_sets:
        return frozenset()
    return frozenset.intersection(*arg_sets)


def actionable_objects(ground_ops: Iterable[GroundOperator]) -> frozenset[str]:
    """Objects some ground operator's *effects* mention — i.e. objects a plan can act
    on.

    Honestly idle on both current environments (DD2D's culprits are all movable
    blockers; SB2D's table and walls never enter the abstract object universe at all).
    Retained for domains whose abstract universe carries inert furniture.
    """
    out: set[str] = set()
    for op in ground_ops:
        for atom in set(op.add_effects) | set(op.delete_effects):
            out |= _names(atom)
    return frozenset(out)


def anchored(
    atoms: Iterable[GroundAtom], universal: frozenset[str]
) -> frozenset[GroundAtom]:
    """The atoms mentioning at least one non-universal object.

    Nullary atoms are excluded.     Without this filter, bookkeeping atoms thread
    everything together: ``handempty``     chains every DD2D staging pair into the
    causal spine (destroying waste's backward     compatibility), and
    ``AboveNoButton``/``HandEmpty`` make unrelated steps match each     other's
    contexts.
    """
    return frozenset(a for a in atoms if _names(a) - universal)


# --------------------------------------------------------------------------- #
# §2 — records, collateral deviation, culprits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Deviation:
    """Observed minus predicted: atoms unexpectedly added, atoms expected but
    missing."""

    added: frozenset[GroundAtom] = frozenset()
    deleted: frozenset[GroundAtom] = frozenset()

    def is_empty(self) -> bool:
        """No deviation in either polarity."""
        return not self.added and not self.deleted


@dataclass(frozen=True)
class UnifiedRecord:
    """One observed refinement failure, in the two-class taxonomy of §2.

    ``deviation is None`` marks **class 1** (a validity check rejected the sample before
    a successor state existed); blame is then whatever objects the check named.
    Otherwise it is **class 2** (the sample executed and the trace check found observed
    ≠ predicted).
    """

    failed_step: GroundOperator
    deviation: Optional[Deviation] = None
    check_blame: tuple[str, ...] = ()

    @property
    def is_class_1(self) -> bool:
        """A constraint rejection rather than an effect mismatch."""
        return self.deviation is None


def collateral(record: UnifiedRecord) -> Deviation:
    """``Δ̃_r`` — the deviation minus the failed step's own declared effects (§2).

    The stripped remainder is **means failure**: the step's adds that never materialized
    and its deletes that never took, i.e. the statement "this query, as attempted, does
    not produce its effects". That generates no culprits — it belongs to the burned-
    query token, which is the channel this design assigns to reachability/modality
    evidence.

    The distinction is what makes the class-2 coverage test discriminative. On an out-
    of-reach robot press the raw deviation is entirely means failure, so ``Δ̃_r`` is
    empty; without the restriction, its ``D_r`` half would be satisfied by construction
    at any matched step (a step about to add those atoms has them absent), handing full
    credit to a candidate retrying the identical doomed press.
    """
    if record.deviation is None:
        return Deviation()
    return Deviation(
        added=record.deviation.added - frozenset(record.failed_step.delete_effects),
        deleted=record.deviation.deleted - frozenset(record.failed_step.add_effects),
    )


def blame(record: UnifiedRecord) -> frozenset[str]:
    """The objects a record's own explanation names — its collateral damage.

    Class 1: the objects the violated check named. Class 2: the objects mentioned in the
    *collateral* deviation. A pure means failure blames nobody.
    """
    if record.is_class_1:
        return frozenset(record.check_blame)
    dev = collateral(record)
    return frozenset(n for a in (dev.added | dev.deleted) for n in _names(a))


def culprit_pool(
    records: Iterable[UnifiedRecord], ground_ops: Iterable[GroundOperator]
) -> frozenset[str]:
    """``K = (Actionable \\ Universal) ∩ ⋃ blame(r)`` (§2).

    Universal objects are excluded from ``K`` **itself**, not merely from anchoring. The
    ranking-inertness lemma would tolerate a uniformly-covered object inside coverage,
    but it does not extend to waste, whose ``justified`` is a per-step existential over
    ``K`` — one universal object there spuriously justifies every superfluous step that
    touches it, and every step touches the robot.
    """
    ops = list(ground_ops)
    blamed: set[str] = set()
    for record in records:
        blamed |= blame(record)
    return frozenset(blamed & (actionable_objects(ops) - universal_objects(ops)))


# --------------------------------------------------------------------------- #
# §3 — context matching
# --------------------------------------------------------------------------- #
def matched_steps(
    candidate: Sequence[GroundOperator],
    record: UnifiedRecord,
    universal: frozenset[str],
) -> frozenset[int]:
    """Indices of steps that re-enter the record's situation (§3).

    A step matches iff it accomplishes some anchored effect the failed step was trying
    to accomplish, **matched by what the step does rather than what it is called** — so
    a stick-press of ``b2`` re-enters the context of a failed robot-press of ``b2``.
    Sign is respected: adds match adds and deletes match deletes, because a step that
    adds what the failed step deleted is the opposite of a re-attempt, not an instance
    of one.
    """
    sig_add = anchored(record.failed_step.add_effects, universal)
    sig_del = anchored(record.failed_step.delete_effects, universal)
    return frozenset(
        j
        for j, step in enumerate(candidate)
        if (frozenset(step.add_effects) & sig_add)
        or (frozenset(step.delete_effects) & sig_del)
    )


def touch(candidate: Sequence[GroundOperator], obj: str) -> frozenset[int]:
    """Indices of steps whose effects mention ``obj``."""
    return frozenset(
        j
        for j, step in enumerate(candidate)
        if any(
            obj in _names(a) for a in set(step.add_effects) | set(step.delete_effects)
        )
    )


def predicted_states(
    candidate: Sequence[GroundOperator], initial_atoms: Iterable[GroundAtom]
) -> list[frozenset[GroundAtom]]:
    """``ŝ_0 … ŝ_L`` by STRIPS progression; ``ŝ_j`` is the state *before* step ``j``.

    The same machinery ``state_delta`` already runs — no new instrumentation.
    """
    states = [frozenset(initial_atoms)]
    for step in candidate:
        states.append(
            (states[-1] - frozenset(step.delete_effects)) | frozenset(step.add_effects)
        )
    return states


# --------------------------------------------------------------------------- #
# §4 — coverage
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Memo:
    """Per-(candidate, context) precomputation.

    Every scalar below was previously recomputed inside the innermost loops:
    ``matched_steps`` once per (culprit × record), ``touch`` once per (culprit × record
    × superfluous step), and ``blame``/``collateral`` on every one of those. All are
    pure functions of things that do not vary across those loops, so hoisting them is a
    pure speedup with byte-identical output — pinned by
    ``test_memoized_matches_naive_recomputation``.
    """

    matched: tuple[frozenset[int], ...]  # parallel to records
    blames: tuple[frozenset[str], ...]  # parallel to records
    collaterals: tuple[Deviation, ...]  # parallel to records
    touched: dict[str, frozenset[int]]  # per culprit


def _build_memo(
    candidate: Sequence[GroundOperator],
    records: Sequence[UnifiedRecord],
    pool: frozenset[str],
    universal: frozenset[str],
) -> _Memo:
    return _Memo(
        matched=tuple(matched_steps(candidate, r, universal) for r in records),
        blames=tuple(blame(r) for r in records),
        collaterals=tuple(collateral(r) for r in records),
        touched={o: touch(candidate, o) for o in pool},
    )


def covered(
    obj: str,
    candidate: Sequence[GroundOperator],
    records: Sequence[UnifiedRecord],
    states: Sequence[frozenset[GroundAtom]],
    universal: frozenset[str],
    memo: Optional[_Memo] = None,
) -> bool:
    """Whether ``candidate`` discharges culprit ``obj``, conjunctively across records
    (§4).

    Class-dependent, because the abstraction can state class-2 hazards but not class-1
    ones: blockedness is not a predicate, so collisions get an index-precedence proxy,
    while an unpredicted atom *is* a predicate and gets an exact state test.

    ``memo`` is an optimisation only; omitting it recomputes everything in place and
    must give the same answer.
    """
    for idx, record in enumerate(records):
        record_blame = memo.blames[idx] if memo else blame(record)
        if obj not in record_blame:
            continue
        matched = (
            memo.matched[idx] if memo else matched_steps(candidate, record, universal)
        )

        if record.is_class_1:
            touched = (
                memo.touched.get(obj, frozenset()) if memo else touch(candidate, obj)
            )
            if matched:
                # Deal with it before the earliest point the situation recurs.
                if not any(j < min(matched) for j in touched):
                    return False
            elif not touched:
                # No recurrence recognized: bare membership, the deployed DD2D
                # semantics.
                return False
            continue

        # Class 2: replaying the recorded accident against the candidate's own
        # predictions must be a no-op everywhere the situation recurs. Restricted to the
        # atoms mentioning `obj`, so one story's two culprits are judged separately.
        dev = memo.collaterals[idx] if memo else collateral(record)
        need = frozenset(a for a in dev.added if obj in _names(a))
        forbid = frozenset(a for a in dev.deleted if obj in _names(a))
        for j in matched:
            if not need <= states[j]:
                return False
            if forbid & states[j]:
                return False
    return True


def coverage(
    candidate: Sequence[GroundOperator],
    records: Sequence[UnifiedRecord],
    pool: frozenset[str],
    initial_atoms: Iterable[GroundAtom],
    universal: frozenset[str],
    memo: Optional[_Memo] = None,
) -> float:
    """Recall over the failures' own stories: the fraction of ``K`` this candidate
    discharges.

    Exactly ``0`` when no failure has been observed — the leakage invariant, so the
    first attempt stays purely static.
    """
    if not records or not pool:
        return 0.0
    memo = memo or _build_memo(candidate, records, pool, universal)
    states = predicted_states(candidate, initial_atoms)
    hits = sum(
        1 for k in pool if covered(k, candidate, records, states, universal, memo)
    )
    return hits / len(pool)


# --------------------------------------------------------------------------- #
# §5 — waste
# --------------------------------------------------------------------------- #
def superfluous_steps(
    candidate: Sequence[GroundOperator],
    goal_atoms: Iterable[GroundAtom],
    universal: frozenset[str],
) -> frozenset[int]:
    """Steps the abstraction's own causal chain cannot explain (§5).

    Backward relevance over anchored needs. The pass detects irrelevance but **not
    threats** — it never consults delete effects — which is sound here only because
    candidates are STRIPS-valid sequential plans by construction.
    """
    needed = set(anchored(goal_atoms, universal))
    out: set[int] = set()
    for j in range(len(candidate) - 1, -1, -1):
        step = candidate[j]
        adds = frozenset(step.add_effects)
        if adds & needed:
            needed = (needed - adds) | set(anchored(step.preconditions, universal))
        else:
            out.add(j)
    return frozenset(out)


def _justified(
    index: int,
    candidate: Sequence[GroundOperator],
    records: Sequence[UnifiedRecord],
    pool: frozenset[str],
    universal: frozenset[str],
    memo: Optional[_Memo] = None,
) -> bool:
    """Whether a superfluous step answers to some named culprit, at a useful position.

    Justification stays at index level even for class-2 blame — a recorded
    simplification carrying the same monotonicity caveat as the class-1 test.
    """
    memo = memo or _build_memo(candidate, records, pool, universal)
    for obj in pool:
        if index not in memo.touched.get(obj, frozenset()):
            continue
        if all(
            not memo.matched[i] or index < min(memo.matched[i])
            for i, r in enumerate(records)
            if obj in memo.blames[i]
        ):
            return True
    return False


def waste(
    candidate: Sequence[GroundOperator],
    records: Sequence[UnifiedRecord],
    pool: frozenset[str],
    goal_atoms: Iterable[GroundAtom],
    universal: frozenset[str],
    memo: Optional[_Memo] = None,
) -> float:
    """Precision over unexplained work: of the steps the abstraction says you did not
    need, the fraction answering to nothing the evidence has named.

    Live steps — the causal spine that actually produces the goal, tool acquisition
    included — never enter the denominator. That is what dissolves the SB2D anti-signal
    by definition rather than by a per-environment switch.

    **Abstains on an empty culprit pool.** With no named culprits nothing can justify
    any idle step, so the arithmetic would return 1.0 for every candidate -- a maximally
    confident verdict derived from zero evidence. It is also what makes it safe to keep
    blameless records (:func:`records_from_failure_records` no longer filters them):
    every other consumer already skips a record that blames nobody, so without this
    guard the single observable effect of including them would be waste flipping 0.0 to
    1.0 on contexts that named no one.
    """
    if not records or not pool:
        return 0.0
    idle = superfluous_steps(candidate, goal_atoms, universal)
    if not idle:
        return 0.0
    memo = memo or _build_memo(candidate, records, pool, universal)
    unjust = sum(
        1 for j in idle if not _justified(j, candidate, records, pool, universal, memo)
    )
    return unjust / len(idle)


def coverage_and_waste(
    candidate: Sequence[GroundOperator],
    records: Sequence[UnifiedRecord],
    pool: frozenset[str],
    initial_atoms: Iterable[GroundAtom],
    goal_atoms: Iterable[GroundAtom],
    universal: frozenset[str],
) -> tuple[float, float]:
    """Both features for one candidate, sharing a single :class:`_Memo`.

    The tensorizer wants both columns for every candidate, and they depend on the same
    per-(candidate, record) quantities. Computing them together halves the hoisted work
    again and is the entry point ``dataset`` uses.
    """
    if not records:
        return 0.0, 0.0
    memo = _build_memo(candidate, records, pool, universal)
    return (
        coverage(candidate, records, pool, initial_atoms, universal, memo),
        waste(candidate, records, pool, goal_atoms, universal, memo),
    )
