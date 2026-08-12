"""V3 tensorizer: ``EpisodeRecord`` -> ``SpectreV3Batch``, via the domain contract.

Structurally this is v2.2's tensorizer with the DD2D literals removed. Two things it no
longer knows:

- **which objects a candidate manipulates** -- was ``op.name == "place-buffer"``, now
  :func:`domain.DomainSpec.manipulated`, verified identical on 120000/120000 skeletons;
- **which failures license demotion** -- was ``failure_action.startswith("retrieve")``,
  now the per-query axiom declaration plus the observation's own budget flag.

Three v2.2 features are deliberately *not* carried over:

- ``exclude_marginal`` was inert twice over (the DD2D refiner only writes ``status`` in
  ``{feasible, infeasible}``, and even when it fired ``collate`` folded the resulting
  ``None`` back to ``False`` because the batch has no per-candidate ignore mask). v3
  declines to carry a flag that silently does nothing; reinstating the behaviour needs a
  real label mask, not a flag.
- ``demotion_source="computed"`` -- the geometry-reconstruction path -- is dropped (R2).
  It was worth a measured ~14%, but it is the last per-environment geometry routine in
  the deployment story, and the observed signal is what generalizes.
- The **short-first prior** as a scorer feature (R1). The plan-length column survives
  only as the within-length loss's bucket key, which is now :func:`domain.length_key`; the
  model sees no prior (``V3Config.n_prior_feats == 0``). It was a per-dataset hand switch
  that diverged training on the easier collection, and note it was never a clean feature
  ablation anyway: enabling it also zero-inits the scorer head.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.dataset_v2 import (
    _V2Example,
    collate_v2,
    resample_ring,
)
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.facts import TIER_IDS, gather_context_facts
from alphatamp.approaches.spectre.failure_record import records_for_candidate
from alphatamp.approaches.spectre.model_v2 import D_REL_V3, MAX_FACT_ARGS
from alphatamp.approaches.spectre.model_v3 import (
    MAX_DELTA_ATOMS,
    MAX_RECORD_ARGS,
    MAX_RECORD_CULPRITS,
    N_OBJ_EVIDENCE,
    N_OVERLAP_V3,
    N_RECORD_SCALARS,
    SpectreV3Batch,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.tags import assign_tags
from alphatamp.approaches.spectre.unified_evidence import blame as _unified_blame
from alphatamp.approaches.spectre.unified_evidence import (
    coverage_and_waste,
    records_from_failure_records,
    scene_filters,
)
from alphatamp.approaches.spectre.vocab import Vocab

__all__ = [
    "build_v3_example",
    "collate_v3",
    "sample_context",
    "build_record_arrays",
]

#: One atom of a state delta, as ``(predicate id, argument tags)``.
DeltaAtomArray = tuple[int, list[int]]
#: One record's ``s_j - s_0``, as ``(added atoms, deleted atoms)``.
DeltaArrays = tuple[list[DeltaAtomArray], list[DeltaAtomArray]]
# : One record token: ``(schema id, arg tags, culprit tags, scalars)``, optionally
# followed : by the state delta. The trailing element is present iff the delta was
# requested -- the : model reads it by position, and a 4-tuple stream is byte-for-byte
# the pre-delta one.
RecordArray = tuple  # (int, list[int], list[int], list[float][, DeltaArrays])


@dataclasses.dataclass
class _V3Example(_V2Example):
    """``_V2Example`` plus the per-object evidence summary.

    A subclass rather than a third return value: ``collate_v2`` reads attributes, so a
    ``_V3Example`` flows through the v2 collation untouched, and every existing caller of
    ``build_v3_example`` keeps its two-tuple. The field defaults to ``None``, so the
    compat path is byte-identical (D-8).
    """

    obj_evidence: Optional[np.ndarray] = None


def records_for_evidence(
    episode: EpisodeRecord, ctx: frozenset, spec: DomainSpec
) -> tuple[list, list]:
    """Split the context's records into ``(hint_tier, proof_tier)``.

    The token path consumes only the hint tier, exactly as v2.2 did. The object summary
    consumes both, but keeps them in **separate columns** -- see
    :func:`_object_evidence` for why that is not a violation of the tier split.
    """
    hint, proof = [], []
    for idx in sorted(ctx):
        for rec in records_for_candidate(episode, idx, spec):
            if spec.axioms_for(rec.schema).proof_tier() and rec.proves_failure():
                proof.append(rec)
            else:
                hint.append(rec)
    return hint, proof


def _object_evidence(
    ctx: frozenset,
    subsets: list,
    objects: list,
    hint_records: list,
    proof_records: list,
    max_len: int = 8,
) -> np.ndarray:
    """Summarise the observed failures onto the objects they name.

    See SceneEncoderV3.
        Five columns per object, all fractions in [0, 1], all zero before any failure:

        0. fraction of failed candidates that manipulate ``o``
        1. fraction of hint records naming ``o`` as an argument
        2. fraction of hint records naming ``o`` as a culprit
        3. mean normalized depth of the records naming ``o``
        4. fraction of **proof-tier** records naming ``o`` as a culprit

        **Column 4 needs its own justification, because it looks like a tier violation and is
        not.** Proof-tier records are kept out of the *token* path because what a token
        exposes there is the failed **set**, and the net learns the crude size correlate from
        it ("blocked sets are large, so prefer longer") -- L4, which cost +13.5 FP on s1.
        Column 4 exposes something categorically different: the **identity of an object the
        refiner's own collision check reported as blocking**. That is an *observation* (C2's
        legal source), not the *deduction*; the deduction still acts only outside the net as
        demotion (C5), and no weight can override it.

        It is also the honest, observed version of the `clears` predicate that L2 rejected.
        `clears` was rejected for being a hand-coded per-environment geometric routine we ran
        ourselves. This is the refiner reporting what it already computed -- the same
        legality class as `failure_action`, and exactly what §6.1 lists ``culprits`` for. On
        dd2d_v4 `retrieve` records carry culprits 100% of the time, so this column is where
        "which object is actually blocking the target" enters the model at all.
    """
    ev = np.zeros((len(objects), N_OBJ_EVIDENCE), dtype=np.float32)
    if not ctx:
        return ev
    index = {o: i for i, o in enumerate(objects)}
    n_ctx = float(len(ctx))
    for f in ctx:
        for o in subsets[f]:
            if o in index:
                ev[index[o], 0] += 1.0 / n_ctx
    n_hint = float(len(hint_records)) or 1.0
    depth_sum = np.zeros(len(objects), dtype=np.float32)
    depth_cnt = np.zeros(len(objects), dtype=np.float32)
    for rec in hint_records:
        for o in rec.args:
            if o in index:
                ev[index[o], 1] += 1.0 / n_hint
                depth_sum[index[o]] += rec.step_index
                depth_cnt[index[o]] += 1.0
        # `culprits or dev_blame`: class-1 envs (DD2D) report `culprits`; class-2 envs
        # (SB2D, and any future kinder env) report `dev_blame` instead. The token path
        # (build_record_arrays) already falls back this way; without it here, this column
        # is identically zero on every class-2 environment -- the same silent per-env
        # degradation the goal-channel fix removed. See docs/decisions 2026-08-08.
        for o in rec.culprits or rec.dev_blame:
            if o in index:
                ev[index[o], 2] += 1.0 / n_hint
    # Normalise depth by the episode's own longest plan, not a hard-coded DD2D constant of
    # 8. `step_index` ranges over plan positions, so the right scale is the deepest plan in
    # this problem's pool; `/ 8.0` under-normalised SB2D (plans length 6) and would
    # mis-scale any env whose plans are longer.
    ev[:, 3] = depth_sum / np.maximum(depth_cnt, 1.0) / float(max(max_len, 1))
    n_proof = float(len(proof_records)) or 1.0
    for rec in proof_records:
        for o in rec.culprits:
            if o in index:
                ev[index[o], 4] += 1.0 / n_proof
    return np.clip(ev, 0.0, 1.0)


def sample_context(
    fail_idx: list[int],
    rng: np.random.Generator,
    p_empty: float = 0.35,
    p_drop_facts: float = 0.3,
    max_f: int = 8,
    tail_max_f: int = 0,
) -> tuple[frozenset[int], bool]:
    """Sample a failure context ``F`` plus an evidence-dropout flag.

    Mass is heavy at ``|F| = 0`` because that is the deployment start: the static pathway
    has to stand on its own before any failure has been observed. ``hide_facts`` drops
    the evidence for an example so the ranker cannot become dependent on it.

    ``tail_max_f > 0`` switches on **rollout-aligned** sampling: half the non-empty mass
    stays uniform on ``1..max_f`` (the easy strata, where rollouts end after a few
    attempts) and half spreads uniformly out to ``tail_max_f``. The default ``max_f=8``
    inherited from v2.2 is a genuine train/deploy mismatch at the hard strata -- an s3
    rollout is queried at ``|F|`` up to ~40, a regime training never showed it -- and
    this is the knob that closes it. Named for the original RT2D
    ``rollout_aligned_mix``, whose purpose was the same: make training mass match the
    test-time visit distribution.
    """
    if not fail_idx or rng.random() < p_empty:
        return frozenset(), False
    hi = max_f if (tail_max_f <= 0 or rng.random() < 0.5) else tail_max_f
    size = int(rng.integers(1, min(hi, len(fail_idx)) + 1))
    chosen = rng.choice(np.asarray(fail_idx), size=size, replace=False)
    return frozenset(int(i) for i in chosen), bool(rng.random() < p_drop_facts)


def _hint_fact_arrays(
    episode: EpisodeRecord, context_f: frozenset[int], tags: dict[str, int]
) -> tuple[list[int], list[int], list[list[int]]]:
    """Hint-tier facts of the failed candidates, bound to episode-local tags.

    Proof-tier facts are excluded on purpose. Their sound consequence is applied outside
    the network as demotion; handing them to the scorer as *tokens* invites it to learn
    the crude correlate instead ("blocked sets are large, so prefer longer plans"),
    which measurably harmed the easy strata in v2.2 until the tiers were split.
    """
    hint = TIER_IDS["hint"]
    type_ids: list[int] = []
    tier_ids: list[int] = []
    arg_tags: list[list[int]] = []
    for fact in gather_context_facts(episode, sorted(context_f)):
        if fact.tier_id != hint:
            continue
        type_ids.append(fact.type_id)
        tier_ids.append(fact.tier_id)
        arg_tags.append([tags[a] for a in fact.args if a in tags][:MAX_FACT_ARGS])
    return type_ids, tier_ids, arg_tags


def _aggregate_per_query(records: list) -> list:
    """Collapse a candidate's failures to one record per distinct ``(schema, args)``.

    §6.1 defines a record as *the failing query and its arguments* -- one per query, not
    one per failed sample. The instrumented refiner emits one per sample, so a candidate
    whose `place-buffer(o)` was retried across many buffer poses contributes hundreds of
    near-identical tokens (measured: mean 2.2 per candidate but **max 290**, so a single
    s1 context at |F|=30 reached ~720 tokens against v2.2's ~40 facts). That is not extra
    information -- the samples are 99.3% distinct only in *which pose* failed, which the
    token does not even encode -- but it does dilute the scorer's attention and let one
    unlucky candidate dominate the evidence memory.

    Aggregation keeps the deepest occurrence (the furthest the plan got), sums effort,
    and takes the union of culprits, so nothing the token *encodes* is lost.
    """
    best: dict[tuple, list] = {}
    for rec in records:
        key = (rec.schema, rec.args)
        cur = best.get(key)
        if cur is None:
            best[key] = [rec, rec.n_step, set(rec.culprits)]
            continue
        cur[1] += rec.n_step
        cur[2].update(rec.culprits)
        if rec.step_index > cur[0].step_index:
            cur[0] = rec
    out = []
    for rec, effort, culprits in best.values():
        out.append(
            dataclasses.replace(rec, n_step=effort, culprits=tuple(sorted(culprits)))
        )
    return out


def _delta_arrays(delta, tags: dict[str, int], vocab: Vocab, arity: int) -> DeltaArrays:
    """One record's state delta as ``(added, deleted)`` lists of ``(pred id, arg
    tags)``.

    Predicate ids are shifted by **+1**: the vocab reserves index 0 for ``<OOV>`` while
    the embedding reserves it for padding, so without the shift an unknown predicate
    would be indistinguishable from an empty slot. ``Vocab.pred_idx`` raises on OOV,
    hence the ``.get`` -- the same idiom the schema lookup below uses.
    """

    def _role(atoms) -> list[DeltaAtomArray]:
        out = []
        for pred, args in atoms[:MAX_DELTA_ATOMS]:
            pid = int(vocab.predicates.get(pred, {"idx": 0})["idx"]) + 1
            out.append((pid, [tags[a] for a in args if a in tags][:arity]))
        return out

    if delta is None:
        return ([], [])
    return (_role(delta.added), _role(delta.deleted))


def build_record_arrays(
    episode: EpisodeRecord,
    context_f: frozenset[int],
    tags: dict[str, int],
    vocab: Vocab,
    spec: DomainSpec,
    aggregate: bool = False,
    state_delta: bool = False,
) -> list[RecordArray]:
    """Failure records of the tried candidates, as ``(schema_id, args, culprits,
    scalars)``.

    **Proof-tier records are excluded**, exactly as v2.2 excluded proof-tier facts. Their
    sound consequence is applied outside the net as demotion; feeding them in as tokens
    invites the scorer to learn the crude correlate instead ("blocked sets are large, so
    prefer longer plans"), which measurably wrecked the easy strata in v2.2 until the
    tiers were split. What reaches the net is evidence the deduction could *not* use.

    Scalars are ``[j/L, log1p(effort)/10, exhausted, effort_is_total]``. The last is not
    decoration: backfilled records report whole-attempt effort while instrumented ones
    report per-step, so the flag tells the net which quantity it is reading instead of
    letting a re-collection silently redefine the column.

    ``state_delta`` appends §6.1's ``s_j`` as a fifth element, the delta from ``s_0``.
    It is computed here, on ``episode`` -- which ``build_v3_example`` has already
    canonicalized -- so its object names land in the same ``tags`` namespace as the args
    and culprits. Computing it *after* aggregation is what makes it the *furthest
    reached* state per query: ``_aggregate_per_query`` keeps the deepest record, and the
    delta rides along on it. Note the delta's **size** is ~fully determined by the
    ``j/L`` scalar already present (measured corr 0.940), so what it adds is object
    *identity*; no count feature is derived from it, which would just be another length proxy.
    """
    arity = max(vocab.max_predicate_arity, 1)
    out: list[RecordArray] = []
    for idx in sorted(context_f):
        skeleton = episode.skeleton_pool[idx]
        plan_len = max(len(skeleton.operator_seq), 1)
        cand_records = records_for_candidate(
            episode, idx, spec, with_state_delta=state_delta
        )
        if aggregate:
            cand_records = _aggregate_per_query(cand_records)
        for rec in cand_records:
            if spec.axioms_for(rec.schema).proof_tier() and rec.proves_failure():
                continue  # handled structurally by demotion, not learned
            row: RecordArray = (
                int(vocab.operators.get(rec.schema, 0)),
                [tags[a] for a in rec.args if a in tags][:MAX_RECORD_ARGS],
                # `dev_blame` is the fallback for environments with no class-1 channel:
                # objects named by the collateral deviation rather than by a check. Never
                # both -- `culprits` is empty exactly where `dev_blame` is populated -- so
                # the slot always carries one provenance, and on DD2D `dev_blame` is
                # absent and this reduces to the original expression.
                [tags[c] for c in (rec.culprits or rec.dev_blame) if c in tags][
                    :MAX_RECORD_CULPRITS
                ],
                [
                    rec.step_index / plan_len,
                    math.log1p(max(rec.n_step, 0)) / 10.0,
                    1.0 if rec.exhausted else 0.0,
                    1.0 if rec.effort_is_total else 0.0,
                ],
            )
            if state_delta:
                row = row + (_delta_arrays(rec.state_delta, tags, vocab, arity),)
            out.append(row)
    return out


def build_v3_example(
    episode: EpisodeRecord,
    vocab: Vocab,
    rng: Optional[np.random.Generator] = None,
    max_tags: int = 32,
    evidence: bool = False,
    context_f: Optional[frozenset[int]] = None,
    hide_facts: bool = False,
    augment_tags: bool = True,
    spec: Optional[DomainSpec] = None,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
    coverage_feats: bool = False,
    coverage_mode: str = "both",
    # Deliberately **False** here even though unified is the deployed default since
    # 2026-07-31. This is the primitive; policy lives one level up, in
    # ``TrainV3Config`` (what a new run trains with) and in the checkpoint cfg (what a
    # trained model is scored with, via ``inference_v3._emit_kwargs``). Flipping it here
    # would silently change definition for any caller that does not pass it -- e.g.
    # ``spectre_d2_s2.py``, a frozen diagnostic that must keep reproducing its
    # deployed-definition numbers.
    unified_coverage: bool = False,
    state_delta: bool = False,
) -> tuple[_V2Example, list[RecordArray]]:
    """Tensorize one geometry-carrying episode for the v3 model.

    Returns ``(example, record_arrays)``. The records come back from *here* rather than
    from a separate call because they must use the same canonicalization and the same
    tag assignment as the example -- computing them separately would both double the
    canonicalization cost and risk binding record tags to a different permutation than
    the scene tags, which is exactly the identity bug that made record tokens carry no
    object information at all. A caller whose model has no record encoder simply ignores
    the second element; it costs nothing to produce when the context is empty.

    ``spec`` defaults to the contract registered for the episode's own ``env_variant``,
    so a caller cannot accidentally tensorize DD2D under another domain's axioms.
    """
    if episode.scene_geometry is None:
        raise ValueError("build_v3_example requires scene_geometry")
    spec = spec or spec_for(episode.provenance.env_variant)

    canon = canonicalize_episode(episode, rng=None)
    geo = canon.scene_geometry
    assert geo is not None
    tags = assign_tags(
        [o.name for o in geo.objects],
        rng=(rng if augment_tags else None),
        max_tags=max_tags,
    )

    # --- scene tokens -------------------------------------------------------
    # Normalisation frame. `drawer_w`/`drawer_d` are DD2D's spelling and are kept as
    # accepted aliases so every stored DD2D episode tensorizes byte-identically; a second
    # environment writes the generic `frame_w`/`frame_d` instead. An absent frame still
    # falls back to `scale = 1.0` -- unnormalised, which is what the older RT2D/kinder
    # records get and what SB2D would silently get if it wrote neither spelling.
    frame = geo.frame or {}
    # `drawer_w`/`drawer_d` are DD2D's spelling; `frame_w`/`frame_d` the generic one a
    # second environment writes. Require at least one: an absent frame used to fall back to
    # scale=1.0 *silently*, leaving obj_pose unnormalized and mixing units across
    # environments (cm on DD2D, m on SB2D). Fail loudly and name the fix instead.
    _fw = frame.get("drawer_w", frame.get("frame_w"))
    _fd = frame.get("drawer_d", frame.get("frame_d"))
    if _fw is None and _fd is None:
        raise ValueError(
            "scene_geometry.frame lacks a normalization extent: write frame_w/frame_d "
            "(DD2D uses drawer_w/drawer_d). Without it obj_pose is unnormalized "
            "(scale=1)."
        )
    scale = max(float(_fw or 0.0), float(_fd or 0.0), 1.0)
    # The goal channel is `is_goal` (any object named by the goal atoms), not `is_target`
    # (the one object a DD2D JSON flagged). `is_target` presupposes a single distinguished
    # target and is silently all-zero on an env whose goal names several objects (SB2D);
    # `is_goal` is well-defined for any goal, including N>1 targets, and is byte-identical
    # to `is_target` on every DD2D episode (proven 720/720). The target-anchored `obj_rel`
    # columns (dx, dy, dist to the target, area ratio to the target) and the privileged
    # `concave` flag are cut for the same reason -- only the three anchor-free per-object
    # scalars `[area, sinθ, cosθ]` remain. Absolute position is unaffected (it lives in
    # `obj_pose`). See docs/decisions 2026-08-08.
    goal_objs = spec.goal_objects(canon)

    n_obj = len(geo.objects)
    obj_tags = np.array([tags[o.name] for o in geo.objects], dtype=np.int64)
    obj_boundary = np.stack([resample_ring(list(o.boundary)) for o in geo.objects])
    obj_pose = np.array(
        [[o.pose[0] / scale, o.pose[1] / scale, o.pose[2]] for o in geo.objects],
        dtype=np.float32,
    )
    obj_is_goal = np.array(
        [1.0 if o.name in goal_objs else 0.0 for o in geo.objects], dtype=np.float32
    )
    rel = np.zeros((n_obj, D_REL_V3), dtype=np.float32)
    for i, o in enumerate(geo.objects):
        rel[i] = [o.area, *_sin_cos(o.pose[2])]

    # --- candidate tokens ---------------------------------------------------
    op_ids: list[list[int]] = []
    arg_tags: list[list[list[int]]] = []
    success: list[Optional[bool]] = []
    subsets: list[frozenset[str]] = []
    lengths: list[int] = []
    for skel, out in zip(canon.skeleton_pool, canon.outcomes):
        op_ids.append(
            [int(vocab.operators.get(op.name, 0)) for op in skel.operator_seq]
        )
        arg_tags.append(
            [[tags.get(a.name, 0) for a in op.parameters] for op in skel.operator_seq]
        )
        subsets.append(spec.manipulated(skel, goal_objs))
        lengths.append(spec.length_key(skel))
        success.append(out.outcome == "success")

    # Aux targets stay -1 (= ignore) until the necessity labeller lands: no collection
    # has ever populated `aux_labels`, so v2.2's aux head was masked out entirely and
    # never received a gradient. Pretending otherwise here would silently train it.
    aux = np.full(n_obj, -1.0, dtype=np.float32)

    # --- failure context ----------------------------------------------------
    k = len(op_ids)
    fail_idx = [i for i, out in enumerate(canon.outcomes) if out.outcome == "fail"]
    hide = hide_facts
    if context_f is not None:
        ctx: frozenset[int] = frozenset(int(i) for i in context_f)
    elif evidence and rng is not None:
        ctx, hide = sample_context(fail_idx, rng)
    else:
        ctx = frozenset()
    avail = [i not in ctx for i in range(k)]

    if hide or not ctx:
        ftype: list[int] = []
        ftier: list[int] = []
        farg: list[list[int]] = []
    else:
        ftype, ftier, farg = _hint_fact_arrays(canon, ctx, tags)

    # --- planner signals + structural evidence ------------------------------
    # Column 0 (enumeration order) is inert: the v3 scorer takes no prior features.
    # Column 1 carries the normalized plan length, kept only because the within-length
    # PL loss buckets on it; it is `domain.length_key` rescaled, so the partition is the
    # domain's.
    max_len = max(lengths) if lengths else 1
    prior = [[-(i / max(k - 1, 1)), -(lengths[i] / max(max_len, 1))] for i in range(k)]

    # `overlap_mode` zeroes a column rather than narrowing the tensor, so the state dict
    # shape is untouched and the D-8 exact-absence oracle keeps loading. A zeroed column
    # *is* the feature's absence: its weight receives no gradient signal from it.
    #
    # Dropping `dead` is a C5 argument, not a tuning knob. `dead` is the proof rule fed
    # to the net as a feature, and it is strongly anti-correlated with subset size
    # (corr −0.284; mean |S| 1.38 dead vs 2.39 alive on dd2d_v4 train), so the net can
    # fit it as "short ⇒ bad". That is sound only where the rule actually fired and is
    # L4's failure mode everywhere else -- which is why s1, the stratum on which short
    # *is* correct, regressed. The sound consequence still applies outside the net as the
    # demotion offset, where a wrong weight cannot override it.
    want_dead = overlap_mode in ("both", "dead")
    want_jac = overlap_mode in ("both", "jaccard")
    want_cov = coverage_feats
    # `coverage_mode` splits the pair the same way, and for the same reason: they answer
    # different questions -- `coverage` asks "does this candidate remove the objects the
    # refiner reported as blocking", `waste` asks "does it also remove objects that were
    # never implicated". They have only ever been measured together, so which one carries
    # the effect is unknown; zeroing one column isolates it without changing any shape.
    want_coverage = want_cov and coverage_mode in ("both", "coverage")
    want_waste = want_cov and coverage_mode in ("both", "waste")
    n_ov = N_OVERLAP_V3 if want_cov else 2
    overlap = [[0.0] * n_ov for _ in range(k)]
    culprits: frozenset = frozenset()
    _uni_records: list = []
    _uni_pool: frozenset = frozenset()
    _uni_universal: frozenset = frozenset()
    if ctx and not hide:
        blocked = [subsets[f] for f in ctx if spec.licenses_demotion(canon.outcomes[f])]
        failed = [subsets[f] for f in ctx]
        if want_cov and unified_coverage:
            # Lifted operators come from the pool's own `GroundOperator.parent`, so the
            # filters stay env-agnostic -- nothing here needs to know the domain.
            lifted = frozenset(
                op.parent for skel in canon.skeleton_pool for op in skel.operator_seq
            )
            _uni_universal, actionable = scene_filters(
                lifted, frozenset(canon.initial_abstract_state.objects)
            )
            _uni_records = records_from_failure_records(canon, ctx, spec)
            _uni_pool = frozenset(
                n
                for r in _uni_records
                for n in _unified_blame(r)
                if n in actionable and n not in _uni_universal
            )
        elif want_cov:
            _h, _p = records_for_evidence(canon, ctx, spec)
            culprits = frozenset(o for r in (_h + _p) for o in r.culprits)
        for i, si in enumerate(subsets):
            dead = 1.0 if any(si <= b for b in blocked) else 0.0
            jaccard = max(
                (len(si & f) / max(len(si | f), 1) for f in failed), default=0.0
            )
            row = [
                dead if want_dead else 0.0,
                float(jaccard) if want_jac else 0.0,
            ]
            if want_cov and unified_coverage:
                # The unified definitions (`unified_evidence.py`). Same two columns and
                # the same tensor shape, so the state dict is untouched -- what changes
                # is only how each scalar is computed. Deployed asks "is this culprit in
                # `args \ goal_objects`"; unified asks "does this candidate discharge the
                # culprit before re-entering the situation that named it", and computes
                # discretionary work from the candidate's own causal structure rather
                # than from a goal-object subtraction.
                _cov, _wst = coverage_and_waste(
                    list(canon.skeleton_pool[i].operator_seq),
                    _uni_records,
                    _uni_pool,
                    canon.initial_abstract_state.atoms,
                    canon.goal_atoms,
                    _uni_universal,
                )
                row += [
                    _cov if want_coverage else 0.0,
                    _wst if want_waste else 0.0,
                ]
            elif want_cov:
                # §5.1's `coverage` / `waste`, but grounded in **observed** culprits
                # instead of a predicted necessity head -- which makes them more
                # C2-legal, not less: nothing is inferred by us, the refiner reported
                # which objects blocked. This is the signal `dead` was crudely proxying
                # for. At s3 three distinct objects block, and the right candidate is
                # the one that removes all three; a length bias can only ever
                # approximate that.
                row += [
                    (
                        len(si & culprits) / max(len(culprits), 1)
                        if want_coverage
                        else 0.0
                    ),
                    (len(si - culprits) / max(len(si), 1) if want_waste else 0.0),
                ]
            overlap[i] = row

    records = (
        build_record_arrays(
            canon, ctx, tags, vocab, spec, aggregate_records, state_delta
        )
        if (ctx and not hide)
        else []
    )

    obj_evidence = None
    if ctx and not hide:
        _hint, _proof = records_for_evidence(canon, ctx, spec)
        obj_evidence = _object_evidence(
            ctx, subsets, [o.name for o in geo.objects], _hint, _proof, max_len=max_len
        )

    return (
        _V3Example(
            obj_tags,
            obj_boundary,
            obj_pose,
            rel,
            obj_is_goal,
            op_ids,
            arg_tags,
            success,
            aux,
            aux.copy(),
            avail,
            ftype,
            ftier,
            farg,
            prior,
            overlap,
            obj_evidence,
        ),
        records,
    )


def _sin_cos(theta: float) -> tuple[float, float]:
    return math.sin(theta), math.cos(theta)


def collate_v3(
    examples: list[_V2Example],
    max_arity: int,
    records: Optional[list[list[RecordArray]]] = None,
    max_pred_arity: int = 1,
) -> SpectreV3Batch:
    """Pad + stack examples into a batch the v3 model consumes.

    ``records`` is per-example and optional; without it the result is exactly a v2.2
    batch, which is what keeps the compat path (and the equivalence oracle) intact.
    """
    # `collate_v2` hard-codes a width-2 `cand_overlap` and D-7 freezes it, so a wider
    # example is stacked here instead. Narrow *copies* go to the v2 collator -- mutating
    # the caller's examples in place and restoring them afterwards would work today and
    # break the moment anything holds a reference.
    wide = max((len(e.overlap[0]) if e.overlap else 2) for e in examples)
    narrow = (
        [
            dataclasses.replace(e, overlap=[row[:2] for row in e.overlap])
            for e in examples
        ]
        if wide > 2
        else examples
    )
    base = collate_v2(narrow, max_arity=max_arity)
    batch = SpectreV3Batch(**base.__dict__)
    if wide > 2:
        b_, k_ = batch.pool_mask.shape
        ov_arr = np.zeros((b_, k_, wide), np.float32)
        for bi, e in enumerate(examples):
            for ki, row in enumerate(e.overlap[:k_]):
                ov_arr[bi, ki] = row
        batch.cand_overlap = torch.as_tensor(ov_arr)
    evs = [getattr(e, "obj_evidence", None) for e in examples]
    if any(e is not None for e in evs):
        n_obj = int(batch.obj_tags.shape[1])
        stacked = np.zeros((len(examples), n_obj, N_OBJ_EVIDENCE), np.float32)
        for bi, ev in enumerate(evs):
            if ev is not None:
                stacked[bi, : ev.shape[0]] = ev[:n_obj]
        batch.obj_evidence = torch.as_tensor(stacked)
    if not records or not any(records):
        return batch

    b = len(examples)
    r = max(len(rs) for rs in records)
    schema = np.zeros((b, r), np.int64)
    args = np.zeros((b, r, MAX_RECORD_ARGS), np.int64)
    culprits = np.zeros((b, r, MAX_RECORD_CULPRITS), np.int64)
    scalars = np.zeros((b, r, N_RECORD_SCALARS), np.float32)
    mask = np.zeros((b, r), bool)
    # Emitted on the *presence of the delta element*, never on whether any delta is
    # non-empty. Gating on non-emptiness would encode a j=0 record one way beside a
    # batch-mate that has a delta and another way alone -- and deploy collates a single
    # example per step, so both cases are routine (~48% of aggregated tokens are empty).
    wants_delta = any(len(row) > 4 for rs in records for row in rs)
    a = max(max_pred_arity, 1)
    d_pred = np.zeros((b, r, 2, MAX_DELTA_ATOMS), np.int64)
    d_args = np.zeros((b, r, 2, MAX_DELTA_ATOMS, a), np.int64)
    for bi, rs in enumerate(records):
        for ri, row in enumerate(rs):
            sid, ar, cu, sc = row[:4]
            schema[bi, ri] = sid
            args[bi, ri, : len(ar)] = ar
            culprits[bi, ri, : len(cu)] = cu
            scalars[bi, ri] = sc
            mask[bi, ri] = True
            if not wants_delta or len(row) <= 4:
                continue
            for role, atoms in enumerate(row[4]):
                for ai, (pid, atags) in enumerate(atoms[:MAX_DELTA_ATOMS]):
                    d_pred[bi, ri, role, ai] = pid
                    d_args[bi, ri, role, ai, : len(atags)] = atags[:a]

    t = torch.as_tensor
    batch.rec_schema_ids = t(schema)
    batch.rec_arg_tags = t(args)
    batch.rec_culprit_tags = t(culprits)
    batch.rec_scalars = t(scalars)
    batch.rec_mask = t(mask)
    if wants_delta:
        batch.rec_delta_pred_ids = t(d_pred)
        batch.rec_delta_arg_tags = t(d_args)
    return batch
