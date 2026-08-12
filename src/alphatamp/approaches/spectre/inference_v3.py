"""The v3 deployed ranker: model scores filtered by sound proof-demotion.

Mirrors ``evidence.deployed_rollout_traced`` (v2.2) so the two can be compared decision
for decision. The loop is the deployment story in six lines: score the pool with the
failures observed so far, push provably-dead candidates to the back, try the argmax, stop
on the first success.

Two properties are invariants, not implementation details:

- **Demotion reorders, it never removes** (P-E). A demoted candidate loses a finite
  offset, so if every candidate is proven dead they are still all attemptable, in order.
  A wrong proof therefore costs attempts; it cannot lose the feasible plan.
- **The net never sees a proof.** Sound consequences are applied here, outside the
  network, so no learned weight can override them and no proof can corrupt the
  representation.

The per-step trace exists because the comparison cache stores it: persisting raw logits
plus the demoted set lets the analysis notebook show what the ranker thought at every
step without ever running inference at load time.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset_v3 import build_v3_example, collate_v3
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.failure_record import records_for_candidate
from alphatamp.approaches.spectre.model_v2 import D_REL_V3
from alphatamp.approaches.spectre.model_v3 import (
    N_OVERLAP_V3,
    SpectreV3Model,
    V3Config,
)
from alphatamp.approaches.spectre.proof_demotion import demote_scores
from alphatamp.approaches.spectre.proof_demotion_v3 import (
    DemotionMode,
    ProofStateV3,
    candidate_queries,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.vocab import Vocab

# Sentinel applied to already-attempted candidates before the argmax. Distinct from the
# demotion offset so the two effects stay legible in a trace.
_TRIED = -1e9


def _zero_scene_columns(batch, cols: frozenset[str]):
    """Zero a whole scene channel in place -- a deploy-time diagnostic.

    Mirrors ``suppress_records``: it feeds a trained model a *null* version of an input it
    was trained on, to price how much the deployed model leans on that channel.
    ``"is_goal"`` blanks the goal-membership boolean; ``"rel"`` blanks the anchor-free
    ``obj_rel`` triple ``[area, sinθ, cosθ]``. Not a deployment mode: it only measures
    reliance. Batch tensors are rebuilt every step, so mutating them never leaks across
    steps. (An earlier form of this hook, tied to the pre-narrowing width-8 ``obj_rel``,
    priced the removal of the target-anchored columns before they were cut -- see the
    Step-0 measurement in docs/notebook 2026-08-08.)
    """
    if "is_goal" in cols:
        batch.obj_is_goal.zero_()
    if "rel" in cols:
        batch.obj_rel.zero_()
    return batch


def load_v3_checkpoint(
    ckpt: Path | str, vocab: Vocab, device: str = "cpu"
) -> tuple[SpectreV3Model, dict]:
    """Rebuild a trained v3 model, with dropout off, plus its **deploy kwargs**.

    The second return value is the set of feature switches that change what
    :func:`build_v3_example` *emits* rather than what the model *contains*, so they are
    invisible to ``load_state_dict`` and a mismatch fails silently instead of loudly:
    deploying under a different ``overlap_mode`` (or ``coverage_mode``) than a model
    trained under feeds it a column it has never seen populated, or blanks one it relies
    on. Reading them back off the checkpoint — never accepting them from the caller — is
    what makes that unrepresentable. Splat the dict into
    :func:`deployed_rollout_v3_traced` / :func:`build_v3_example`.

    Switches that *do* change the architecture (``use_records``, ``evidence_attn``,
    ``use_obj_evidence``, ``sinusoidal_pos``, and ``coverage_feats`` via the
    ``cand_overlap`` width) are rebuilt into ``V3Config`` here, where ``strict=True``
    catches any error. Older checkpoints predate several of these keys, hence ``.get``.
    """
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = SpectreV3Model(
        n_ops=int(ck["n_ops"]),
        max_arity=vocab.max_operator_arity,
        cfg=V3Config(
            n_overlap_feats=(
                (N_OVERLAP_V3 if cfg.get("coverage_feats") else 2)
                if cfg.get("use_overlap")
                else 0
            ),
            n_prior_feats=0,
            # Scene-relation width is bound to the checkpoint: deployed v3 is 3, and a
            # checkpoint predating the narrowing has no key and was 8-wide. `strict=True`
            # below is the backstop -- a wrong width fails to load rather than silently
            # scoring the un-narrowed model.
            d_rel=int(cfg.get("d_rel", D_REL_V3)),
            max_tags=int(cfg.get("max_tags", 32)),
            dropout_p=0.0,
            use_records=bool(cfg.get("use_records")),
            sinusoidal_pos=bool(cfg.get("sinusoidal_pos")),
            use_obj_evidence=bool(cfg.get("use_obj_evidence")),
            evidence_attn=bool(cfg.get("evidence_attn")),
            coverage_feats=bool(cfg.get("coverage_feats")),
            use_state_delta=bool(cfg.get("use_state_delta")),
            n_predicates=len(vocab.predicates),
            max_pred_arity=vocab.max_predicate_arity,
        ),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval().to(device), {
        "overlap_mode": str(cfg.get("overlap_mode", "both")),
        "aggregate_records": bool(cfg.get("aggregate_records")),
        "coverage_feats": bool(cfg.get("coverage_feats")),
        "coverage_mode": str(cfg.get("coverage_mode", "both")),
        # Absent key => False, and that is load-bearing rather than incidental: every
        # checkpoint trained before 2026-07-31 was trained on the deployed
        # `S(c) = args \ goal_objects` features and must keep being scored on them, even
        # though unified is now the default for new runs. The checkpoint decides, not
        # the current default.
        "unified_coverage": bool(cfg.get("unified_coverage")),
        # Architectural *and* emitted: the encoder needs the submodules and the
        # tensorizer
        # needs to produce the arrays, so it appears in both places -- exactly as
        # `coverage_feats` does -- with the checkpoint as the single source of truth.
        "state_delta": bool(cfg.get("use_state_delta")),
    }


@dataclasses.dataclass(frozen=True)
class V3Trace:
    """Step-aligned record of one rollout; one entry per attempt made.

    ``step_scores`` are the **raw** model logits, before the tried-mask and before the
    demotion offset. Raw on purpose: those sentinels would swamp a rendered score
    column, and the effective row is exactly reconstructible from ``step_dead`` and
    ``order``. Entries for candidates already in the failure context come back ``-inf``
    from the model's own availability mask, so at step ``t`` the non-finite entries are
    exactly ``order[:t]``; a JSON serialiser must map them to ``null``.
    """

    order: list[int]
    step_scores: list[list[float]]
    step_dead: list[list[int]]
    infer_seconds: float = 0.0
    """Wall-clock spent on inference across the rollout: per-step tensorization
    (``build_v3_example`` + ``collate_v3``) + the model forward, summed.

    Defaulted so callers that ignore timing are unaffected; the timing bracket includes
    the device sync, so on cuda it is a true end-to-end measure. Warm the model up once
    before a timed pass so one-time CUDA init does not land in the first step.
    """

    refine_capped_seconds: float = 0.0
    """Refinement wall-clock along the realized order, each candidate's stored
    ``refinement_wall_clock_s`` clamped to ``refine_cap_s`` (uncapped sum when no cap).

    Reuses the per-candidate refiner times stored on the episode; the rollout must
    accumulate it here (rather than a caller summing ``_refine_seconds`` over ``order``)
    because a capped rollout's order can contain a *slow-feasible* candidate that did
    not stop the loop -- a plain "sum to first success" would break there and
    undercount. 0.0 when the episode carries no per-candidate times.
    """


@torch.no_grad()
def deployed_rollout_v3_traced(
    model: SpectreV3Model,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    spec: Optional[DomainSpec] = None,
    max_tags: int = 32,
    mode: DemotionMode = "strict",
    max_attempts: Optional[int] = None,
    apply_demotion: bool = False,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
    coverage_feats: bool = False,
    coverage_mode: str = "both",
    unified_coverage: bool = False,
    state_delta: bool = False,
    suppress_records: bool = False,
    zero_scene_cols: frozenset[str] = frozenset(),
    refine_cap_s: Optional[float] = None,
) -> tuple[int, V3Trace]:
    """Run the deployed ranker; return ``(attempts_to_first_success, trace)``.

    ``attempts`` is 1-indexed (the rollout FP reported downstream is ``attempts - 1``).
    ``spec`` defaults to the contract registered for the episode's own ``env_variant``.

    ``max_attempts`` censors the rollout at a fixed budget. Reporting always runs
    uncensored -- the budget equals the pool cap, so it never binds -- and censoring
    exists only for *checkpoint selection*, where the metric is recomputed every epoch
    and the full loop otherwise costs several times the training step it is selecting
    over. This mirrors the split the project already runs: selection under a budget,
    reporting without one.

    ``apply_demotion`` defaults to **False**: proof-tier demotion was cut from the
    deployed method on 2026-07-30 (``decisions.md``). v3 is now a purely learned ranker
    -- nothing outside the network touches the ordering -- which is what the deployed
    numbers report. It costs a measured 0.23 FP (7.20 -> 7.44, CI [+0.08, +0.43]) and
    buys a system with one kind of component in it.

    Passing ``True`` re-enables the finite offset and is still fully supported: the proof
    machinery (``ProofStateV3``, the axiom registry, ``strict``/``permissive``) is kept,
    tested and correct, because the deduction is sound and a domain where proofs fire
    more often than DD2D's 6% may well want it. It is *off by default*, not removed.

    ``mode`` selects how much exactness evidence a failure must carry before it licenses
    demotion, and is therefore only consulted when ``apply_demotion=True``. ``strict``
    requires positive evidence that the query ran to exhaustion, which is what keeps the
    deduction sound; ``permissive`` reproduces v2.2's semantics.

    The proof state is advanced either way, so ``V3Trace.step_dead`` stays populated even
    with demotion off -- it then reads as "what a proof *would* have demoted", which is
    what the planner inspector shows.

    ``suppress_records=True`` is a **diagnostic**, not a deployment mode: it runs a
    records-trained model with its evidence memory emptied at every step. Deliberately a
    train/deploy mismatch, and useful precisely because of that -- it separates "training
    with records damaged the weights" (still bad with records suppressed) from "the
    evidence input misleads at deploy" (good with them suppressed). Never report a number
    produced with it as a method result.

    ``zero_scene_cols`` is the geometry analogue of ``suppress_records`` and is likewise a
    **diagnostic**: it blanks a scene channel at deploy to price how much the model leans
    on it. ``"is_goal"`` blanks the goal-membership boolean; ``"rel"`` blanks the
    anchor-free ``obj_rel`` triple. A small FP delta means the channel is close to inert
    for ranking; a large one means it is load-bearing. Never a method number.

    There is no ``demotion_source`` knob: v3 reads the refiner's own report, and the
    geometry-reconstruction alternative is not ported (R2).

    ``refine_cap_s`` models a **per-candidate refinement-abandonment cap**: a deployment
    that bounds each skeleton's refinement at ``refine_cap_s`` seconds before moving on.
    A feasible candidate whose stored ``refinement_wall_clock_s`` exceeds the cap is
    then *not* a stopping success -- it is abandoned and observed like any other
    failure (so it enters the failure context and re-ranks the pool), and the loop
    continues. This only reorders the *ranking*, never removes a plan (P-E holds): the
    pool is still exhausted in order, so a problem is lost only if every feasible
    candidate exceeds the cap. ``V3Trace.refine_capped_seconds`` accumulates the
    wall-clock the capped deployment pays. ``None`` (default) is the uncapped rollout.
    """
    model.eval()
    spec = spec or spec_for(episode.provenance.env_variant)
    n_candidates = len(episode.skeleton_pool)

    def _stops(o) -> bool:
        # A candidate ends the rollout only if it refines *and* does so within the cap;
        # a slow-feasible candidate over the cap is abandoned and treated as a failure.
        if o.outcome != "success":
            return False
        if refine_cap_s is None:
            return True
        return float(o.refinement_wall_clock_s or 0.0) <= refine_cap_s

    success = {i for i, o in enumerate(episode.outcomes) if _stops(o)}
    state = ProofStateV3(candidate_queries(episode, spec), spec, mode=mode)
    tried: list[int] = []
    step_scores: list[list[float]] = []
    step_dead: list[list[int]] = []
    infer_seconds = 0.0
    refine_capped_seconds = 0.0

    budget = n_candidates if max_attempts is None else min(max_attempts, n_candidates)
    while len(tried) < budget:
        _t_infer = time.perf_counter()
        example, records = build_v3_example(
            episode,
            vocab,
            rng=None,
            max_tags=max_tags,
            evidence=True,
            context_f=frozenset(tried),
            augment_tags=False,
            spec=spec,
            overlap_mode=overlap_mode,
            aggregate_records=aggregate_records,
            coverage_feats=coverage_feats,
            coverage_mode=coverage_mode,
            unified_coverage=unified_coverage,
            state_delta=state_delta,
        )
        # Records are passed at deployment too, not just in training. Omitting them here
        # would deploy a records-trained model blind to its own evidence -- the train/
        # deploy input mismatch the proposal warns about, and one that degrades silently.
        batch = collate_v3(
            [example],
            max_arity=vocab.max_operator_arity,
            records=[[] if suppress_records else records],
            max_pred_arity=vocab.max_predicate_arity,
        ).to(device)
        if zero_scene_cols:
            batch = _zero_scene_columns(batch, zero_scene_cols)
        logits, _ = model(batch)
        raw = logits[0].detach().cpu().numpy().astype(float)
        # end-to-end inference time: tensorize + collate + forward. The .cpu() above
        # already forces a device sync; synchronize() is a defensive no-op making the
        # bracket a true wall-clock even if that copy is ever removed.
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        infer_seconds += time.perf_counter() - _t_infer
        step_scores.append([float(x) for x in raw])
        step_dead.append(sorted(int(i) for i in state.dead))

        row = raw.copy()
        if tried:
            row[tried] = _TRIED
        if apply_demotion:
            row = demote_scores(row, state.dead)
        pick = int(np.argmax(row))
        tried.append(pick)
        _t_pick = float(episode.outcomes[pick].refinement_wall_clock_s or 0.0)
        refine_capped_seconds += (
            _t_pick if refine_cap_s is None else min(_t_pick, refine_cap_s)
        )
        if pick in success:
            break
        state.observe(records_for_candidate(episode, pick, spec))

    return len(tried), V3Trace(
        order=list(tried),
        step_scores=step_scores,
        step_dead=step_dead,
        infer_seconds=infer_seconds,
        refine_capped_seconds=round(refine_capped_seconds, 6),
    )


def deployed_rollout_v3(
    model: SpectreV3Model,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    spec: Optional[DomainSpec] = None,
    max_tags: int = 32,
) -> int:
    """Attempts to first success.

    See :func:`deployed_rollout_v3_traced` for the trace.
    """
    attempts, _ = deployed_rollout_v3_traced(
        model, episode, vocab, device, spec, max_tags
    )
    return attempts
