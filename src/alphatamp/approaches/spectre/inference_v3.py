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
from typing import Optional

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset_v3 import build_v3_example, collate_v3
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.failure_record import records_for_candidate
from alphatamp.approaches.spectre.model_v3 import SpectreV3Model
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


@dataclasses.dataclass(frozen=True)
class V3Trace:
    """Step-aligned record of one rollout; one entry per attempt made.

    ``step_scores`` are the **raw** model logits, before the tried-mask and before the
    demotion offset. Raw on purpose: those sentinels would swamp a rendered score column,
    and the effective row is exactly reconstructible from ``step_dead`` and ``order``.
    Entries for candidates already in the failure context come back ``-inf`` from the
    model's own availability mask, so at step ``t`` the non-finite entries are exactly
    ``order[:t]``; a JSON serialiser must map them to ``null``.
    """

    order: list[int]
    step_scores: list[list[float]]
    step_dead: list[list[int]]


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
    apply_demotion: bool = True,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
    coverage_feats: bool = False,
    suppress_records: bool = False,
) -> tuple[int, V3Trace]:
    """Run the deployed ranker; return ``(attempts_to_first_success, trace)``.

    ``attempts`` is 1-indexed (the rollout FP reported downstream is ``attempts - 1``).
    ``spec`` defaults to the contract registered for the episode's own ``env_variant``.

    ``max_attempts`` censors the rollout at a fixed budget. Reporting always runs
    uncensored -- the budget equals the pool cap, so it never binds -- and censoring exists
    only for *checkpoint selection*, where the metric is recomputed every epoch and the
    full loop otherwise costs several times the training step it is selecting over. This
    mirrors the split the project already runs: selection under a budget, reporting
    without one.

    ``mode`` selects how much exactness evidence a failure must carry before it licenses
    demotion. ``strict`` (default) requires positive evidence that the query ran to
    exhaustion, which is what keeps the deduction sound; ``permissive`` reproduces v2.2's
    semantics and exists so the two can be compared candidate-for-candidate.

    ``apply_demotion=False`` runs the ranker with the proof-demotion offset withheld, so
    the model's own ordering is what gets measured. This is deliberately *not* a third
    ``DemotionMode``: the modes say how much evidence licenses a sound deduction, whereas
    this says whether to act on the deduction at all. It exists for the G7 2x2, which
    crosses it with the net's ``[dead, jaccard]`` features to ask whether the learned
    ``dead`` column is redundant with the rule applied outside the net. The proof state is
    still advanced either way, so the trace's ``step_dead`` stays populated and the two
    arms differ only in whether the offset is applied.

    ``suppress_records=True`` is a **diagnostic**, not a deployment mode: it runs a
    records-trained model with its evidence memory emptied at every step. Deliberately a
    train/deploy mismatch, and useful precisely because of that -- it separates "training
    with records damaged the weights" (still bad with records suppressed) from "the
    evidence input misleads at deploy" (good with them suppressed). Never report a number
    produced with it as a method result.

    There is no ``demotion_source`` knob: v3 reads the refiner's own report, and the
    geometry-reconstruction alternative is not ported (R2).
    """
    model.eval()
    spec = spec or spec_for(episode.provenance.env_variant)
    n_candidates = len(episode.skeleton_pool)
    success = {i for i, o in enumerate(episode.outcomes) if o.outcome == "success"}
    state = ProofStateV3(candidate_queries(episode, spec), spec, mode=mode)
    tried: list[int] = []
    step_scores: list[list[float]] = []
    step_dead: list[list[int]] = []

    budget = n_candidates if max_attempts is None else min(max_attempts, n_candidates)
    while len(tried) < budget:
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
        )
        # Records are passed at deployment too, not just in training. Omitting them here
        # would deploy a records-trained model blind to its own evidence -- the train/
        # deploy input mismatch the proposal warns about, and one that degrades silently.
        batch = collate_v3(
            [example],
            max_arity=vocab.max_operator_arity,
            records=[[] if suppress_records else records],
        ).to(device)
        logits, _ = model(batch)
        raw = logits[0].detach().cpu().numpy().astype(float)
        step_scores.append([float(x) for x in raw])
        step_dead.append(sorted(int(i) for i in state.dead))

        row = raw.copy()
        if tried:
            row[tried] = _TRIED
        if apply_demotion:
            row = demote_scores(row, state.dead)
        pick = int(np.argmax(row))
        tried.append(pick)
        if pick in success:
            break
        state.observe(records_for_candidate(episode, pick, spec))

    return len(tried), V3Trace(
        order=list(tried), step_scores=step_scores, step_dead=step_dead
    )


def deployed_rollout_v3(
    model: SpectreV3Model,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    spec: Optional[DomainSpec] = None,
    max_tags: int = 32,
) -> int:
    """Attempts to first success. See :func:`deployed_rollout_v3_traced` for the trace."""
    attempts, _ = deployed_rollout_v3_traced(
        model, episode, vocab, device, spec, max_tags
    )
    return attempts
