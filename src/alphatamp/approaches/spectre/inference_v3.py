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
from alphatamp.approaches.spectre.model_v3 import SpectreV3Model
from alphatamp.approaches.spectre.proof_demotion import ProofState, demote_scores
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
) -> tuple[int, V3Trace]:
    """Run the deployed ranker; return ``(attempts_to_first_success, trace)``.

    ``attempts`` is 1-indexed (the rollout FP reported downstream is ``attempts - 1``).
    ``spec`` defaults to the contract registered for the episode's own ``env_variant``.

    There is no ``demotion_source`` knob: v3 reads the refiner's own report, and the
    geometry-reconstruction alternative is not ported (R2).
    """
    model.eval()
    spec = spec or spec_for(episode.provenance.env_variant)
    subsets = spec.subsets(episode)
    success = {i for i, o in enumerate(episode.outcomes) if o.outcome == "success"}
    state = ProofState(subsets=subsets)
    tried: list[int] = []
    step_scores: list[list[float]] = []
    step_dead: list[list[int]] = []

    while len(tried) < len(subsets):
        example = build_v3_example(
            episode,
            vocab,
            rng=None,
            max_tags=max_tags,
            evidence=True,
            context_f=frozenset(tried),
            augment_tags=False,
            spec=spec,
        )
        batch = collate_v3([example], max_arity=vocab.max_operator_arity).to(device)
        logits, _ = model(batch)
        raw = logits[0].detach().cpu().numpy().astype(float)
        step_scores.append([float(x) for x in raw])
        step_dead.append(sorted(int(i) for i in state.dead))

        row = raw.copy()
        if tried:
            row[tried] = _TRIED
        row = demote_scores(row, state.dead)
        pick = int(np.argmax(row))
        tried.append(pick)
        if pick in success:
            break
        if spec.licenses_demotion(episode.outcomes[pick]):
            state.observe_failure(pick, blocked=True, pack_impossible=False)

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
