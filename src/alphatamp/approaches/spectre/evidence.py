"""Step-11 evidence-pathway runtime: the live scramble gauge + the typed-evidence
rollout.

Two things the static path never needs:

- **Scramble gauge** — permute the *identity* of every fact's object arguments (a fact
  ``extraction-failed(item_4)`` becomes ``extraction-failed(item_j)`` for a random other
  ``item_j`` present in the scene) while keeping the fact *type*, *tier* and *count* fixed.
  A ranker that only counts failures is invariant to this; one that reads fact identity is
  not. The gauge is the mean per-candidate logit shift under scrambling — a training-time
  detector that the shared-parameter path actually uses facts (never used as a loss).
- **Evidence rollout** — the deployed loop: score the pool at ``F=∅``, try the argmax, and
  on failure add it to ``F``, re-gather its facts, and re-score, until a success. Run with
  facts on vs off, the difference is the *evidence increment* (P5).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset_v2 import build_v2_example, collate_v2
from alphatamp.approaches.spectre.model_v2 import SpectreV2Batch, SpectreV2Model
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.vocab import Vocab


def scramble_fact_identities(
    batch: SpectreV2Batch, generator: np.random.Generator
) -> SpectreV2Batch:
    """Return a copy of ``batch`` with each example's fact argument tags remapped
    through a random permutation of that example's present object tags (identity
    scrambled, structure kept).

    A no-op when the batch carries no facts.
    """
    if batch.fact_arg_tags is None or batch.fact_type_ids is None:
        return batch
    arg = batch.fact_arg_tags.clone()
    obj_tags = batch.obj_tags
    obj_mask = batch.obj_mask
    b = arg.shape[0]
    for bi in range(b):
        present = torch.unique(obj_tags[bi][obj_mask[bi]])
        present = present[present != 0]
        if present.numel() < 2:
            continue
        perm = present[torch.as_tensor(generator.permutation(present.numel()))]
        remap = {int(a): int(c) for a, c in zip(present.tolist(), perm.tolist())}
        slab = arg[bi]
        for a, c in remap.items():
            slab[slab == a] = -c  # negative marker avoids chained remaps
        arg[bi] = slab.abs()
    return dataclasses.replace(batch, fact_arg_tags=arg)


@torch.no_grad()
def scramble_gauge(
    model: SpectreV2Model,
    batch: SpectreV2Batch,
    device: str,
    generator: np.random.Generator,
) -> float:
    """Mean |Δ logit| over available candidates between real and identity-scrambled
    facts.

    ~0 when the ranker ignores fact identity; grows as it learns to use it.
    """
    if batch.fact_type_ids is None or int(batch.fact_mask.sum()) == 0:  # type: ignore[union-attr]
        return 0.0
    model.eval()
    real = batch.to(device)
    scr = scramble_fact_identities(batch, generator).to(device)
    lr, _ = model(real)
    ls, _ = model(scr)
    avail = real.avail_mask if real.avail_mask is not None else real.pool_mask
    diff = (lr - ls).abs()
    diff = diff[torch.isfinite(diff) & avail]
    return float(diff.mean().item()) if diff.numel() else 0.0


@torch.no_grad()
def evidence_rollout(
    model: SpectreV2Model,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    use_facts: bool = True,
    max_tags: int = 32,
) -> int:
    """Deployed rollout FP: try the argmax, on failure grow ``F`` and re-score, until a
    success.

    ``use_facts=False`` hides evidence (the static pathway) for the increment.
    """
    model.eval()
    success = {i for i, o in enumerate(episode.outcomes) if o.outcome == "success"}
    k = len(episode.skeleton_pool)
    tried: set[int] = set()
    fp = 0
    while len(tried) < k:
        ctx = frozenset(tried)
        ex = build_v2_example(
            episode,
            vocab,
            rng=None,
            max_tags=max_tags,
            evidence=True,
            context_f=ctx,
            hide_facts=not use_facts,
            augment_tags=False,
        )
        batch = collate_v2([ex], max_arity=vocab.max_operator_arity).to(device)
        logits, _ = model(batch)
        row = logits[0].clone()
        row[list(tried)] = float("-inf")
        pick = int(torch.argmax(row).item())
        if pick in success:
            return fp
        tried.add(pick)
        fp += 1
    return fp


def observed_blocked(outcome, demotion_source: str) -> bool:
    """Whether a *failed* candidate is blocked-at-contents, per the demotion signal.

    ``observed`` (default) reads the refiner's own failure (``failure_action`` reached
    ``retrieve`` ⇒ all removals ran and the target was still ungraspable) — no geometry.
    ``computed`` reads the harvested geometry fact (adds counterfactual demotions).
    """
    if demotion_source == "computed":
        pm = outcome.post_mortem
        return pm is not None and any(
            f.fact_type == "blocked-at-contents" for f in pm.facts
        )
    return str((outcome.refiner_metadata or {}).get("failure_action", "")).startswith(
        "retrieve"
    )


@dataclasses.dataclass(frozen=True)
class DeployedTrace:
    """Per-step record of one :func:`deployed_rollout_traced` rollout.

    All three lists are step-aligned and have one entry per *attempt* made.

    - ``order`` — the realized sequence of attempted pool indices (ends at the first
      success, or at pool exhaustion).
    - ``step_scores`` — the **raw** ``(K,)`` model logits at each step, *before* this
      function's already-tried mask and *before* the proof-demotion offset. Raw on
      purpose: those sentinels (``-1e9`` / ``-1e6``) would swamp the column when this is
      rendered, and the effective row is exactly reconstructible from ``step_dead`` +
      ``order``. Note the *model* still masks its own context: entries for candidates
      already in ``F`` come back ``-inf`` (the batch ``avail_mask``), so at step ``t``
      the non-finite entries are exactly ``order[:t]``. Serialisers must map those to
      ``null`` — ``-inf`` is not representable in strict JSON.
    - ``step_dead`` — the sorted provably-dead indices in force at that step, i.e. the
      candidates proof-demotion had already ruled out when the pick was made.
    """

    order: list[int]
    step_scores: list[list[float]]
    step_dead: list[list[int]]


@torch.no_grad()
def deployed_rollout_traced(
    model: SpectreV2Model,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    demotion_source: str = "observed",
    max_tags: int = 32,
) -> tuple[int, DeployedTrace]:
    """:func:`deployed_rollout` plus the per-step :class:`DeployedTrace`.

    Same loop and the same attempts count — the trace is recorded alongside, not
    re-derived — mirroring the ``spectre_evaluate`` / ``spectre_evaluate_traced`` split
    in ``eda.py``. The comparison cache persists the trace so the notebook's planner
    inspector can show what the adaptive ranker thought at each step without ever
    running inference itself.
    """
    from alphatamp.approaches.spectre.proof_demotion import ProofState, demote_scores

    model.eval()
    subsets = [
        frozenset(
            op.parameters[0].name for op in s.operator_seq if op.name == "place-buffer"
        )
        for s in episode.skeleton_pool
    ]
    success = {i for i, o in enumerate(episode.outcomes) if o.outcome == "success"}
    state = ProofState(subsets=subsets)
    tried: list[int] = []
    step_scores: list[list[float]] = []
    step_dead: list[list[int]] = []
    while len(tried) < len(subsets):
        ex = build_v2_example(
            episode,
            vocab,
            rng=None,
            max_tags=max_tags,
            evidence=True,
            context_f=frozenset(tried),
            augment_tags=False,
            demotion_source=demotion_source,
        )
        batch = collate_v2([ex], max_arity=vocab.max_operator_arity).to(device)
        logits, _ = model(batch)
        raw = logits[0].detach().cpu().numpy().astype(float)
        step_scores.append([float(x) for x in raw])
        step_dead.append(sorted(int(i) for i in state.dead))
        row = raw.copy()
        if tried:
            row[tried] = -1e9
        row = demote_scores(row, state.dead)
        pick = int(row.argmax())
        tried.append(pick)
        if pick in success:
            break
        if observed_blocked(episode.outcomes[pick], demotion_source):
            state.observe_failure(pick, blocked=True, pack_impossible=False)
    return len(tried), DeployedTrace(
        order=list(tried), step_scores=step_scores, step_dead=step_dead
    )


def deployed_rollout(
    model: SpectreV2Model,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    demotion_source: str = "observed",
    max_tags: int = 32,
) -> int:
    """The full deployed ranker: model scores + the sound proof-demotion filter.

    Step 10. Each step scores the pool (facts on), pushes provably-dead candidates
    (subset ⊆ an observed-blocked set) to the back, tries the argmax, and on a blocked
    failure updates the demotion state. Returns attempts-to-first-success (1-indexed).
    The demotion is applied *outside* the network, so it can only reorder — never lose
    the feasible plan.

    Use :func:`deployed_rollout_traced` when the per-step trace is wanted too.
    """
    attempts, _ = deployed_rollout_traced(
        model,
        episode,
        vocab,
        device,
        demotion_source=demotion_source,
        max_tags=max_tags,
    )
    return attempts
