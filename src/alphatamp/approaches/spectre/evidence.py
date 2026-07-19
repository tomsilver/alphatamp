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
