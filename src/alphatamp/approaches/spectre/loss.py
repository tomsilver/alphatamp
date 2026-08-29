"""Plackett-Luce listwise loss for SPECTRE training.

Per ``docs/archive/SPECTRE_RT2D_METHOD_SPEC.md`` §8.3 / §10.4. The top-1 PL formulation
``ℒ = −log(Z_+ / Z)`` is rollout-aligned: it is the negative log-probability
that an argmax-pick over R lands on a successful skeleton, which is the
training-time analog of the time-to-first-success metric.

Computed entirely in log-space via :func:`torch.logsumexp` for numerical
stability across `R` sizes up to ~30 and logit magnitudes up to ~1e6 (the
saturation case used by the smoke tests).
"""

from __future__ import annotations

import torch


def plackett_luce_loss(
    logits: torch.Tensor,  # (B, R)
    success_mask: torch.Tensor,  # (B, R)  bool
    pool_mask: torch.Tensor,  # (B, R)  bool
) -> torch.Tensor:
    """Listwise top-1 Plackett-Luce loss, mean over the batch.

    ``pool_mask`` is True for valid R-pool slots; pad slots get -inf logits.
    ``success_mask`` is True for R-pool slots that succeeded in the episode. Examples
    with ``|SUCC_R| == 0`` are filtered upstream by ``SpectreDataset``; this function
    does not re-check the precondition.
    """
    neg_inf = torch.tensor(-float("inf"), dtype=logits.dtype, device=logits.device)
    masked = torch.where(pool_mask, logits, neg_inf)
    z = torch.logsumexp(masked, dim=-1)
    succ_logits = torch.where(success_mask & pool_mask, masked, neg_inf)
    z_plus = torch.logsumexp(succ_logits, dim=-1)
    return -(z_plus - z).mean()


def within_length_pl_loss(
    logits: torch.Tensor,  # (B, K)  — already availability-masked (-inf for tried)
    success_mask: torch.Tensor,  # (B, K)  bool
    pool_mask: torch.Tensor,  # (B, K)  bool
    length_key: torch.Tensor,  # (B, K)  float/int — one value per plan length
) -> torch.Tensor:
    """Top-1 PL loss computed **within each plan-length bucket**, averaged over buckets.

    The global PL loss can be minimized by a length shortcut (on hard problems only long
    plans succeed, so "prefer long" satisfies it) — which is right across lengths but
    leaves the ranker near-chance *within* a length, where feasibility is decided by
    geometry, not count. Restricting the listwise objective to same-length candidates
    removes length as an exploitable cue and forces the geometry signal at every
    stratum. Plan length is a universal plan property, so this is domain-agnostic.
    """
    neg_inf = torch.tensor(-float("inf"), dtype=logits.dtype, device=logits.device)
    key = (length_key * 1000.0).round()  # stable bucket id within each episode
    total = logits.new_zeros(())
    n = 0
    for b in range(logits.shape[0]):
        valid = pool_mask[b]
        if not valid.any():
            continue
        for lv in torch.unique(key[b][valid]):
            m = valid & (key[b] == lv)
            if int(m.sum()) < 2:
                continue
            succ_m = m & success_mask[b]
            if not bool(succ_m.any()):
                continue
            bl = torch.where(m, logits[b], neg_inf)
            z = torch.logsumexp(bl, dim=-1)
            z_plus = torch.logsumexp(torch.where(succ_m, logits[b], neg_inf), dim=-1)
            total = total + (z - z_plus)
            n += 1
    return total / max(n, 1)
