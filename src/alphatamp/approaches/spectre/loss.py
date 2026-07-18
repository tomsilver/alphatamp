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
    ``success_mask`` is True for R-pool slots that succeeded in the episode.
    Examples with ``|SUCC_R| == 0`` are filtered upstream by
    ``SpectreDataset``; this function does not re-check the precondition.
    """
    neg_inf = torch.tensor(-float("inf"), dtype=logits.dtype, device=logits.device)
    masked = torch.where(pool_mask, logits, neg_inf)
    z = torch.logsumexp(masked, dim=-1)
    succ_logits = torch.where(success_mask & pool_mask, masked, neg_inf)
    z_plus = torch.logsumexp(succ_logits, dim=-1)
    return -(z_plus - z).mean()
