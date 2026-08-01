"""PIGINet training losses (Step 7 of docs/piginet_dd2d_plan.md).

Three arms the imbalance gate compares on the DD2D 50:1 class imbalance:

* :func:`weighted_bce` — paper baseline (§IV-C): BCE with ``pos_weight = N_neg/N_pos``.
* :func:`focal_loss` — down-weights the many *easy* negatives (buried-blocker / obvious
  buffer-overflow plans fail fast), focusing on the confusable ones.
* :func:`listwise_ranking_loss` — per problem-group, ``-log softmax(logits)[positive]``; the
  feasible plan should score highest among its problem's plans. Deployment-aligned (PIGINet
  *ranks* within a problem) and inherently imbalance-robust (per-problem relative).

All take raw logits (sigmoid/softmax applied inside) for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), pos_weight=pw
    )


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """Binary focal loss (Lin et al.

    2017). ``alpha`` optionally weights the positive class
    (``alpha`` for pos, ``1-alpha`` for neg); ``None`` = unweighted.
    """
    y = labels.to(logits.dtype)
    p = torch.sigmoid(logits)
    p_t = p * y + (1.0 - p) * (1.0 - y)  # prob of the true class
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = ((1.0 - p_t) ** gamma) * ce
    if alpha is not None:
        a_t = alpha * y + (1.0 - alpha) * (1.0 - y)
        loss = a_t * loss
    return loss.mean()


def listwise_ranking_loss(
    logits: torch.Tensor, group_id: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean over problem-groups of ``-log softmax(group_logits)[positive]``.

    Minimised when each group's (single) positive scores highest. Groups without a positive
    are skipped. ``group_id`` maps each row to its problem.
    """
    total = torch.zeros((), dtype=logits.dtype, device=logits.device)
    n = 0
    for g in torch.unique(group_id):
        m = group_id == g
        gl = logits[m]
        gy = labels[m]
        pos = torch.nonzero(gy > 0.5, as_tuple=False).flatten()
        if pos.numel() == 0 or gl.numel() < 2:
            continue
        logp = F.log_softmax(gl, dim=0)
        total = total - logp[pos].mean()  # (exactly one positive per DD2D problem)
        n += 1
    return total / max(n, 1)
