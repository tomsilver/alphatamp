"""Composite negative log-likelihood loss for (Y, F, T) prediction heads.

Computes L = L_Y + λ_F · L_F + λ_T · L_T where:

- L_Y: Bernoulli NLL (BCE with logits) on all active entries.
- L_F: Beta NLL on all active entries.
- L_T: LogNormal NLL on active entries where Y=1 only.

Active = applicable AND revealed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, LogNormal

__all__ = ["PredictionNLLLoss"]


class PredictionNLLLoss(nn.Module):
    """Composite NLL loss for distributional prediction heads.

    Parameters
    ----------
    lambda_f:
        Weight for the Beta (F) loss term.
    lambda_t:
        Weight for the LogNormal (T) loss term.
    f_boundary_eps:
        Clamp f_true to [eps, 1-eps] before Beta log_prob.
    t_floor:
        Clamp t_true to [t_floor, ∞) before LogNormal log_prob.
    """

    def __init__(
        self,
        lambda_f: float = 0.5,
        lambda_t: float = 0.3,
        f_boundary_eps: float = 1e-4,
        t_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_f = lambda_f
        self.lambda_t = lambda_t
        self.f_boundary_eps = f_boundary_eps
        self.t_floor = t_floor

    def forward(
        self,
        y_logits: torch.Tensor,
        f_dist: Beta,
        t_dist: LogNormal,
        y_true: torch.Tensor,
        f_true: torch.Tensor,
        t_true: torch.Tensor,
        applicable_mask: torch.Tensor,
        revealed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute composite NLL loss.

        Parameters
        ----------
        y_logits:
            (B, M) raw logits from JointYHead.
        f_dist:
            Beta distribution with batch_shape (B, M) from FHead.
        t_dist:
            LogNormal distribution with batch_shape (B, M) from THead.
        y_true:
            (B, M) float in {0, 1}.
        f_true:
            (B, M) float in [0, 1].
        t_true:
            (B, M) float > 0 (only meaningful where y_true=1).
        applicable_mask:
            (B, M) bool — True where skeleton is applicable.
        revealed_mask:
            (B, M) bool — True where outcome has been observed.

        Returns
        -------
        dict with keys ``'loss'``, ``'loss_y'``, ``'loss_f'``, ``'loss_t'``.
        """
        device = y_logits.device
        zero = torch.tensor(0.0, device=device)

        active = applicable_mask & revealed_mask  # (B, M)

        # --- L_Y: Bernoulli NLL ---
        if active.any():
            loss_y = F.binary_cross_entropy_with_logits(
                y_logits[active], y_true[active], reduction="mean",
            )
        else:
            loss_y = zero

        # --- L_F: Beta NLL ---
        if active.any():
            alpha_active = f_dist.concentration1[active]
            beta_active = f_dist.concentration0[active]
            active_f_dist = Beta(alpha_active, beta_active)
            f_clamped = f_true[active].clamp(
                self.f_boundary_eps, 1.0 - self.f_boundary_eps,
            )
            loss_f = -active_f_dist.log_prob(f_clamped).mean()
        else:
            loss_f = zero

        # --- L_T: LogNormal NLL (Y=1 only) ---
        t_mask = active & (y_true > 0.5)
        if t_mask.any():
            mu_active = t_dist.loc[t_mask]
            sigma_active = t_dist.scale[t_mask]
            active_t_dist = LogNormal(mu_active, sigma_active)
            t_clamped = t_true[t_mask].clamp(min=self.t_floor)
            loss_t = -active_t_dist.log_prob(t_clamped).mean()
        else:
            loss_t = zero

        loss = loss_y + self.lambda_f * loss_f + self.lambda_t * loss_t

        return {
            "loss": loss,
            "loss_y": loss_y,
            "loss_f": loss_f,
            "loss_t": loss_t,
        }
