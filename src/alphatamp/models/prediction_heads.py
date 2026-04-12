"""Distributional prediction heads for per-candidate outcome forecasting.

Each head consumes contextualized per-skeleton embeddings (B, M, d_model) from
the BeliefEncoder and produces distributional predictions:

- YHead:      Bernoulli logit for refinement success P(Y=1).
- FHead:      Beta(α, β) for steps-completed fraction F ∈ [0, 1].
- THead:      LogNormal(μ, σ) for refinement time T | Y=1.
- JointYHead: Cross-candidate attention → low-rank additive correction to Y logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, LogNormal

__all__ = ["YHead", "FHead", "THead", "JointYHead"]


class YHead(nn.Module):
    """Bernoulli logit head for refinement success probability.

    Parameters
    ----------
    d_model:
        Dimension of contextualized embeddings from BeliefEncoder.
    hidden_dim:
        Hidden layer dimension. Defaults to *d_model*.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        contextualized: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, M) raw logits; sigmoid(logit) = P(Y=1)."""
        return self.mlp(contextualized).squeeze(-1)  # (B, M)


class FHead(nn.Module):
    """Beta distribution head for steps-completed fraction.

    Parameters
    ----------
    d_model:
        Dimension of contextualized embeddings.
    hidden_dim:
        Hidden layer dimension. Defaults to *d_model*.
    dropout:
        Dropout probability.
    concentration_floor:
        Minimum value for α, β to avoid degenerate distributions.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        concentration_floor: float = 0.01,
    ) -> None:
        super().__init__()
        self.concentration_floor = concentration_floor
        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        contextualized: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Beta:
        """Return Beta distribution with batch_shape (B, M)."""
        raw = self.mlp(contextualized)  # (B, M, 2)
        alpha = F.softplus(raw[..., 0]) + self.concentration_floor
        beta = F.softplus(raw[..., 1]) + self.concentration_floor
        return Beta(alpha, beta)


class THead(nn.Module):
    """LogNormal distribution head for refinement time conditional on success.

    Parameters
    ----------
    d_model:
        Dimension of contextualized embeddings.
    hidden_dim:
        Hidden layer dimension. Defaults to *d_model*.
    dropout:
        Dropout probability.
    sigma_floor:
        Minimum σ to prevent distribution collapse.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        sigma_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.sigma_floor = sigma_floor
        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        contextualized: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> LogNormal:
        """Return LogNormal distribution with batch_shape (B, M).

        Policy uses ``dist.mean`` = exp(μ + σ²/2) for E[T | Y=1].
        """
        raw = self.mlp(contextualized)  # (B, M, 2)
        mu = raw[..., 0]
        sigma = F.softplus(raw[..., 1]) + self.sigma_floor
        return LogNormal(mu, sigma)


class JointYHead(nn.Module):
    """Cross-candidate correlation correction for Y logits.

    Applies one self-attention layer across the M candidate positions, then
    projects through a low-rank bottleneck to produce an additive correction
    to the marginal Y logits from :class:`YHead`.

    Parameters
    ----------
    d_model:
        Dimension of contextualized embeddings.
    n_heads:
        Attention heads for cross-candidate attention.
    rank:
        Bottleneck rank of the low-rank projection to scalar correction.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        rank: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, rank),
            nn.ReLU(),
            nn.Linear(rank, 1),
        )

    def forward(
        self,
        contextualized: torch.Tensor,
        marginal_logits: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, M) corrected logits = marginal_logits + delta."""
        x = self.norm(contextualized)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x = contextualized + attn_out  # residual
        delta = self.proj(x).squeeze(-1)  # (B, M)
        return marginal_logits + delta
