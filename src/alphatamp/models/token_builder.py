"""Outcome encoder and token builder for the belief-encoder pipeline.

OutcomeEncoder: MLP that maps per-skeleton outcome tuples (Y, F, T, L) to
    R^{d_out}, applying log1p to refinement time T.

TokenBuilder: Assembles a dense (B, M, d_skel + d_out) token tensor from
    pre-computed skeleton embeddings, applicability, history, and candidate
    status using three learned status embeddings:

    - eta_inapp:  inapplicable skeletons (applicability = 0)
    - eta_cand:   candidates (applicable, not yet attempted)
    - history:    outcome encoding from OutcomeEncoder (replaces status embed)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["OutcomeEncoder", "TokenBuilder"]


class OutcomeEncoder(nn.Module):
    """Encode per-skeleton outcomes (Y, F, T, L) into a dense vector.

    Applies ``log1p`` to refinement time *T* before the MLP so that the
    network sees a compressed, well-scaled input.

    Parameters
    ----------
    d_out:
        Output embedding dimension.
    hidden_dim:
        Hidden layer width.  Defaults to *d_out*.
    dropout:
        Dropout probability between layers.
    """

    INPUT_DIM = 4  # Y, F, log1p(T), L

    def __init__(
        self,
        d_out: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_out
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_out),
        )

    def forward(
        self,
        y: Tensor,
        f: Tensor,
        t: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Encode outcome tuples.

        All inputs broadcast to the same leading shape ``(*,)``.

        Parameters
        ----------
        y:  Refinement success indicator (0 or 1).
        f:  Steps-completed fraction in [0, 1].
        t:  Refinement time (non-negative).
        lengths:  Skeleton length (number of operators).

        Returns
        -------
        Tensor of shape ``(*, d_out)``.
        """
        features = torch.stack([y, f, torch.log1p(t), lengths], dim=-1)
        return self.mlp(features)


class TokenBuilder(nn.Module):
    """Build per-skeleton token tensor for the BeliefEncoder.

    Each skeleton slot receives its pre-computed embedding concatenated with a
    status-dependent suffix of dimension *d_out*:

    - **Inapplicable** (``applicability == 0``): learned ``eta_inapp``
    - **Candidate** (applicable, not yet attempted): learned ``eta_cand``
    - **History** (applicable, already attempted): ``OutcomeEncoder(Y, F, T, L)``

    Parameters
    ----------
    d_skel:
        Dimension of skeleton embeddings from :class:`SkeletonEncoder`.
    d_out:
        Dimension of status / outcome suffix (and OutcomeEncoder output).
    hidden_dim:
        Hidden dimension for the OutcomeEncoder MLP.  Defaults to *d_out*.
    dropout:
        Dropout probability inside the OutcomeEncoder.
    """

    def __init__(
        self,
        d_skel: int,
        d_out: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_skel = d_skel
        self.d_out = d_out

        self.eta_inapp = nn.Parameter(torch.randn(d_out))
        self.eta_cand = nn.Parameter(torch.randn(d_out))
        self.outcome_encoder = OutcomeEncoder(
            d_out=d_out, hidden_dim=hidden_dim, dropout=dropout,
        )

    @property
    def d_token(self) -> int:
        """Total token dimension fed to BeliefEncoder."""
        return self.d_skel + self.d_out

    def forward(
        self,
        skeleton_embeds: Tensor,
        applicability: Tensor,
        revealed_mask: Tensor,
        y: Tensor,
        f: Tensor,
        t: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Assemble the token tensor.

        Parameters
        ----------
        skeleton_embeds:
            ``(B, M, d_skel)`` pre-computed skeleton embeddings.
        applicability:
            ``(B, M)`` float ``{0, 1}`` — 1 if applicable.
        revealed_mask:
            ``(B, M)`` bool — True for skeletons in history H_t.
        y:
            ``(B, M)`` float — refinement success (ignored where not revealed).
        f:
            ``(B, M)`` float — steps-completed fraction.
        t:
            ``(B, M)`` float — refinement time.
        lengths:
            ``(B, M)`` float — skeleton lengths.

        Returns
        -------
        Tensor of shape ``(B, M, d_skel + d_out)``.
        """
        B, M, _ = skeleton_embeds.shape

        # Outcome encoding for every position (only used where revealed)
        outcome_enc = self.outcome_encoder(y, f, t, lengths)  # (B, M, d_out)

        # Build status suffix: start with eta_cand everywhere
        status = self.eta_cand.expand(B, M, -1).clone()  # (B, M, d_out)

        # Overwrite inapplicable positions
        inapp_mask = (applicability == 0).unsqueeze(-1)  # (B, M, 1)
        status = torch.where(inapp_mask, self.eta_inapp.expand_as(status), status)

        # Overwrite revealed (history) positions with outcome encoding
        rev_mask = revealed_mask.unsqueeze(-1)  # (B, M, 1)
        status = torch.where(rev_mask, outcome_enc, status)

        return torch.cat([skeleton_embeds, status], dim=-1)  # (B, M, d_skel+d_out)
