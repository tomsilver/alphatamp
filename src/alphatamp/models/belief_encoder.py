"""Belief encoder: per-skeleton tokens → contextualized embeddings + belief vector.

Consumes a (B, M, d_token) tensor of per-skeleton tokens (from a future
TokenBuilder) and produces:

1. Contextualized per-skeleton embeddings (B, M, d_model) via cross-skeleton
   Transformer attention.
2. Pooled belief vector β_t (B, d_model) = mean over unmasked positions.

No positional encoding is applied — skeleton slots are an unordered set.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["BeliefEncoder"]


class BeliefEncoder(nn.Module):
    """Encode per-skeleton tokens into a belief state.

    Parameters
    ----------
    d_token:
        Dimension of input tokens (from TokenBuilder).
    d_model:
        Internal Transformer dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of Transformer encoder layers.
    ffn_dim:
        Feedforward hidden dimension.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_token: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # (1) Input projection
        if d_token != d_model:
            self.input_proj: nn.Module = nn.Linear(d_token, d_model)
        else:
            self.input_proj = nn.Identity()

        # (2) Transformer encoder (pre-norm, no positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode skeleton tokens into contextualized embeddings and belief.

        Parameters
        ----------
        tokens:
            (B, M, d_token) per-skeleton token vectors.
        pad_mask:
            (B, M) bool — True where the slot is padding.

        Returns
        -------
        contextualized:
            (B, M, d_model) per-skeleton embeddings after cross-attention.
        belief:
            (B, d_model) pooled belief vector β_t.
        """
        # (1) Project to Transformer dimension
        x = self.input_proj(tokens)  # (B, M, d_model)

        # (2) Transformer encoder with padding mask
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, M, d_model)

        # (3) Mask-aware mean pool → belief vector
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, M, 1)
        belief = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, d_model)

        return x, belief
