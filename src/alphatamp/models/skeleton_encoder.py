"""Skeleton encoder: grounded operator sequence → R^d embedding.

Embeds a variable-length skeleton (sequence of grounded operators) into a
fixed-dimensional vector using:

1. Operator-type embedding table
2. Object-ID embedding table
3. Per-operator token = op-type embed + DeepSets mean-pool over object embeds
4. Sinusoidal positional encoding over the operator sequence
5. 2-layer Transformer encoder
6. Mean-pool over sequence → single vector
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["SkeletonEncoder"]


def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding.

    Returns shape (1, max_len, d_model).
    """
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class SkeletonEncoder(nn.Module):
    """Embed a grounded operator sequence into R^d.

    Parameters
    ----------
    num_op_types:
        Number of distinct operator types in the vocabulary.
    num_objects:
        Number of distinct grounded objects in the vocabulary.
    d_model:
        Embedding dimension.
    n_heads:
        Number of Transformer attention heads.
    n_layers:
        Number of Transformer encoder layers.
    max_seq_len:
        Maximum operator sequence length (determines PE buffer size).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        num_op_types: int,
        num_objects: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # (1) Operator-type embedding table
        self.op_type_embed = nn.Embedding(num_op_types, d_model)

        # (2) Object-ID embedding table
        # Input obj_ids are 0-indexed with -1 for padding.  We shift +1 so
        # padding maps to index 0, which is kept at zero via padding_idx.
        self.obj_embed = nn.Embedding(num_objects + 1, d_model, padding_idx=0)

        # (4) Sinusoidal positional encoding (non-learnable buffer)
        self.register_buffer("pos_encoding", _sinusoidal_encoding(max_seq_len, d_model))

        # (5) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(
        self,
        op_type_ids: torch.Tensor,
        obj_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of operator sequences.

        Parameters
        ----------
        op_type_ids:
            (B, L) int — 0-indexed operator-type token per step.
        obj_ids:
            (B, L, P) int — 0-indexed object IDs per parameter position,
            padded with -1 for unused parameter slots.
        lengths:
            (B,) int — actual (unpadded) sequence length for each item.

        Returns
        -------
        torch.Tensor
            (B, d_model) skeleton embeddings.
        """
        B, L = op_type_ids.shape

        # (1) Operator-type embeddings: (B, L, d)
        op_embed = self.op_type_embed(op_type_ids)

        # (2) Object embeddings: shift -1 → 0 (padding_idx), real ids → id+1
        shifted_ids = obj_ids + 1  # (B, L, P)
        obj_embeds = self.obj_embed(shifted_ids)  # (B, L, P, d)

        # (3) DeepSets mean-pool over object arguments (dim=2), excluding padding
        param_mask = (obj_ids >= 0).unsqueeze(-1).float()  # (B, L, P, 1)
        param_count = param_mask.sum(dim=2).clamp(min=1)  # (B, L, 1)
        obj_pool = (obj_embeds * param_mask).sum(dim=2) / param_count  # (B, L, d)

        # Per-operator token = op-type embed + object pool
        tokens = op_embed + obj_pool  # (B, L, d)

        # (4) Add sinusoidal positional encoding
        tokens = tokens + self.pos_encoding[:, :L, :]

        # (5) Transformer encoder with sequence-padding mask
        # src_key_padding_mask: True where the position is padding
        seq_positions = torch.arange(L, device=lengths.device).unsqueeze(0)  # (1, L)
        padding_mask = seq_positions >= lengths.unsqueeze(1)  # (B, L)
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # (6) Mean-pool over non-padding positions
        seq_mask = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        output = (tokens * seq_mask).sum(dim=1) / seq_mask.sum(dim=1).clamp(min=1)

        return output  # (B, d_model)
