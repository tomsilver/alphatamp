"""Set-Transformer attention primitives shared across the SPECTRE encoders.

These building blocks — a masked Set-Attention Block (SAB) and Pooling-by-Multihead-
Attention (PMA) — are the relational-pooling substrate the geometry/evidence encoders in
:mod:`alphatamp.approaches.spectre.encoders` and the ranker in
:mod:`alphatamp.approaches.spectre.model` are built from. They carry no vocabulary or
domain knowledge; they are pure ``nn.Module`` primitives over masked token sets.

Hidden size is ``d = 64`` throughout; multihead attention uses 4 heads. Post-norm layout
per the original Set Transformer paper, with an empty-set guard so a fully-padded token
set pools to a zero vector rather than a NaN.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

D_MODEL = 64
N_HEADS = 4
DROPOUT = 0.1  # default; per-instance ``dropout_p`` overrides
FFN_DIM = 256


class SetAttentionBlock(nn.Module):
    """One SAB: multihead self-attention + LN + FFN + LN, mask-aware.

    Post-norm layout per the original Set Transformer paper. No positional
    embeddings — used over true sets of atoms / failure embeddings.
    """

    def __init__(
        self,
        dim: int = D_MODEL,
        n_heads: int = N_HEADS,
        dropout_p: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, FFN_DIM),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(FFN_DIM, dim),
            nn.Dropout(dropout_p),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """``x``: (..., N, D); ``mask``: (..., N) bool, True = real token."""
        flat_x, flat_mask, restore = _flatten_set_dims(x, mask)
        # Rows where every entry is masked produce NaN attention weights;
        # MultiheadAttention's `key_padding_mask` blocks attending TO pads,
        # but if every key is a pad, the softmax is over -inf only.
        # We unmask one slot in those rows — its output is then re-masked
        # by the caller via ``flat_mask``-based pooling.
        all_pad = ~flat_mask.any(dim=-1)
        safe_mask = flat_mask.clone()
        safe_mask[all_pad, 0] = True
        kpm = ~safe_mask  # MHA expects True = ignore
        attn_out, _ = self.attn(flat_x, flat_x, flat_x, key_padding_mask=kpm)
        h = self.ln1(flat_x + attn_out)
        h = self.ln2(h + self.ffn(h))
        # Re-mask outputs at fully-padded positions; downstream pools
        # treat masked entries via the mask itself, but zeroing here keeps
        # numerics clean.
        h = h * flat_mask.unsqueeze(-1)
        return restore(h)


class PoolingByMultiheadAttention(nn.Module):
    """``PMA_{k=1}``: one learned seed attends over a masked token set."""

    def __init__(
        self,
        dim: int = D_MODEL,
        n_heads: int = N_HEADS,
        dropout_p: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.seed, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, FFN_DIM),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(FFN_DIM, dim),
            nn.Dropout(dropout_p),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Returns (..., D); ``x``: (..., N, D); ``mask``: (..., N) bool."""
        flat_x, flat_mask, _ = _flatten_set_dims(x, mask)
        bsz = flat_x.size(0)
        seed = self.seed.expand(bsz, 1, -1)
        # Empty-set guard — see SAB.forward note.
        all_pad = ~flat_mask.any(dim=-1)
        safe_mask = flat_mask.clone()
        safe_mask[all_pad, 0] = True
        kpm = ~safe_mask
        attn_out, _ = self.attn(seed, flat_x, flat_x, key_padding_mask=kpm)
        h = self.ln(seed + attn_out)
        h = self.ln2(h + self.ffn(h))
        h = h.squeeze(1)  # (B, D)
        # If the original set was fully-padded, return zero to remove
        # the synthetic-unmask leak.
        h = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(h), h)
        # Restore leading dims.
        leading = x.shape[:-2]
        return h.view(*leading, h.size(-1))


def _flatten_set_dims(
    x: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """Flatten leading dims so MHA sees ``(B*, N, D)``.

    Returns ``(flat_x, flat_mask, restore)`` where ``restore`` reshapes back
    to the input's leading dims.
    """
    n = x.size(-2)
    d = x.size(-1)
    flat_x = x.reshape(-1, n, d)
    flat_mask = mask.reshape(-1, n)
    leading = x.shape[:-2]

    def restore(h: Tensor) -> Tensor:
        return h.view(*leading, n, d)

    return flat_x, flat_mask, restore
