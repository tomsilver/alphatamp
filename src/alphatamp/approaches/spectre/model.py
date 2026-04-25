"""SPECTRE model: Φ skeleton encoder, Ψ context encoder, σ scorer.

Implements ``SPECTRE_RT2D_METHOD_SPEC.md`` §3–§6 with the four mandatory
fixes from §9 already wired in:

- **Fix #1 (Set-Transformer atom pool).** :class:`SkeletonEncoder` pools
  per-state atom tokens via a single SAB followed by ``PMA_{k=1}``, not the
  Deep-Sets mean of the original spec. The relational join over shared
  arguments (``PassageWidth(p, w)`` ↔ ``TraverseLoadedColor⟨X⟩(…, p, …)``)
  is representable in a single SAB layer.
- **Fix #3 (vocab-driven dynamic MLP sizing).** ``D_in`` for the operator
  MLP is ``32 + A*16 + 16`` and for the atom Linear is ``32 + P*24``, with
  ``A = vocab.max_operator_arity`` and ``P = vocab.max_predicate_arity``
  read from the vocab at construction time.

Hidden size is ``d = 64`` throughout. Multi-head attention uses 4 heads;
the sequence transformer uses 2 layers, the atom-pool 1 SAB + 1 PMA, the
context encoder 2 SAB + 1 PMA. Total trainable ≈ 185k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from alphatamp.approaches.spectre.dataset import SpectreBatch
from alphatamp.approaches.spectre.vocab import Vocab

D_MODEL = 64
N_HEADS = 4
DROPOUT = 0.1
FFN_DIM = 256

D_OP_NAME = 32
D_PRED_NAME = 32
D_TYPE = 8
D_LOCAL = 16
D_ARG_SLOT_OUT = 16
D_OP_POS = 16
D_TYPE_HIST = 16

# Token-type ids inside the sequence transformer.
TOKEN_TYPE_S0 = 0
TOKEN_TYPE_OP = 1
TOKEN_TYPE_SL = 2

# Hard ceiling on local ids — vocab.max_objects_per_type may be empty for
# legacy vocabs, so we default to a generous bound. Real RT2D values are
# ≤ 6 (zones); kinder envs ≤ 25 obstructions.
_DEFAULT_MAX_LOCAL_ID = 64


def _max_local_id(vocab: Vocab) -> int:
    """Upper bound on local-id values; sized for the largest type bucket."""
    if not vocab.max_objects_per_type:
        return _DEFAULT_MAX_LOCAL_ID
    return max(_DEFAULT_MAX_LOCAL_ID, *vocab.max_objects_per_type.values())


# ---------------------------------------------------------------------------
# Set-Transformer building blocks
# ---------------------------------------------------------------------------


class SetAttentionBlock(nn.Module):
    """One SAB: multihead self-attention + LN + FFN + LN, mask-aware.

    Post-norm layout per the original Set Transformer paper. No positional
    embeddings — used over true sets of atoms / failure embeddings.
    """

    def __init__(self, dim: int = D_MODEL, n_heads: int = N_HEADS) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, FFN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_DIM, dim),
            nn.Dropout(DROPOUT),
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

    def __init__(self, dim: int = D_MODEL, n_heads: int = N_HEADS) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.seed, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, FFN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_DIM, dim),
            nn.Dropout(DROPOUT),
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


# ---------------------------------------------------------------------------
# Φ — skeleton encoder
# ---------------------------------------------------------------------------


class _OperatorTokenEncoder(nn.Module):
    """Operator-token sub-encoder per spec §4.2."""

    def __init__(self, vocab: Vocab) -> None:
        super().__init__()
        self.max_op_arity = max(int(vocab.max_operator_arity), 1)
        self.op_name_emb = nn.Embedding(
            num_embeddings=len(vocab.operators), embedding_dim=D_OP_NAME, padding_idx=0
        )
        self.arg_type_emb = nn.Embedding(
            num_embeddings=len(vocab.types), embedding_dim=D_TYPE, padding_idx=0
        )
        self.arg_local_emb = nn.Embedding(
            num_embeddings=_max_local_id(vocab) + 1,
            embedding_dim=D_LOCAL,
            padding_idx=0,
        )
        # Slot-specific projection: separate Linear per arg slot.
        self.arg_slot_proj = nn.ModuleList(
            [
                nn.Linear(D_TYPE + D_LOCAL, D_ARG_SLOT_OUT)
                for _ in range(self.max_op_arity)
            ]
        )
        self.op_pos_emb = nn.Embedding(
            num_embeddings=int(vocab.max_skeleton_length) + 2,
            embedding_dim=D_OP_POS,
        )
        d_in = D_OP_NAME + self.max_op_arity * D_ARG_SLOT_OUT + D_OP_POS
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, D_MODEL),
        )
        self._d_in = d_in

    @property
    def in_features(self) -> int:
        """Operator-token MLP input dim ``32 + A*16 + 16`` (spec §4.2)."""
        return self._d_in

    def forward(
        self,
        op_ids: Tensor,  # (..., L)
        arg_type_ids: Tensor,  # (..., L, A)
        arg_local_ids: Tensor,  # (..., L, A)
        op_position: Tensor,  # (..., L)
    ) -> Tensor:  # (..., L, D_MODEL)
        """Encode operator tokens to ``D_MODEL``-dim per spec §4.2."""
        name_emb = self.op_name_emb(op_ids)
        type_emb = self.arg_type_emb(arg_type_ids)
        local_emb = self.arg_local_emb(arg_local_ids)
        # (..., L, A, D_TYPE + D_LOCAL)
        slot_in = torch.cat([type_emb, local_emb], dim=-1)
        # Apply per-slot Linear: split along arg dim.
        slot_outs = []
        for slot_idx, proj in enumerate(self.arg_slot_proj):
            slot_outs.append(proj(slot_in[..., slot_idx, :]))
        # (..., L, A, D_ARG_SLOT_OUT)
        arg_token = torch.stack(slot_outs, dim=-2)
        # Flatten the (A, D_ARG_SLOT_OUT) axes into one feature dim.
        arg_token = arg_token.flatten(start_dim=-2)
        pos_emb = self.op_pos_emb(op_position)
        op_in = torch.cat([name_emb, arg_token, pos_emb], dim=-1)
        return self.mlp(op_in)


class _StateTokenEncoder(nn.Module):
    """Per-state atom-pool sub-encoder Φ_s per spec §4.3 (Set Transformer pool).

    Single-pool by default. When constructed with a non-empty
    ``static_tag_predicate_ids`` buffer, atoms whose predicate id is in
    that set are routed to a separate SAB+PMA stream (F3-B-(1)
    "predicate-type-conditioned pooling"). The two stream outputs and the
    type-histogram are concatenated and projected back to ``D_MODEL``.
    """

    def __init__(
        self,
        vocab: Vocab,
        use_atom_sab2: bool = True,
        static_tag_predicates: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.max_pred_arity = max(int(vocab.max_predicate_arity), 1)
        self.pred_emb = nn.Embedding(
            num_embeddings=len(vocab.predicates),
            embedding_dim=D_PRED_NAME,
            padding_idx=0,
        )
        self.arg_type_emb = nn.Embedding(
            num_embeddings=len(vocab.types), embedding_dim=D_TYPE, padding_idx=0
        )
        self.arg_local_emb = nn.Embedding(
            num_embeddings=_max_local_id(vocab) + 1,
            embedding_dim=D_LOCAL,
            padding_idx=0,
        )
        atom_in = D_PRED_NAME + self.max_pred_arity * (D_TYPE + D_LOCAL)
        self.atom_proj = nn.Linear(atom_in, D_MODEL)
        self.atom_ln = nn.LayerNorm(D_MODEL)
        # Primary (legacy) atom pool. With ``use_static_tag_pool`` enabled
        # this becomes the "fluent" stream; otherwise it pools all atoms.
        self.atom_sab1 = SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS)
        self.atom_sab2: SetAttentionBlock | None = (
            SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS) if use_atom_sab2 else None
        )
        self.atom_pma = PoolingByMultiheadAttention(dim=D_MODEL, n_heads=N_HEADS)

        # F3-B-(1) static-tag stream. Built only when caller supplies a
        # non-empty predicate-name list AND at least one of those names
        # exists in the vocab. Stored as a buffer of vocab pred-ids so a
        # checkpoint round-trips cleanly across vocab order changes.
        resolved_ids: list[int] = []
        if static_tag_predicates:
            for name in static_tag_predicates:
                if name in vocab.predicates:
                    pid = vocab.pred_idx(name)
                    if pid > 0:  # exclude <OOV>/pad
                        resolved_ids.append(pid)
        resolved_ids = sorted(set(resolved_ids))
        self.use_static_tag_pool = len(resolved_ids) > 0
        # Non-persistent: the buffer is always rebuilt from the constructor
        # ``static_tag_predicates`` argument, so we don't write it into
        # ``state_dict`` (would break legacy checkpoints that pre-date
        # F3-B-(1) with a "missing key" error). The cfg dict + env_registry
        # already let ``inference.load_checkpoint`` recover the list.
        self.register_buffer(
            "static_tag_predicate_ids",
            torch.tensor(resolved_ids, dtype=torch.long),
            persistent=False,
        )
        if self.use_static_tag_pool:
            self.atom_sab1_static = SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS)
            self.atom_sab2_static: SetAttentionBlock | None = (
                SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS)
                if use_atom_sab2
                else None
            )
            self.atom_pma_static = PoolingByMultiheadAttention(
                dim=D_MODEL, n_heads=N_HEADS
            )

        # Type-histogram path
        self.type_hist_proj = nn.Linear(len(vocab.types), D_TYPE_HIST)
        # state_proj input grows by one D_MODEL when the static stream is on.
        state_proj_in = (
            D_MODEL + D_TYPE_HIST + (D_MODEL if self.use_static_tag_pool else 0)
        )
        self.state_proj = nn.Linear(state_proj_in, D_MODEL)
        self.state_ln = nn.LayerNorm(D_MODEL)
        self._atom_in = atom_in

    @property
    def atom_in_features(self) -> int:
        """Atom-token Linear input dim ``32 + P*24`` (spec §4.3)."""
        return self._atom_in

    def _pool_one_stream(
        self,
        atom_tok: Tensor,
        mask: Tensor,
        sab1: SetAttentionBlock,
        sab2: SetAttentionBlock | None,
        pma: PoolingByMultiheadAttention,
    ) -> Tensor:
        """Run a single SAB(+SAB)+PMA pool stream over the masked atoms."""
        h = sab1(atom_tok, mask)
        if sab2 is not None:
            h = sab2(h, mask)
        return pma(h, mask)

    def forward(
        self,
        pred_ids: Tensor,  # (..., M)
        arg_type_ids: Tensor,  # (..., M, P)
        arg_local_ids: Tensor,  # (..., M, P)
        atom_mask: Tensor,  # (..., M) bool
        type_histogram: Tensor,  # (..., T) long
    ) -> Tensor:  # (..., D_MODEL)
        """Pool atom tokens via SAB+PMA (single or dual stream), then concat type-histogram."""
        pe = self.pred_emb(pred_ids)
        te = self.arg_type_emb(arg_type_ids)
        le = self.arg_local_emb(arg_local_ids)
        # Concat type/local along feature dim, then flatten the P axis.
        arg_tok = torch.cat([te, le], dim=-1).flatten(start_dim=-2)
        atom_in = torch.cat([pe, arg_tok], dim=-1)
        atom_tok = self.atom_proj(atom_in)
        atom_tok = self.atom_ln(atom_tok)

        thist = self.type_hist_proj(type_histogram.float())

        if self.use_static_tag_pool:
            # Build static-vs-fluent partition from atom_mask AND predicate.
            # ``isin`` over a buffer of allowed pred-ids works on any
            # leading-dim shape so this handles (M,), (B, M), (B, K, M).
            # Cast: ``register_buffer`` annotation returns ``Tensor | Module``
            # per nn.Module typing; the buffer we registered is a Tensor.
            static_ids = self.static_tag_predicate_ids
            assert isinstance(static_ids, Tensor)
            is_static_pred = torch.isin(pred_ids, static_ids)
            static_mask = atom_mask & is_static_pred
            fluent_mask = atom_mask & (~is_static_pred)
            static_pool = self._pool_one_stream(
                atom_tok,
                static_mask,
                self.atom_sab1_static,
                self.atom_sab2_static,
                self.atom_pma_static,
            )
            fluent_pool = self._pool_one_stream(
                atom_tok, fluent_mask, self.atom_sab1, self.atom_sab2, self.atom_pma
            )
            state_in = torch.cat([static_pool, fluent_pool, thist], dim=-1)
        else:
            atom_pool = self._pool_one_stream(
                atom_tok, atom_mask, self.atom_sab1, self.atom_sab2, self.atom_pma
            )
            # Empty-state edge case: PMA returns zero vector when atom_mask is
            # all-False. type-histogram path still carries a signal.
            state_in = torch.cat([atom_pool, thist], dim=-1)

        out = self.state_ln(self.state_proj(state_in))
        return torch.nn.functional.gelu(out)


class SkeletonEncoder(nn.Module):
    """Φ: skeleton → 64-dim embedding (spec §4)."""

    def __init__(
        self,
        vocab: Vocab,
        use_atom_sab2: bool = True,
        static_tag_predicates: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.op_enc = _OperatorTokenEncoder(vocab)
        self.state_enc = _StateTokenEncoder(
            vocab,
            use_atom_sab2=use_atom_sab2,
            static_tag_predicates=static_tag_predicates,
        )
        self.token_type_emb = nn.Embedding(num_embeddings=3, embedding_dim=D_MODEL)
        # +2 to accommodate (s_0, ..., s_L) sequence; +4 of slack.
        self.seq_pos_emb = nn.Embedding(
            num_embeddings=int(vocab.max_skeleton_length) + 4,
            embedding_dim=D_MODEL,
        )
        self.seq_ln = nn.LayerNorm(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        op_ids: Tensor,  # (B, K, L)
        op_arg_type_ids: Tensor,  # (B, K, L, A)
        op_arg_local_ids: Tensor,  # (B, K, L, A)
        op_mask: Tensor,  # (B, K, L)
        s0_pred_ids: Tensor,  # (B, M0)
        s0_arg_type_ids: Tensor,  # (B, M0, P)
        s0_arg_local_ids: Tensor,  # (B, M0, P)
        s0_atom_mask: Tensor,  # (B, M0)
        s0_type_histogram: Tensor,  # (B, T)
        sL_pred_ids: Tensor,  # (B, K, ML)
        sL_arg_type_ids: Tensor,  # (B, K, ML, P)
        sL_arg_local_ids: Tensor,  # (B, K, ML, P)
        sL_atom_mask: Tensor,  # (B, K, ML)
        sL_type_histogram: Tensor,  # (B, K, T)
    ) -> Tensor:  # (B, K, D_MODEL)
        """Encode every skeleton in the (B, K) pool to a 64-dim e(s) (spec §4)."""
        bsz, k, l_max = op_ids.shape
        device = op_ids.device

        # ------- s_0 token (per-example, broadcast to per-skeleton) -------
        s0_tok = self.state_enc(
            s0_pred_ids,
            s0_arg_type_ids,
            s0_arg_local_ids,
            s0_atom_mask,
            s0_type_histogram,
        )  # (B, D)
        s0_tok = s0_tok.unsqueeze(1).expand(bsz, k, D_MODEL)  # (B, K, D)

        # ------- s_L token (per-skeleton) -------
        sL_tok = self.state_enc(
            sL_pred_ids,
            sL_arg_type_ids,
            sL_arg_local_ids,
            sL_atom_mask,
            sL_type_histogram,
        )  # (B, K, D)

        # ------- Operator tokens -------
        positions = (
            torch.arange(l_max, device=device).view(1, 1, l_max).expand(bsz, k, l_max)
        )
        op_tok = self.op_enc(
            op_ids, op_arg_type_ids, op_arg_local_ids, positions
        )  # (B, K, L, D)

        # ------- Stitch sequence: [STATE_0, OP_1, ..., OP_L, STATE_L] -------
        seq_len = l_max + 2
        seq = torch.zeros(bsz, k, seq_len, D_MODEL, device=device, dtype=op_tok.dtype)
        seq[:, :, 0, :] = s0_tok
        seq[:, :, 1 : 1 + l_max, :] = op_tok
        seq[:, :, 1 + l_max, :] = sL_tok
        seq_mask = torch.zeros(bsz, k, seq_len, dtype=torch.bool, device=device)
        seq_mask[:, :, 0] = True
        seq_mask[:, :, 1 : 1 + l_max] = op_mask
        seq_mask[:, :, 1 + l_max] = True

        # Token-type embeddings
        type_ids = torch.full(
            (seq_len,), TOKEN_TYPE_OP, dtype=torch.long, device=device
        )
        type_ids[0] = TOKEN_TYPE_S0
        type_ids[seq_len - 1] = TOKEN_TYPE_SL
        type_emb = self.token_type_emb(type_ids)  # (seq_len, D)

        pos_ids = torch.arange(seq_len, device=device)
        pos_emb = self.seq_pos_emb(pos_ids)  # (seq_len, D)

        seq = (
            seq
            + type_emb.view(1, 1, seq_len, D_MODEL)
            + pos_emb.view(1, 1, seq_len, D_MODEL)
        )
        seq = self.seq_ln(seq)

        # Flatten (B, K) into batch dim for the TransformerEncoder.
        flat = seq.reshape(bsz * k, seq_len, D_MODEL)
        flat_mask = seq_mask.reshape(bsz * k, seq_len)
        # `key_padding_mask` expects True = ignore.
        # Sequences where every position is padded shouldn't occur (s_0 and
        # s_L are always present), but we guard anyway.
        all_pad = ~flat_mask.any(dim=-1)
        safe_mask = flat_mask.clone()
        safe_mask[all_pad, 0] = True
        encoded = self.transformer(flat, src_key_padding_mask=~safe_mask)
        # Mean-pool with mask. Use safe_mask as the divisor lower-bound to
        # avoid division-by-zero for the (impossible) all-pad row.
        denom = safe_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(encoded.dtype)
        masked = encoded * safe_mask.unsqueeze(-1).to(encoded.dtype)
        e = masked.sum(dim=1) / denom
        return e.view(bsz, k, D_MODEL)


# ---------------------------------------------------------------------------
# Ψ — context encoder
# ---------------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """Ψ: failure-set → 64-dim context (spec §5)."""

    def __init__(self) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(D_MODEL)
        self.sab1 = SetAttentionBlock(D_MODEL, N_HEADS)
        self.sab2 = SetAttentionBlock(D_MODEL, N_HEADS)
        self.pma = PoolingByMultiheadAttention(D_MODEL, N_HEADS)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        # Learned "no-failure" context, returned when |F| == 0.
        self.c0 = nn.Parameter(torch.zeros(D_MODEL))

    def forward(
        self,
        f_embeddings: Tensor,  # (B, F, D)
        f_mask: Tensor,  # (B, F)
    ) -> Tensor:  # (B, D)
        """Pool the failure-set embeddings into ``c_t`` (spec §5)."""
        any_f = f_mask.any(dim=-1, keepdim=True)  # (B, 1) bool
        x = self.input_ln(f_embeddings)
        x = self.sab1(x, f_mask)
        x = self.sab2(x, f_mask)
        c = self.pma(x, f_mask)  # (B, D); zeros for empty-F rows
        c = self.out_proj(c)
        # Where |F| == 0, return broadcast c_0; otherwise c.
        c0 = self.c0.view(1, -1).expand_as(c)
        return torch.where(any_f, c, c0)


# ---------------------------------------------------------------------------
# σ — scorer
# ---------------------------------------------------------------------------


class Scorer(nn.Module):
    """σ: ``(e(s), c, π(s)) → scalar`` (spec §6)."""

    def __init__(self, prior_dropout_p: float = 0.2) -> None:
        super().__init__()
        self.prior_dropout_p = prior_dropout_p
        # π_proj: spec calls for diagonal init at α=0.1 with zero bias.
        # Linear(1 → 8): there's only one input dim, so "diagonal" reduces
        # to setting all 8 weights to α and bias to 0.
        self.prior_proj = nn.Linear(1, 8)
        with torch.no_grad():
            self.prior_proj.weight.fill_(0.1)
            self.prior_proj.bias.fill_(0.0)
        in_dim = D_MODEL + D_MODEL + 8
        self.fc1 = nn.Linear(in_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.head = nn.Linear(64, 1)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(
        self,
        e_R: Tensor,  # (B, R, D)
        c: Tensor,  # (B, D)
        priors: Tensor,  # (B, R)
        prior_dropout: bool = False,
    ) -> Tensor:  # (B, R)
        """Score each candidate as a scalar logit (spec §6)."""
        if prior_dropout and self.training and self.prior_dropout_p > 0:
            keep = (
                torch.rand(priors.size(0), device=priors.device) >= self.prior_dropout_p
            )
            priors = priors * keep.to(priors.dtype).unsqueeze(-1)
        prior_feat = self.prior_proj(priors.unsqueeze(-1))  # (B, R, 8)
        c_broadcast = c.unsqueeze(1).expand_as(e_R)
        x = torch.cat([e_R, c_broadcast, prior_feat], dim=-1)
        h = self.dropout1(torch.nn.functional.gelu(self.ln1(self.fc1(x))))
        h = self.dropout2(torch.nn.functional.gelu(self.ln2(self.fc2(h))))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# SpectreModel — composes Φ, Ψ, σ
# ---------------------------------------------------------------------------


class SpectreModel(nn.Module):
    """Composes Φ, Ψ, σ per spec §10.3."""

    def __init__(
        self,
        vocab: Vocab,
        prior_dropout_p: float = 0.2,
        use_atom_sab2: bool = True,
        static_tag_predicates: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.skeleton_encoder = SkeletonEncoder(
            vocab,
            use_atom_sab2=use_atom_sab2,
            static_tag_predicates=static_tag_predicates,
        )
        self.context_encoder = ContextEncoder()
        self.scorer = Scorer(prior_dropout_p=prior_dropout_p)

    @property
    def empty_context(self) -> Tensor:
        """Returns the learned ``c_0`` vector for inference-time |F|=0 usage."""
        return self.context_encoder.c0

    # --- Encoder fast-paths -------------------------------------------------

    def encode_pool(
        self,
        op_ids: Tensor,
        op_arg_type_ids: Tensor,
        op_arg_local_ids: Tensor,
        op_mask: Tensor,
        s0_pred_ids: Tensor,
        s0_arg_type_ids: Tensor,
        s0_arg_local_ids: Tensor,
        s0_atom_mask: Tensor,
        s0_type_histogram: Tensor,
        sL_pred_ids: Tensor,
        sL_arg_type_ids: Tensor,
        sL_arg_local_ids: Tensor,
        sL_atom_mask: Tensor,
    ) -> Tensor:
        """Run Φ over a pool slice. ``s_0`` carries no skeleton dim."""
        # Replicate s_0 type histogram to per-skeleton (matches spec §4.1
        # which expects ``sL_type_histogram`` per skeleton; in RT2D no
        # operator add/deletes objects, so it equals ``s0_type_histogram``).
        bsz, k = op_ids.shape[0], op_ids.shape[1]
        sL_type_hist = s0_type_histogram.unsqueeze(1).expand(bsz, k, -1)
        return self.skeleton_encoder(
            op_ids,
            op_arg_type_ids,
            op_arg_local_ids,
            op_mask,
            s0_pred_ids,
            s0_arg_type_ids,
            s0_arg_local_ids,
            s0_atom_mask,
            s0_type_histogram,
            sL_pred_ids,
            sL_arg_type_ids,
            sL_arg_local_ids,
            sL_atom_mask,
            sL_type_hist,
        )

    def encode_context(self, f_embeddings: Tensor, f_mask: Tensor) -> Tensor:
        """Run Ψ over the (per-example) failure set."""
        return self.context_encoder(f_embeddings, f_mask)

    def score(
        self,
        e_R: Tensor,
        c: Tensor,
        r_priors: Tensor,
        prior_dropout: bool = False,
    ) -> Tensor:
        """Run σ to produce per-skeleton logits over the R-pool."""
        return self.scorer(e_R, c, r_priors, prior_dropout=prior_dropout)

    # --- Whole-batch forward -----------------------------------------------

    def forward(self, batch: SpectreBatch) -> Tensor:
        """End-to-end SPECTRE forward over a ``SpectreBatch`` (spec §10.3)."""
        e_R = self.encode_pool(
            batch.r_op_ids,
            batch.r_op_arg_type_ids,
            batch.r_op_arg_local_ids,
            batch.r_op_mask,
            batch.s0_pred_ids,
            batch.s0_arg_type_ids,
            batch.s0_arg_local_ids,
            batch.s0_atom_mask,
            batch.s0_type_histogram,
            batch.r_sL_pred_ids,
            batch.r_sL_arg_type_ids,
            batch.r_sL_arg_local_ids,
            batch.r_sL_atom_mask,
        )
        e_F = self.encode_pool(
            batch.f_op_ids,
            batch.f_op_arg_type_ids,
            batch.f_op_arg_local_ids,
            batch.f_op_mask,
            batch.s0_pred_ids,
            batch.s0_arg_type_ids,
            batch.s0_arg_local_ids,
            batch.s0_atom_mask,
            batch.s0_type_histogram,
            batch.f_sL_pred_ids,
            batch.f_sL_arg_type_ids,
            batch.f_sL_arg_local_ids,
            batch.f_sL_atom_mask,
        )
        c = self.encode_context(e_F, batch.f_mask)
        return self.score(e_R, c, batch.r_priors, prior_dropout=self.training)


@dataclass
class ModelInfo:
    """Diagnostic metadata returned by :func:`build_model_info`."""

    op_mlp_in_features: int
    atom_proj_in_features: int
    num_parameters: int


def build_model_info(model: SpectreModel) -> ModelInfo:
    """Inspector for §11.1 #5 (vocab arity sourcing smoke test)."""
    op_in = model.skeleton_encoder.op_enc.in_features
    atom_in = model.skeleton_encoder.state_enc.atom_in_features
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelInfo(
        op_mlp_in_features=op_in, atom_proj_in_features=atom_in, num_parameters=n
    )
