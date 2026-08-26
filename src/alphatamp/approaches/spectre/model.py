"""SPECTRE v3 ranker -- the same job as v2.2, re-derived from three ideas.

v3 keeps v2.2's contribution (a relational, tag-joined, object-centric geometric encoder
scored listwise) and removes the accretions around it: the data-dependent prior knob,
the five bespoke fact types, the inert packing certificate, the part-zeroed global
token.
See ``docs/SPECTRE_v3_proposal.md``.

**The exact-absence invariant (D-8).** Every v3 feature is behind a flag on
:class:`SpectreConfig`, and with every flag off this model *is* deployed v2.2 -- not
"equivalent to" it but literally built from the same submodule classes under the same
attribute names. Two consequences, both load-bearing:

1. A v2.2 checkpoint loads with ``strict=True``, so
   ``tests/approaches/spectre/test_v3_equivalence.py`` can replay the frozen dd2d_v3
   comparison cache through the v3 code path and demand *identical decisions*. That
   oracle is what makes the later data-path rewrites (the domain adapter, the record
   tokens) safe to do at all.
2. The oracle stays alive until the position encoding is replaced, because no gate
   before that changes what compat mode constructs.

So new capability arrives as *additional* config-selected submodules; it never mutates
the compat path. When a flag is on, the state dict legitimately differs and the
equivalence test is expected to be run in compat mode only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from alphatamp.approaches.spectre.encoders import (
    D_REL,
    D_TAG,
    MAX_TAGS_DEFAULT,
    AtomProfileEncoder,
    AuxHead,
    CandidateEncoder,
    CrossAttentionScorer,
    FactEncoder,
    SceneEncoder,
    SpectreBatch,
)
from alphatamp.approaches.spectre.layers import D_MODEL, FFN_DIM, N_HEADS
from alphatamp.approaches.spectre.tags import PAD_TAG

DROPOUT = 0.1

# Record-token dims. `MAX_RECORD_ARGS` / `MAX_RECORD_CULPRITS` cap how many objects one
# record names in each role; DD2D queries are unary and a grasp is blocked by a handful
# of objects, so these are generous.
MAX_RECORD_ARGS = 4
MAX_RECORD_CULPRITS = 8
D_SCHEMA = 32
N_RECORD_SCALARS = 4  # [depth j/L, effort (log1p, scaled), exhausted, effort_is_total]

# State-delta dims (`s_j` relative to `s_0`, §6.1). `MAX_DELTA_ATOMS` caps how many
# atoms one role contributes; measured on dd2d_v4 the maxima are |added| = 4 and
# |deleted| = 5, so 8 is slack and truncation never fires. It is a pooled sequence axis,
# so it appears in no parameter shape and can be raised for another domain for free.
MAX_DELTA_ATOMS = 8
D_PRED = 32
D_DELTA = 32

# Rung-1 evidence-step stream (docs/failed_records_fix.md F-A). Shared with the tensorizer
# (`dataset` imports these), so there is one source of truth for the widths.
N_STEP_SCALARS = 3  # [exhausted, log1p(effort)/10, effort_is_total]
N_STEP_STATUS = 2  # failed-here (1) / succeeded-and-blamed (2); 0 = pad
MAX_ATTEMPTS = 32  # attempt-segment table size (deploy |F| tops out at the pool cap)
D_STEP_STATUS = 16
D_STEP_ATTEMPT = 16


class RecordEncoder(nn.Module):
    """One observed failure -> one token, with the object roles kept apart.

    Replaces `FactEncoder`'s hand-built type vocabulary with the domain's own operator
    schemas, and finally consumes the scalars v2.2 harvested and then dropped on the
    floor (`Fact.scalars` never reached the tensorizer). No tier embedding: only hint-
    tier evidence ever entered the network, so it was a constant column.
    """

    def __init__(
        self,
        n_schemas: int,
        max_tags: int,
        dropout_p: float = DROPOUT,
        n_predicates: int = 0,
        max_pred_arity: int = 0,
        state_delta: bool = False,
    ) -> None:
        super().__init__()
        self.schema_emb = nn.Embedding(n_schemas + 1, D_SCHEMA, padding_idx=0)
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        self.proj = nn.Sequential(
            nn.Linear(D_SCHEMA + 2 * D_TAG + N_RECORD_SCALARS, D_MODEL),
            nn.Dropout(dropout_p),
            nn.LayerNorm(D_MODEL),
        )
        # The delta enters as an ADDITIVE, ZERO-INITIALIZED branch rather than by
        # widening `proj[0]`. Widening re-randomizes every weight in that layer
        # (measured: 0.177 max shift on the shared block against a kaiming bound of
        # 0.100), which is the same init confound `SpectreConfig` warns about for
        # `n_prior_feats` -- the flag would then change the draw as well as the
        # features. Built LAST, and `self.records` is itself built last in
        # `SpectreModel`, so every pre-existing parameter keeps its exact
        # initialization and a flag-on model is functionally identical to flag-off at
        # step 0. Anything measured afterwards is the feature.
        self.pred_emb: Optional[nn.Embedding] = None
        self.atom_proj: Optional[nn.Linear] = None
        self.delta_proj: Optional[nn.Linear] = None
        self.delta_arity = 0
        if state_delta:
            if n_predicates <= 0:
                raise ValueError(
                    "use_state_delta needs n_predicates from the vocab; a 1-row "
                    "embedding table would train silently and mean nothing"
                )
            self.delta_arity = max(max_pred_arity, 1)
            self.pred_emb = nn.Embedding(n_predicates + 1, D_PRED, padding_idx=0)
            self.atom_proj = nn.Linear(D_PRED + self.delta_arity * D_TAG, D_DELTA)
            self.delta_proj = nn.Linear(2 * D_DELTA, D_MODEL)
            nn.init.zeros_(self.delta_proj.weight)
            nn.init.zeros_(self.delta_proj.bias)

    @staticmethod
    def _pool(emb: Tensor, ids: Tensor) -> Tensor:
        """Masked mean over a role's tag slots; zeros when the role is empty."""
        present = (ids != PAD_TAG).float().unsqueeze(-1)
        return (emb * present).sum(dim=2) / present.sum(dim=2).clamp(min=1.0)

    def _delta(self, pred_ids: Tensor, arg_tags: Tensor) -> Tensor:
        """``(B, R, 2*D_DELTA)`` from the per-role atom sets; exact zeros when empty.

        Two properties are load-bearing and easy to lose:

        - an atom's argument slots are **concatenated positionally**, never pooled, so
          ``p(a, b)`` and ``p(b, a)`` do not collide. DD2D is all-unary and would never
          show the difference, which is exactly why it is pinned by a test;
        - the per-atom projection happens **before** the pool over atoms, so
          ``{on-buffer(o1), holding(o2)}`` and ``{on-buffer(o2), holding(o1)}`` differ.
          Concatenating the roles and pooling afterwards would make them identical.

        An empty role pools to exactly zero (masked sum over nothing, denominator
        clamped), so ``j = 0`` -- about half of the aggregated tokens -- contributes
        nothing rather than a bias, and the first attempt of a rollout stays purely
        static.
        """
        assert self.pred_emb is not None and self.atom_proj is not None
        b, r = pred_ids.shape[0], pred_ids.shape[1]
        present = pred_ids.ne(0).unsqueeze(-1).float()
        args = self.tag_emb(arg_tags).reshape(*arg_tags.shape[:-1], -1)
        atom = self.atom_proj(torch.cat([self.pred_emb(pred_ids), args], dim=-1))
        pooled = (atom * present).sum(dim=3) / present.sum(dim=3).clamp(min=1.0)
        return pooled.reshape(b, r, 2 * D_DELTA)

    def forward(
        self,
        schema_ids: Tensor,
        arg_tags: Tensor,
        culprit_tags: Tensor,
        scalars: Tensor,
        mask: Tensor,
        delta_pred_ids: Optional[Tensor] = None,
        delta_arg_tags: Optional[Tensor] = None,
    ) -> Tensor:
        parts = [
            self.schema_emb(schema_ids),
            self._pool(self.tag_emb(arg_tags), arg_tags),
            self._pool(self.tag_emb(culprit_tags), culprit_tags),
            scalars,
        ]
        hidden = self.proj[0](torch.cat(parts, dim=-1))
        if self.delta_proj is not None:
            # Substituted zeros rather than a skipped branch: a batch whose records all
            # sit at j=0 must encode identically to the same record beside a batch-mate
            # that has a delta. Deploy collates ONE example at a time, so the two cases
            # are not hypothetical.
            if delta_pred_ids is None or delta_arg_tags is None:
                b, r = schema_ids.shape
                delta_pred_ids = schema_ids.new_zeros(b, r, 2, MAX_DELTA_ATOMS)
                delta_arg_tags = schema_ids.new_zeros(
                    b, r, 2, MAX_DELTA_ATOMS, self.delta_arity
                )
            hidden = hidden + self.delta_proj(
                self._delta(delta_pred_ids, delta_arg_tags)
            )
        return self.proj[2](self.proj[1](hidden)) * mask.unsqueeze(-1)


class RecordStepEncoder(nn.Module):
    """Rung-1 evidence-step tokens (docs/failed_records_fix.md F-A rung 1).

    Each evidence step's geometry is embedded by the **shared** ``CandidateEncoder``
    (passed in as ``base`` from ``self.cands.encode_steps`` so its weights are not
    double-registered), which makes a failed ``place_short(b)`` and the current
    candidate's ``place_short(b)`` identical vectors — the join the step-join then does
    becomes similarity in a shared space. On top of that, an **additive, zero-init**
    enrichment adds the status, the attempt segment, the count-weighted culprit role, and
    the effort scalars; zero-init means at step 0 an evidence token equals its shared step
    vector, so the enrichment's contribution is measured, not assumed.
    """

    def __init__(self, max_tags: int, dropout_p: float = DROPOUT) -> None:
        super().__init__()
        self.status_emb = nn.Embedding(N_STEP_STATUS + 1, D_STEP_STATUS, padding_idx=0)
        self.attempt_emb = nn.Embedding(MAX_ATTEMPTS + 1, D_STEP_ATTEMPT, padding_idx=0)
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        self.enrich = nn.Linear(
            D_STEP_STATUS + D_STEP_ATTEMPT + D_TAG + N_STEP_SCALARS, D_MODEL
        )
        nn.init.zeros_(self.enrich.weight)
        nn.init.zeros_(self.enrich.bias)

    def forward(
        self,
        base: Tensor,
        status: Tensor,
        attempt: Tensor,
        cul_tags: Tensor,
        cul_counts: Tensor,
        scalars: Tensor,
        mask: Tensor,
    ) -> Tensor:
        present = (cul_tags != PAD_TAG).float().unsqueeze(-1)  # (B, S, C, 1)
        w = cul_counts.unsqueeze(-1) * present  # weight the culprit role by log-count
        cul = (self.tag_emb(cul_tags) * w).sum(2) / w.sum(2).clamp(min=1e-6)
        # attempt is 0-indexed; shift by +1 so the first real attempt is not the pad slot.
        enrich_in = torch.cat(
            [
                self.status_emb(status),
                self.attempt_emb((attempt + 1).clamp(max=MAX_ATTEMPTS)),
                cul,
                scalars,
            ],
            dim=-1,
        )
        return (base + self.enrich(enrich_in)) * mask.unsqueeze(-1).float()


def _arg_bitmask(tags: Tensor) -> Tensor:
    """Bitmask over the object tags in the last axis (tag ``t`` -> bit ``t``; PAD=0 skipped).

    Episode-local tags are ids in ``[1, max_tags]`` (``max_tags`` <= 32, well within int64's
    62 usable bits), so a step's / record's object set becomes one int64 and set relations
    are bitwise: intersection = ``a & b != 0``, set-equality = ``a == b`` — exact and cheap,
    with no per-pair broadcast over the argument axes.
    """
    out = tags.new_zeros(tags.shape[:-1])
    for a in range(tags.shape[-1]):
        t = tags[..., a]
        out = torch.bitwise_or(
            out,
            torch.where(
                t > 0,
                torch.bitwise_left_shift(torch.ones_like(t), t),
                torch.zeros_like(t),
            ),
        )
    return out


def step_match_indicators(
    cand_op_ids: Tensor,
    cand_arg_tags: Tensor,
    rec_schema_ids: Tensor,
    rec_arg_tags: Tensor,
    rec_culprit_tags: Tensor,
) -> Tensor:
    """Exact candidate-step × record-token match indicators (F-B1), ``(B, K*L, R, 3)``.

    The three equality tests, in ``StepJoin.bias_gate`` order: (1) the step touches an
    object a record blamed (``args ∩ culprits ≠ ∅``); (2) the step **is** the failed step
    (same schema *and* same argument set); (3) the step touches the failed query's own
    objects (``args ∩ rec_args ≠ ∅``). Equality-only and domain-agnostic — reads no scalar.
    """
    b = cand_op_ids.shape[0]
    cand_op = cand_op_ids.reshape(b, -1)[:, :, None]  # (B, KL, 1)
    cand_bm = _arg_bitmask(cand_arg_tags).reshape(b, -1)[:, :, None]  # (B, KL, 1)
    rec_schema = rec_schema_ids[:, None, :]  # (B, 1, R)
    rec_cul_bm = _arg_bitmask(rec_culprit_tags)[:, None, :]  # (B, 1, R)
    rec_arg_bm = _arg_bitmask(rec_arg_tags)[:, None, :]  # (B, 1, R)
    touch_cul = ((cand_bm & rec_cul_bm) != 0).float()
    is_failed = (
        (cand_op == rec_schema) & (cand_bm == rec_arg_bm) & (cand_bm != 0)
    ).float()
    touch_q = ((cand_bm & rec_arg_bm) != 0).float()
    return torch.stack([touch_cul, is_failed, touch_q], dim=-1)  # (B, KL, R, 3)


class StepJoin(nn.Module):
    """Pre-pooling evidence interaction (docs/failed_records_fix.md F-B2).

    The scorer's own evidence query is the *pooled* candidate embedding, so a step-level
    candidate×evidence join is not representable there. This module lets the candidate's
    **per-step** tokens cross-attend over the evidence-step memory *before* the PMA pool,
    which makes that join representable at all. Zero-init output projection + residual, so
    a flag-on model is identical to flag-off at step 0 and old checkpoints load
    ``strict=True`` (the module only exists when ``use_step_join`` is set).

    **Match-primitive edge biases (F-B1, ``match_bias``).** Soft attention is weak at the
    near-*exact* relational join this needs (does a candidate step re-touch the object that
    blocked a failure? is it the failed step itself?). When ``match_bias`` is on, exact
    equality indicators between each candidate step and each record token are added to the
    pre-softmax attention scores through **learned, zero-initialized** per-indicator gates
    (so it is a no-op at step 0). The indicators compute **equality only** — domain-agnostic
    and content-free ("the model is told *what matches*, it learns *what matching means*") —
    they read no compiled coverage/waste/repeat scalar. The gate scalars are the only new
    parameters, so ``match_bias`` off is byte-identical to the plain step-join.
    """

    #: attention-bias indicators, in the order the ``bias_gate`` scalars weight them.
    N_MATCH_INDIC = (
        3  # [touches-a-culprit, is-the-failed-step, touches-the-failed-query]
    )

    def __init__(self, dropout_p: float = DROPOUT, match_bias: bool = False) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=dropout_p, batch_first=True
        )
        self.out = nn.Linear(D_MODEL, D_MODEL)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.match_bias = match_bias
        if match_bias:
            # Zero-init: at step 0 the bias is 0, so a match_bias model == a plain
            # step-join model until the gates learn a per-indicator weight.
            self.bias_gate = nn.Parameter(torch.zeros(self.N_MATCH_INDIC))

    def forward(
        self,
        step: Tensor,
        memory: Tensor,
        mem_mask: Tensor,
        indicators: Optional[Tensor] = None,
    ) -> Tensor:
        b, k, ell, d = step.shape
        q = step.reshape(b, k * ell, d)
        has = mem_mask.any(dim=1)  # (B,) — some evidence to attend to
        safe = mem_mask.clone()
        safe[~has, 0] = True  # dodge the all-masked-row NaN; zeroed straight after
        if self.match_bias and indicators is not None:
            # Fold the match-bias and the key padding into ONE float attn_mask (mixing a
            # bool key_padding_mask with a float attn_mask is deprecated): additive
            # per-indicator bias, then −inf at padded keys, broadcast over heads.
            bias = (indicators * self.bias_gate).sum(dim=-1)  # (B, K*L, R)
            bias = bias.masked_fill((~safe).unsqueeze(1), float("-inf"))
            out, _ = self.attn(
                q, memory, memory, attn_mask=bias.repeat_interleave(N_HEADS, dim=0)
            )
        else:
            out, _ = self.attn(q, memory, memory, key_padding_mask=~safe)
        out = out * has.view(b, 1, 1).float()
        return step + self.out(out).reshape(b, k, ell, d)


class EvidenceCrossAttentionScorer(CrossAttentionScorer):
    """Scorer with a **separate attention channel for evidence**.

    v2.2 concatenates scene tokens, the global token and the evidence tokens into one
    memory and runs a single cross-attention over it. That is the architectural reason
    the record tokens end up inert, and it is a competition the evidence cannot win:

    - **One softmax must split its mass.** With ~10 scene tokens against up to 2045
      record tokens, the geometry that actually determines feasibility is outnumbered
      ~200:1; with aggregation it is still ~3:1 and grows with |F|.
    - **Geometry is reliably useful, evidence is noisy.** The loss-minimizing policy for
      a *shared* attention budget is therefore to spend it on geometry and ignore
      evidence -- exactly what the ``suppress_records`` diagnostic measured
      (16.17 -> 16.40, i.e. the trained model had already learned to discard its own
      records).

    Two channels remove the competition: the candidate attends over ``[scene ; global]``
    and, independently, over the evidence memory, and the head sees both. Evidence can
    now be attended to *without* giving up geometry, so a useful record no longer has to
    out-compete the scene to be read.

    Fully domain-agnostic -- it is a change to how tokens are consumed, not to what they
    are. The head widens from ``2*D_MODEL`` to ``3*D_MODEL``, so enabling it retires the
    D-8 oracle exactly as the other v3 architecture switches do.
    """

    def __init__(
        self,
        n_overlap_feats: int = 0,
        n_prior_feats: int = 0,
        dropout_p: float = DROPOUT,
    ) -> None:
        super().__init__(n_overlap_feats, n_prior_feats, dropout_p)
        self.evid_attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=dropout_p, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(3 * D_MODEL + n_overlap_feats + n_prior_feats, FFN_DIM),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(FFN_DIM, 1),
        )

    def forward(  # type: ignore[override]
        self,
        cand_emb: Tensor,
        scene_tok: Tensor,
        obj_mask: Tensor,
        glob_feats: Tensor,
        overlap: Optional[Tensor] = None,
        fact_tok: Optional[Tensor] = None,
        fact_mask: Optional[Tensor] = None,
        prior: Optional[Tensor] = None,
        glob_extra: Optional[Tensor] = None,
    ) -> Tensor:
        b, k, _ = cand_emb.shape
        # `glob_extra` carries the 0-ary init/goal atom profile (zero-init), so `None`
        # reduces this to the pre-atom-input `glob_proj(glob_feats)` exactly.
        glob = self.glob_proj(glob_feats)
        if glob_extra is not None:
            glob = glob + glob_extra
        glob = glob.unsqueeze(1)
        memory = torch.cat([scene_tok, glob], dim=1)
        key_pad = torch.cat(
            [~obj_mask, torch.zeros(b, 1, dtype=torch.bool, device=obj_mask.device)],
            dim=1,
        )
        attended, _ = self.attn(cand_emb, memory, memory, key_padding_mask=key_pad)

        ev = cand_emb.new_zeros(b, k, D_MODEL)
        if fact_tok is not None and fact_tok.shape[1] > 0 and fact_mask is not None:
            # A batch row with no records would be an all-True key-padding mask, which
            # makes MultiheadAttention emit NaN rather than an empty result. Attend
            # under a mask that always leaves one key live, then zero those rows
            # afterwards -- the same guard the v1 encoder uses.
            has = fact_mask.any(dim=1)
            safe = fact_mask.clone()
            safe[~has, 0] = True
            out, _ = self.evid_attn(
                cand_emb, fact_tok, fact_tok, key_padding_mask=~safe
            )
            ev = out * has.view(b, 1, 1)

        parts = [cand_emb, attended, ev]
        if self.n_overlap_feats:
            parts.append(
                overlap
                if overlap is not None
                else cand_emb.new_zeros(b, k, self.n_overlap_feats)
            )
        pr = cand_emb.new_zeros(b, k, self.n_prior_feats) if prior is None else prior
        if self.n_prior_feats:
            parts.append(pr)
        logit = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        if self.n_prior_feats:
            logit = logit + self.prior_gate(pr).squeeze(-1)
        return logit


class ResidualEvidenceScorer(CrossAttentionScorer):
    """X2: the failure-record channel as a zero-init, |F|-gated **residual** over static.

    ``logit = static_logit + g(|F|) · adjustment``, where

    - ``static_logit`` = ``head([cand_emb ; geometry-attended])`` -- exactly the pure-static
      ``CrossAttentionScorer`` computation (2·D_MODEL head over the candidate and a
      ``[scene ; glob]``-only attention). So ``self.{attn, glob_proj, head}`` key- and
      shape-match a pure-static checkpoint and are what ``--init-static-from`` warm-starts and
      ``--freeze-static`` freezes; before any residual training the model reproduces the static
      ranker bit-for-bit.
    - ``adjustment`` = ``adaptive_head([ev ; overlap])`` where ``ev`` is the record/evidence
      cross-attention (the existing v3 channel), **output layer zero-initialized** so it is 0 at
      step 0 -> ``logit ≡ static_logit`` at init. This is the ``prior_gate``/``delta_proj`` idiom.
    - ``g`` = a tiny MLP on ``log1p(|F|)`` through a sigmoid, its output layer zero-init so
      ``g = σ(0) = 0.5`` **flat** across |F| at init: a NEUTRAL gate that is free to learn to
      amplify *or* suppress the residual by context size -- it is not pre-biased to shut off in
      large contexts (with the static trunk frozen the interference cause is removed, so the
      residual may help at every stratum).

    Property, not hope: with the static half frozen + warm-started and the adjustment zero-init,
    attaching this channel cannot make the ranker worse than static at init, and training only
    *adds* the residual. Overlap columns are routed into the adjustment (they are
    failure-conditioned), keeping the static head at 2·D_MODEL and warm-start-compatible; the X2
    probe runs with no overlap, so ``ev`` alone drives the residual.
    """

    GATE_HID = 16

    def __init__(
        self,
        n_overlap_feats: int = 0,
        n_prior_feats: int = 0,
        dropout_p: float = DROPOUT,
    ) -> None:
        # n_overlap=0 to super so the STATIC head is 2·D_MODEL(+n_prior) and key-matches a
        # pure-static checkpoint; the overlap feats (if any) feed the adjustment instead.
        super().__init__(0, n_prior_feats, dropout_p)
        self.adaptive_overlap = n_overlap_feats
        self.evid_attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=dropout_p, batch_first=True
        )
        self.adaptive_head = nn.Sequential(
            nn.Linear(D_MODEL + n_overlap_feats, FFN_DIM),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(FFN_DIM, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(1, self.GATE_HID), nn.GELU(), nn.Linear(self.GATE_HID, 1)
        )
        # Zero-init the adjustment output (step-0 residual is 0 -> logit ≡ static) and the gate
        # output (g = σ(0) = 0.5 flat -> neutral, no suppression prior).
        for seq in (self.adaptive_head, self.gate):
            last = seq[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(  # type: ignore[override]
        self,
        cand_emb: Tensor,
        scene_tok: Tensor,
        obj_mask: Tensor,
        glob_feats: Tensor,
        overlap: Optional[Tensor] = None,
        fact_tok: Optional[Tensor] = None,
        fact_mask: Optional[Tensor] = None,
        prior: Optional[Tensor] = None,
        glob_extra: Optional[Tensor] = None,
        context_size: Optional[Tensor] = None,
    ) -> Tensor:
        b, k, _ = cand_emb.shape
        # --- static path (identical to pure-static CrossAttentionScorer, facts absent) ---
        glob = self.glob_proj(glob_feats)
        if glob_extra is not None:
            glob = glob + glob_extra
        glob = glob.unsqueeze(1)
        memory = torch.cat([scene_tok, glob], dim=1)
        key_pad = torch.cat(
            [~obj_mask, torch.zeros(b, 1, dtype=torch.bool, device=obj_mask.device)],
            dim=1,
        )
        attended, _ = self.attn(cand_emb, memory, memory, key_padding_mask=key_pad)
        pr = cand_emb.new_zeros(b, k, self.n_prior_feats) if prior is None else prior
        static_parts = [cand_emb, attended]
        if self.n_prior_feats:
            static_parts.append(pr)
        static_logit = self.head(torch.cat(static_parts, dim=-1)).squeeze(-1)
        if self.n_prior_feats:
            static_logit = static_logit + self.prior_gate(pr).squeeze(-1)

        # --- residual path: record/evidence cross-attention (same guard as evid channel) ---
        ev = cand_emb.new_zeros(b, k, D_MODEL)
        if fact_tok is not None and fact_tok.shape[1] > 0 and fact_mask is not None:
            has = fact_mask.any(dim=1)
            safe = fact_mask.clone()
            safe[~has, 0] = True
            out, _ = self.evid_attn(
                cand_emb, fact_tok, fact_tok, key_padding_mask=~safe
            )
            ev = out * has.view(b, 1, 1)
        adap_parts = [ev]
        if self.adaptive_overlap:
            adap_parts.append(
                overlap
                if overlap is not None
                else cand_emb.new_zeros(b, k, self.adaptive_overlap)
            )
        adjustment = self.adaptive_head(torch.cat(adap_parts, dim=-1)).squeeze(-1)

        # --- neutral |F|-gate; broadcast the per-episode g over candidates ---
        if context_size is None:
            cs = cand_emb.new_zeros(b, 1)
        else:
            cs = torch.log1p(context_size.to(cand_emb.dtype)).view(b, 1)
        g = torch.sigmoid(self.gate(cs))  # (B, 1), = 0.5 flat at init
        return static_logit + g * adjustment


N_OVERLAP_COV = 4
"""``[dead, jaccard, coverage, waste]``.

``coverage`` and ``waste`` are the **observed** necessity features -- recall / precision
over the failures the refiner reported. They are the *unified* definition, computed in
:mod:`alphatamp.approaches.spectre.unified_evidence` (spec:
``docs/unified_culprits_coverage_waste.md``) over a filtered culprit pool, **not** the
older object-set ratio, which was removed. This line only labels the column vector;
``unified_evidence`` is the authoritative definition -- do not restate a formula here.
"""


@dataclass(frozen=True)
class SpectreConfig:
    """Architecture switches.

    Every v3 feature defaults **off**, so the default config reproduces deployed v2.2
    exactly (D-8).

    ``n_prior_feats`` is retained only so a pre-v3 checkpoint that *was* trained with
    the short-first prior still loads for comparison. The deployed dd2d_v3 model has it
    off, and v3 does not reintroduce it: the prior was a per-dataset hand switch that
    diverged training on the easier collection (``decisions.md`` 2026-07-25). Note the
    v2 scorer couples ``n_prior_feats > 0`` to a zero-init of the head's output layer,
    so prior-on and prior-off differ in initialization as well as in features -- a
    confound to remember when reading any historical prior on/off delta.
    """

    n_overlap_feats: int = 0
    n_prior_feats: int = 0
    max_tags: int = MAX_TAGS_DEFAULT
    dropout_p: float = DROPOUT
    # Scene-relation width: the anchor-free ``[area, sinθ, cosθ]`` triple (3). The
    # target-anchored offsets, target area ratio and privileged ``concave`` flag were cut
    # (see ``encoders.D_REL``). Persisted, because it changes ``scene.rel_proj``'s shape;
    # a checkpoint is bound to the width it was trained on (docs/decisions 2026-08-08).
    d_rel: int = D_REL
    # Scene point/pose widths. Default 2/3 reproduce the 2D boundary-ring + (x,y,theta)
    # path byte-for-byte (DD2D/SB2D); Restock3D opts into 3/4 (a 3D point cloud + (x,y,z,
    # yaw) pose). Persisted, because they change ``FootprintEncoder``'s and
    # ``SceneEncoder``'s input Linear shapes; a checkpoint is bound to them. See
    # docs/decisions 2026-08-18.
    point_dim: int = 2
    pose_dim: int = 3
    # --- PointSetEncoder upgrade switches (doc pointset_encoder_upgrade.md) ---
    # All default off / seeds=1, so the default config still builds the v1
    # FootprintEncoder and reproduces the deployed state dict byte-for-byte (D-8).
    # ``use_pca_feats`` /
    # ``edgeconv_k`` also change what the tensorizer *emits* (C_pt / knn), so both are
    # persisted; the other three change submodule shapes. Any switch on (or pma_seeds>1)
    # selects ``PointSetEncoder`` instead of ``FootprintEncoder``.
    use_pca_feats: bool = False  # per-point normal/curvature/flatness columns
    use_edgeconv: bool = False  # one DGCNN EdgeConv interaction layer
    use_point_sab: bool = (
        False  # optional global SAB over the point set (off by default)
    )
    pma_seeds: int = 1  # MultiSeedPMA seeds (1 == single-seed pool width)
    edgeconv_k: int = 0  # kNN degree; 0 == resolve by dim (4 in 2D, 6 in 3D)
    # --- v3 feature switches (added by later gates; all no-ops here) ---
    use_records: bool = False  # G6: role-separated FailureRecord tokens
    use_necessity: bool = False  # G8: necessity head + its candidate features
    # Give evidence its own cross-attention channel instead of making it compete with
    # the scene inside one softmax. See EvidenceCrossAttentionScorer.
    evidence_attn: bool = False
    # Observed coverage/waste appended to cand_overlap (width 2 -> 4).
    coverage_feats: bool = False
    # §6.1's `s_j`: each record token also carries the abstract state at the failing
    # step, as the delta from s_0. Additive and zero-initialized inside `RecordEncoder`,
    # so a pre-flag v3 checkpoint still loads `strict=True` -- D-8's discipline one
    # level down, against the *deployed v3* state dict rather than v2.2's.
    use_state_delta: bool = False
    # Vocab-derived sizing for the delta's predicate table, filled by whichever caller
    # holds the vocab (`run_training`, `load_checkpoint`) exactly as `max_arity` already
    # is. Not persisted: they are properties of the vocab, and `strict=True` is the
    # backstop if one ever moves under a checkpoint.
    n_predicates: int = 0
    max_pred_arity: int = 0
    # --- atom-input switches (doc spectre_atom_input_guide.md) ---
    # ``atom_mode="profiles"`` selects the ``AtomProfileEncoder`` submodule: it injects
    # per-object init/goal atom profiles (additive, zero-init) into scene tokens, plus a
    # 0-ary global term into the scorer's global token. Default "off" builds nothing new,
    # so the state dict is byte-identical to a pre-atom checkpoint (D-8). "tokens" is
    # reserved for Rung B (not built). ``use_init_atoms``/``use_goal_atoms`` only gate
    # emission + zero a profile half, so they never change the state dict. Sized from the
    # same ``n_predicates``/``max_pred_arity`` the record delta uses.
    atom_mode: str = "off"
    use_init_atoms: bool = True
    use_goal_atoms: bool = True
    # --- rung-1 learned-pathway switches (docs/failed_records_fix.md F-A/F-B2) ---
    # ``record_mode="steps"`` replaces the one-summary-token-per-record evidence memory
    # with the rung-1 evidence-STEP stream (failed step + culprit establishing steps,
    # shared-encoder embedded); ``use_step_join`` adds the pre-pooling StepJoin so
    # candidate step tokens cross-attend over that memory before pooling. Both default off
    # / "summary", building nothing new, so the state dict is byte-identical to a
    # pre-flag checkpoint (D-8). They select submodules AND change what the tensorizer
    # emits, so both are persisted and rebuilt at load time (like ``coverage_feats``).
    record_mode: str = "summary"
    use_step_join: bool = False
    # F-B1: add exact match-primitive edge biases (candidate-step × record-token equality
    # indicators) to the step-join's attention, through zero-init learned gates. Requires
    # ``use_step_join``; off = byte-identical (adds only the gate scalars when on). Reads no
    # compiled scalar — equality-only, domain-agnostic.
    step_join_match_bias: bool = False
    # X2 (docs/failed_records_fix_part2.md §3): train the record/evidence channel as a
    # zero-init |F|-gated RESIDUAL on top of a static base, selecting `ResidualEvidenceScorer`.
    # `logit = static_logit([cand_emb; geometry-attended]) + g(|F|)·adjustment([ev; overlap])`,
    # the adjustment output zero-initialized so step-0 ≡ static. The static half
    # (`scorer.{attn,glob_proj,head}`, a 2·D_MODEL head) key-matches a pure-static checkpoint,
    # so it warm-starts + freezes (train.py `--init-static-from`/`--freeze-static`). Off builds
    # nothing new, so the state dict is byte-identical to a pre-flag checkpoint (D-8). Persisted
    # + rebuilt at load time like the other architecture switches.
    residual_adaptive: bool = False


class SpectreModel(nn.Module):
    """The v3 listwise re-ranker.

    Submodule names (``scene`` / ``cands`` / ``facts`` / ``scorer`` / ``aux``) are fixed
    by the compat contract above -- renaming one silently breaks checkpoint loading, so
    ``test_v3_equivalence.py`` pins the key set.
    """

    def __init__(
        self,
        n_ops: int,
        max_arity: int,
        cfg: Optional[SpectreConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SpectreConfig()
        c = self.cfg
        # Scene width comes from the config: the anchor-free ``[area, sinθ, cosθ]``.
        self.scene = SceneEncoder(
            c.max_tags,
            c.dropout_p,
            d_rel=c.d_rel,
            point_dim=c.point_dim,
            pose_dim=c.pose_dim,
            use_pca_feats=c.use_pca_feats,
            use_edgeconv=c.use_edgeconv,
            use_point_sab=c.use_point_sab,
            pma_seeds=c.pma_seeds,
        )
        self.cands = CandidateEncoder(n_ops, c.max_tags, max_arity, c.dropout_p)
        self.facts = FactEncoder(c.max_tags, c.dropout_p)
        if c.residual_adaptive:
            scorer_cls: type[CrossAttentionScorer] = ResidualEvidenceScorer
        elif c.evidence_attn:
            scorer_cls = EvidenceCrossAttentionScorer
        else:
            scorer_cls = CrossAttentionScorer
        self.scorer = scorer_cls(c.n_overlap_feats, c.n_prior_feats, c.dropout_p)
        self.aux = AuxHead()
        # Additive by construction: the record encoder only exists when asked for, so a
        # default-config state dict is byte-identical to v2.2's (D-8) and the
        # equivalence oracle keeps loading.
        # Summary record tokens (rung 0) OR the rung-1 evidence-step stream, never both:
        # they encode the same failures, and `record_mode="steps"` uses `rec_steps` as the
        # evidence memory instead of `records`, so building both would leave dead params.
        self.records = (
            RecordEncoder(
                n_ops,
                c.max_tags,
                c.dropout_p,
                c.n_predicates,
                c.max_pred_arity,
                c.use_state_delta,
            )
            if (c.use_records and c.record_mode != "steps")
            else None
        )
        # Built last (after every other submodule), so a config-off model keeps its exact
        # init draws and an ``atom_mode="profiles"`` model is functionally identical to
        # off at step 0 (the AtomProfileEncoder injections are zero-init). "tokens" is
        # Rung B and not built.
        if c.atom_mode == "tokens":
            raise NotImplementedError(
                "atom_mode='tokens' (Rung B, per-atom tokens) is reserved but not "
                "built; see docs/spectre_atom_input_guide.md"
            )
        self.atoms = (
            AtomProfileEncoder(
                c.n_predicates,
                c.max_pred_arity,
                c.max_tags,
                c.use_init_atoms,
                c.use_goal_atoms,
            )
            if c.atom_mode == "profiles"
            else None
        )
        # Rung-1 learned pathway (F-A/F-B2), built last so an off-config model keeps its
        # exact init draws and a flag-on model is zero-init-identical at step 0. The step
        # encoder owns only the enrichment params; it reuses `self.cands` for the step
        # geometry via `encode_steps`, so those weights are never double-registered.
        self.rec_steps = (
            RecordStepEncoder(c.max_tags, c.dropout_p)
            if (c.use_records and c.record_mode == "steps")
            else None
        )
        self.step_join = (
            StepJoin(c.dropout_p, match_bias=c.step_join_match_bias)
            if c.use_step_join
            else None
        )
        if c.use_necessity:  # pragma: no cover - cut from v3 scope, see decisions.md
            raise NotImplementedError(
                "necessity conditioning was cut from v3 (decisions.md 2026-07-26): D2 "
                "showed the s2 deficit is within-length, which it does not address"
            )

    def forward(
        self, batch: SpectreBatch, overlap: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """``(logits (B, K), aux (B, M, 2))`` -- the v2.2 contract, unchanged.

        Logits are ``-inf`` at unavailable candidates (pads, and during a rollout the
        already-tried ones), so ``argmax`` is the next attempt.
        """
        # Atom profiles (doc spectre_atom_input_guide.md): zero when off/absent, so the
        # config-off path is `self.scene(batch)` / no `glob_extra` -- byte-identical.
        atom_obj_add: Optional[Tensor] = None
        atom_glob_add: Optional[Tensor] = None
        if self.atoms is not None:
            atom_obj_add, atom_glob_add = self.atoms(batch)
        scene_tok = self.scene(batch, atom_obj_add=atom_obj_add)
        # Evidence memory: rung-1 evidence steps when `record_mode="steps"`, else the v3
        # record tokens, else the legacy fact tokens. Never more than one -- they encode
        # the same failures, so stacking them would double-count the evidence. Built
        # before the candidates so the step-join can consume it.
        fact_tok = None
        fact_mask = batch.fact_mask
        if (
            self.rec_steps is not None
            and getattr(batch, "rec_step_op_ids", None) is not None
            and batch.rec_step_op_ids is not None
            and batch.rec_step_op_ids.shape[1] > 0
        ):
            base = self.cands.encode_steps(
                batch.rec_step_op_ids, batch.rec_step_pos, batch.rec_step_arg_tags
            )
            fact_tok = self.rec_steps(
                base,
                batch.rec_step_status,
                batch.rec_step_attempt,
                batch.rec_step_culprit_tags,
                batch.rec_step_culprit_counts,
                batch.rec_step_scalars,
                batch.rec_step_mask,
            )
            fact_mask = batch.rec_step_mask
        elif (
            self.records is not None
            and getattr(batch, "rec_schema_ids", None) is not None
            and batch.rec_schema_ids is not None
            and batch.rec_schema_ids.shape[1] > 0
        ):
            fact_tok = self.records(
                batch.rec_schema_ids,
                batch.rec_arg_tags,
                batch.rec_culprit_tags,
                batch.rec_scalars,
                batch.rec_mask,
                getattr(batch, "rec_delta_pred_ids", None),
                getattr(batch, "rec_delta_arg_tags", None),
            )
            fact_mask = batch.rec_mask
        elif (
            self.records is None
            and batch.fact_type_ids is not None
            and batch.fact_type_ids.shape[1] > 0
        ):
            fact_tok = self.facts(
                batch.fact_type_ids,
                batch.fact_tier_ids,
                batch.fact_arg_tags,
                batch.fact_mask,
            )
        # Candidate embedding, made evidence-aware by the step-join (F-B2) when enabled:
        # the candidate's per-step tokens cross-attend over the evidence memory before the
        # PMA pool. Otherwise the pooled candidate embedding, unchanged.
        if self.step_join is not None and fact_tok is not None:
            cstep = self.cands.encode_steps(
                batch.cand_op_ids, batch.cand_pos, batch.cand_arg_tags
            )
            # F-B1 match-primitive edge biases: exact candidate-step × record-token match
            # indicators, computed from the SUMMARY record fields (the record_mode="steps"
            # memory is the evidence steps, not these records, so bias only on the summary
            # path — `self.records is not None`).
            indic = None
            if (
                self.step_join.match_bias
                and self.records is not None
                and getattr(batch, "rec_schema_ids", None) is not None
                and batch.rec_schema_ids is not None
            ):
                indic = step_match_indicators(
                    batch.cand_op_ids,
                    batch.cand_arg_tags,
                    batch.rec_schema_ids,
                    batch.rec_arg_tags,
                    batch.rec_culprit_tags,
                )
            cstep = self.step_join(cstep, fact_tok, fact_mask, indic)
            cand_emb = self.cands.pool_steps(
                cstep, batch.cand_step_mask, batch.pool_mask
            )
        else:
            cand_emb = self.cands(batch)
        prior = batch.cand_prior if self.cfg.n_prior_feats else None
        if overlap is None and self.cfg.n_overlap_feats:
            overlap = batch.cand_overlap
        avail = batch.avail_mask if batch.avail_mask is not None else batch.pool_mask
        scorer_kwargs = {"glob_extra": atom_glob_add}
        if self.cfg.residual_adaptive:
            # |F| = number of in-context (failed) candidates per episode, for the residual's
            # neutral gate. Derived from avail_mask (avail=False ⟺ in context) intersected
            # with the real pool, so no new batch field and no frozen-shape change.
            scorer_kwargs["context_size"] = ((~avail) & batch.pool_mask).sum(dim=-1)
        logits = self.scorer(
            cand_emb,
            scene_tok,
            batch.obj_mask,
            batch.glob_feats,
            overlap,
            fact_tok,
            fact_mask,
            prior,
            **scorer_kwargs,
        )
        logits = logits.masked_fill(~avail, float("-inf"))
        return logits, self.aux(scene_tok)
