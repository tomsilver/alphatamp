"""SPECTRE v2.2 geometry-aware model (proposal §7).

Additive to v1 (`model.py`).
The static (t=0) architecture that conditions on object-centric geometry + episode-local
tags instead of anonymous local ids — the fix for v1's length-only collapse. Token
families (all width ``D_MODEL=64``, reusing v1's ``SetAttentionBlock``/``PMA``):

- **scene tokens** — per object: ``[tag ; footprint descriptor ; pose ; relation-to-target]``.
  The footprint descriptor is a *point-set* encoding of the boundary ring (not a radial
  profile — concave-safe). A couple of Set-Attention layers let objects attend to each
  other (the relational join).
- **candidate tokens** — a skeleton is a *program over the scene*: per operator, its schema
  embedding + position + argument slots holding the objects' **tags**. Pooled to one
  ``e(s)`` per candidate.
- **global token** — container/buffer geometry + pool statistics.
- **fact tokens / overlap features** — empty at static (t=0); wired for the Step-11 typed-
  evidence pathway. The scorer already accepts them so that step is additive.

**Scorer** — per-candidate cross-attention (candidate query over scene + global memory),
concatenated with computed overlap features → one logit; linear in pool size. **Aux head**
— per scene token → ``necessary``/``relevant`` logits (proposal §8).

The forward returns ``(B, K)`` logits with the same contract as v1's ``Scorer`` so the
rollout / PL-loss machinery is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import torch
from torch import Tensor, nn

from alphatamp.approaches.spectre.facts import N_FACT_TYPES
from alphatamp.approaches.spectre.model import (
    D_MODEL,
    FFN_DIM,
    N_HEADS,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)
from alphatamp.approaches.spectre.tags import PAD_TAG

# geometry token dims
N_BOUNDARY_POINTS = 32
D_TAG = 32
D_POINT = 16
D_DESCRIPTOR = 32
D_POSE = 8
D_REL = 8  # relation-to-target scalars projection
MAX_TAGS_DEFAULT = 32
DROPOUT = 0.1

# typed-evidence (Step 11) dims
MAX_FACT_ARGS = 12  # cap on a fact's argument list (mean-pooled); larger sets truncate
D_FACT_TIER = 8

# a-priori per-candidate prior features: [−index/K, −len/max_len] (default-order /
# short-first) — domain-agnostic planner signals available in any TAMP problem. Column 0 is
# the additive default-order residual anchor the geometry head only has to correct.
N_PRIOR = 2

# structural evidence features relating each candidate's action-set to the OBSERVED failed
# sets (Step 11 fix): [subset⊆blocked (sound proof-demotion — provably also-blocked),
# max-Jaccard-with-failed (hint)]. Domain-agnostic set relations. The unsound "blocked⊊subset
# ⇒ prefer longer" cue is deliberately excluded — it helps s3 but misleads easy strata.
N_OVERLAP = 2


@dataclass
class SpectreV2Batch:
    """Padded tensors for one batch of episodes (0 = pad; see ``dataset_v2``)."""

    # scene (objects)
    obj_tags: Tensor  # (B, N) long — episode-local tag ids (0 = pad)
    obj_boundary: Tensor  # (B, N, P, 2) float — resampled boundary ring, item frame
    obj_pose: Tensor  # (B, N, 3) float — (x, y, theta), normalized
    obj_rel: Tensor  # (B, N, D_REL) float — relation-to-target scalars
    obj_is_target: Tensor  # (B, N) float — 1 for the target object
    obj_mask: Tensor  # (B, N) bool — real object
    # candidates (skeletons)
    cand_op_ids: Tensor  # (B, K, L) long — operator-schema ids (0 = pad)
    cand_arg_tags: Tensor  # (B, K, L, A) long — arg-slot tags (0 = pad)
    cand_pos: Tensor  # (B, K, L) long — operator position index
    cand_step_mask: Tensor  # (B, K, L) bool — real operator
    pool_mask: Tensor  # (B, K) bool — real candidate
    # global
    glob_feats: Tensor  # (B, D_GLOBAL) float — buffer dims + pool stats
    # labels (for training)
    success_mask: Tensor  # (B, K) bool — feasible candidate
    aux_necessary: Tensor  # (B, N) float — necessary(o) target (or -1 = ignore)
    aux_relevant: Tensor  # (B, N) float — relevant(o) target (or -1 = ignore)
    # typed evidence (Step 11); all None in the static (t=0) path.
    fact_type_ids: Optional[Tensor] = None  # (B, F) long — 0 = pad
    fact_tier_ids: Optional[Tensor] = None  # (B, F) long — 0 = pad
    fact_arg_tags: Optional[Tensor] = None  # (B, F, MAX_FACT_ARGS) long — 0 = pad
    fact_mask: Optional[Tensor] = None  # (B, F) bool — real fact
    avail_mask: Optional[Tensor] = None  # (B, K) bool — candidate not yet tried (∉ F)
    # a-priori planner prior: [−index/K, −len/max_len] per candidate (default-order /
    # short-first). Known before any refinement — the scorer treats geometry as a residual
    # correction on this prior (init-toward-prior); ``None`` disables it.
    cand_prior: Optional[Tensor] = None  # (B, K, N_PRIOR) float
    # structural evidence features per candidate vs the observed failed sets (Step 11 fix);
    # 0 when no facts / static path. Lets the ranker use proofs by set-containment.
    cand_overlap: Optional[Tensor] = None  # (B, K, N_OVERLAP) float

    def to(self, device) -> "SpectreV2Batch":
        return SpectreV2Batch(
            **{  # type: ignore[arg-type]
                k: (v.to(device) if v is not None else None)
                for k, v in self.__dict__.items()
            }
        )


D_GLOBAL_IN = 6  # buffer (w, h, area) + pool_size + n_objects + mean_subset_size


class FootprintEncoder(nn.Module):
    """Point-set encoder over a fixed-size boundary ring → per-object descriptor.

    A shared per-point MLP followed by a masked PMA pool. Being a set over the true
    boundary points (not a radial function), it represents concave shapes faithfully and
    is invariant to the ring's starting vertex / point order.
    """

    def __init__(self, dropout_p: float = DROPOUT) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(2, D_POINT),
            nn.GELU(),
            nn.Linear(D_POINT, D_MODEL),
        )
        self.pool = PoolingByMultiheadAttention(
            dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p
        )
        self.out = nn.Linear(D_MODEL, D_DESCRIPTOR)

    def forward(self, boundary: Tensor, obj_mask: Tensor) -> Tensor:
        # boundary (B, N, P, 2) -> per-point features (B, N, P, D_MODEL)
        b, n, p, _ = boundary.shape
        feats = self.point_mlp(boundary)  # (B, N, P, D)
        feats = feats.reshape(b * n, p, D_MODEL)
        pmask = torch.ones(b * n, p, dtype=torch.bool, device=boundary.device)
        pooled = self.pool(feats, pmask).reshape(b, n, D_MODEL)  # (B, N, D)
        desc = self.out(pooled)  # (B, N, D_DESCRIPTOR)
        return desc * obj_mask.unsqueeze(-1)


class SceneEncoder(nn.Module):
    """Object tokens = [tag ; footprint descriptor ; pose ; rel-to-target ; is-target],
    projected to D_MODEL, then two Set-Attention layers (objects attend to each other).
    """

    def __init__(
        self, max_tags: int = MAX_TAGS_DEFAULT, dropout_p: float = DROPOUT
    ) -> None:
        super().__init__()
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        self.footprint = FootprintEncoder(dropout_p)
        self.pose_proj = nn.Linear(3, D_POSE)
        self.rel_proj = nn.Linear(D_REL, D_REL)
        in_dim = D_TAG + D_DESCRIPTOR + D_POSE + D_REL + 1
        self.proj = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.LayerNorm(D_MODEL))
        self.sab1 = SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p)
        self.sab2 = SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p)

    def forward(self, batch: SpectreV2Batch) -> Tensor:
        tag = self.tag_emb(batch.obj_tags)  # (B, N, D_TAG)
        desc = self.footprint(batch.obj_boundary, batch.obj_mask)  # (B, N, D_DESC)
        pose = self.pose_proj(batch.obj_pose)
        rel = self.rel_proj(batch.obj_rel)
        tgt = batch.obj_is_target.unsqueeze(-1)
        tok = self.proj(torch.cat([tag, desc, pose, rel, tgt], dim=-1))
        tok = self.sab1(tok, batch.obj_mask)
        tok = self.sab2(tok, batch.obj_mask)
        return tok * batch.obj_mask.unsqueeze(-1)  # (B, N, D_MODEL)


class CandidateEncoder(nn.Module):
    """Per-candidate embedding from its operator program (schema + position + tag
    args)."""

    def __init__(
        self, n_ops: int, max_tags: int, max_arity: int, dropout_p: float = DROPOUT
    ) -> None:
        super().__init__()
        self.op_emb = nn.Embedding(n_ops + 1, D_MODEL, padding_idx=0)
        self.pos_emb = nn.Embedding(64, D_MODEL)
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        self.arg_proj = nn.Linear(max_arity * D_TAG, D_MODEL)
        self.step_ln = nn.LayerNorm(D_MODEL)
        self.pool = PoolingByMultiheadAttention(
            dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p
        )
        self.max_arity = max_arity

    def forward(self, batch: SpectreV2Batch) -> Tensor:
        b, k, ell = batch.cand_op_ids.shape
        op = self.op_emb(batch.cand_op_ids)  # (B, K, L, D)
        pos = self.pos_emb(batch.cand_pos.clamp(max=63))
        args = self.tag_emb(batch.cand_arg_tags)  # (B, K, L, A, D_TAG)
        args = args.reshape(b, k, ell, self.max_arity * D_TAG)
        step = self.step_ln(op + pos + self.arg_proj(args))  # (B, K, L, D)
        step = step.reshape(b * k, ell, D_MODEL)
        smask = batch.cand_step_mask.reshape(b * k, ell)
        emb = self.pool(step, smask).reshape(b, k, D_MODEL)  # (B, K, D)
        return emb * batch.pool_mask.unsqueeze(-1)


class FactEncoder(nn.Module):
    """Typed-fact token = [fact-type emb ; tier emb ; mean-pooled argument-tag emb] → D_MODEL.

    The fact carries **identity** through its argument tags (the same episode-local tags as
    the scene/candidate tokens), so scrambling those tags changes the token — the property
    the live scramble gauge exploits. Empty-arg facts contribute only type/tier.
    """

    def __init__(self, max_tags: int, dropout_p: float = DROPOUT) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(N_FACT_TYPES + 1, D_MODEL, padding_idx=0)
        self.tier_emb = nn.Embedding(3, D_FACT_TIER, padding_idx=0)
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL + D_FACT_TIER + D_TAG, D_MODEL),
            nn.Dropout(dropout_p),
            nn.LayerNorm(D_MODEL),
        )

    def forward(
        self, type_ids: Tensor, tier_ids: Tensor, arg_tags: Tensor, fact_mask: Tensor
    ) -> Tensor:
        # type_ids (B,F); tier_ids (B,F); arg_tags (B,F,A); fact_mask (B,F)
        typ = self.type_emb(type_ids)  # (B, F, D)
        tier = self.tier_emb(tier_ids)  # (B, F, D_FACT_TIER)
        arg = self.tag_emb(arg_tags)  # (B, F, A, D_TAG)
        arg_present = (arg_tags != PAD_TAG).float().unsqueeze(-1)  # (B, F, A, 1)
        denom = arg_present.sum(dim=2).clamp(min=1.0)  # (B, F, 1)
        arg_pool = (arg * arg_present).sum(dim=2) / denom  # (B, F, D_TAG)
        tok = self.proj(torch.cat([typ, tier, arg_pool], dim=-1))  # (B, F, D)
        return tok * fact_mask.unsqueeze(-1)


class CrossAttentionScorer(nn.Module):
    """Each candidate (query) cross-attends over scene + global (+ fact) memory; the
    attended vector is concatenated with the candidate embedding and any overlap
    features → one logit."""

    def __init__(
        self,
        n_overlap_feats: int = 0,
        n_prior_feats: int = 0,
        dropout_p: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=dropout_p, batch_first=True
        )
        self.glob_proj = nn.Linear(D_GLOBAL_IN, D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(2 * D_MODEL + n_overlap_feats + n_prior_feats, FFN_DIM),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(FFN_DIM, 1),
        )
        self.n_overlap_feats = n_overlap_feats
        self.n_prior_feats = n_prior_feats
        if n_prior_feats:
            # Additive default-order residual: the geometry head is init-to-zero, and the
            # prior gate is init to trust the default-order column, so an untrained scorer
            # ranks ≈ default-order and the head only learns the correction (init-toward-prior).
            self.prior_gate = nn.Linear(n_prior_feats, 1)
            head_out = cast(nn.Linear, self.head[-1])
            nn.init.zeros_(head_out.weight)
            nn.init.zeros_(head_out.bias)
            nn.init.zeros_(self.prior_gate.bias)
            with torch.no_grad():
                self.prior_gate.weight.zero_()
                self.prior_gate.weight[0, 0] = 3.0  # −index (default order)

    def forward(
        self,
        cand_emb: Tensor,  # (B, K, D)
        scene_tok: Tensor,  # (B, N, D)
        obj_mask: Tensor,  # (B, N)
        glob_feats: Tensor,  # (B, D_GLOBAL_IN)
        overlap: Tensor | None = None,  # (B, K, n_overlap_feats)
        fact_tok: Tensor | None = None,  # (B, F, D)
        fact_mask: Tensor | None = None,  # (B, F)
        prior: Tensor | None = None,  # (B, K, n_prior_feats)
    ) -> Tensor:
        b, k, _ = cand_emb.shape
        glob = self.glob_proj(glob_feats).unsqueeze(1)  # (B, 1, D)
        mems = [scene_tok, glob]
        pads = [
            ~obj_mask,
            torch.zeros(b, 1, dtype=torch.bool, device=obj_mask.device),
        ]
        if fact_tok is not None and fact_tok.shape[1] > 0:
            mems.append(fact_tok)
            assert fact_mask is not None
            pads.append(~fact_mask)
        memory = torch.cat(mems, dim=1)  # (B, N+1(+F), D)
        key_pad = torch.cat(pads, dim=1)  # True = pad-to-ignore
        attended, _ = self.attn(cand_emb, memory, memory, key_padding_mask=key_pad)
        parts = [cand_emb, attended]
        if self.n_overlap_feats:
            parts.append(
                overlap
                if overlap is not None
                else cand_emb.new_zeros(b, k, self.n_overlap_feats)
            )
        pr = (
            prior if prior is not None else cand_emb.new_zeros(b, k, self.n_prior_feats)
        )
        if self.n_prior_feats:
            parts.append(pr)
        logit = self.head(torch.cat(parts, dim=-1)).squeeze(-1)  # (B, K)
        if self.n_prior_feats:
            logit = logit + self.prior_gate(pr).squeeze(-1)  # default-order anchor
        return logit


class AuxHead(nn.Module):
    """Per scene token → (necessary, relevant) logits (proposal §8)."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, scene_tok: Tensor) -> Tensor:
        return self.head(scene_tok)  # (B, N, 2)


class SpectreV2Model(nn.Module):
    """The v2.2-static geometry-aware ranker."""

    def __init__(
        self,
        n_ops: int,
        max_arity: int,
        max_tags: int = MAX_TAGS_DEFAULT,
        n_overlap_feats: int = 0,
        n_prior_feats: int = 0,
        dropout_p: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.scene = SceneEncoder(max_tags, dropout_p)
        self.cands = CandidateEncoder(n_ops, max_tags, max_arity, dropout_p)
        self.facts = FactEncoder(max_tags, dropout_p)
        self.scorer = CrossAttentionScorer(n_overlap_feats, n_prior_feats, dropout_p)
        self.aux = AuxHead()
        self.n_prior_feats = n_prior_feats
        self.n_overlap_feats = n_overlap_feats

    def forward(self, batch: SpectreV2Batch, overlap: Tensor | None = None):
        scene_tok = self.scene(batch)  # (B, N, D)
        cand_emb = self.cands(batch)  # (B, K, D)
        fact_tok = None
        if batch.fact_type_ids is not None and batch.fact_type_ids.shape[1] > 0:
            fact_tok = self.facts(
                batch.fact_type_ids,
                batch.fact_tier_ids,
                batch.fact_arg_tags,
                batch.fact_mask,
            )  # (B, F, D)
        prior = batch.cand_prior if self.n_prior_feats else None
        if overlap is None and self.n_overlap_feats:
            overlap = batch.cand_overlap
        logits = self.scorer(
            cand_emb,
            scene_tok,
            batch.obj_mask,
            batch.glob_feats,
            overlap,
            fact_tok,
            batch.fact_mask,
            prior,
        )  # (B, K)
        avail = batch.avail_mask if batch.avail_mask is not None else batch.pool_mask
        logits = logits.masked_fill(~avail, float("-inf"))
        aux = self.aux(scene_tok)  # (B, N, 2)
        return logits, aux
