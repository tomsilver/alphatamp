"""SPECTRE geometry/evidence encoders (proposal §7).

The object-centric geometric encoders the SPECTRE ranker (``model.py``) is built from:
the static (t=0) architecture conditions on object-centric geometry + episode-local tags
instead of anonymous local ids. Token families (all width ``D_MODEL=64``, built on the
shared ``SetAttentionBlock``/``PMA`` primitives from ``layers.py``):

- **scene tokens** — per object:
  ``[tag ; footprint descriptor ; pose ; relation-to-target]``.
  The footprint descriptor is a *point-set* encoding of the boundary ring (not a radial
  profile — concave-safe). A couple of Set-Attention layers let objects attend to each
  other (the relational join).
- **candidate tokens** — a skeleton is a *program over the scene*: per operator,
  its schema embedding + position + argument slots holding the objects' **tags**.
  Pooled to one ``e(s)`` per candidate.
- **global token** — container/buffer geometry + pool statistics.
- **fact tokens / overlap features** — empty at static (t=0); wired for the Step-11
  typed-evidence pathway. The scorer already accepts them so that step is additive.

**Scorer** — per-candidate cross-attention (candidate query over scene + global memory),
concatenated with computed overlap features → one logit; linear in pool size.
**Aux head** — per scene token → ``necessary``/``relevant`` logits (proposal §8).

The forward returns ``(B, K)`` logits with the same contract as v1's ``Scorer`` so the
rollout / PL-loss machinery is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import torch
from torch import Tensor, nn

from alphatamp.approaches.spectre.facts import N_FACT_TYPES
from alphatamp.approaches.spectre.layers import (
    D_MODEL,
    FFN_DIM,
    N_HEADS,
    MultiSeedPMA,
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
# init/goal atom-profile embedding width (doc spectre_atom_input_guide.md); matches
# D_PRED/D_DELTA in model.py. Used only by AtomProfileEncoder, which projects to D_MODEL.
D_ATOM = 32
# Scene relation: the three *anchor-free* per-object scalars ``[area, sinθ, cosθ]``. The
# earlier target-anchored offsets (dx, dy, dist), the area ratio to a single target, and
# the privileged ``concave`` flag were all cut, because they either presuppose one
# distinguished target (meaningless with N goal objects) or are privileged geometry a
# non-privileged pipeline could not read. ``SceneEncoder`` takes the width per instance;
# a checkpoint is bound to the width it was trained on. See docs/decisions 2026-08-08.
D_REL = 3
MAX_TAGS_DEFAULT = 32
DROPOUT = 0.1

# typed-evidence (Step 11) dims
MAX_FACT_ARGS = 12  # cap on a fact's argument list (mean-pooled); larger sets truncate
D_FACT_TIER = 8

# a-priori per-candidate prior features: [−index/K, −len/max_len] (default-order /
# short-first) — domain-agnostic planner signals available in any TAMP problem.
# Column 0 is the additive default-order residual anchor the geometry head only has to
# correct.
N_PRIOR = 2

# structural evidence features relating each candidate's action-set to the OBSERVED
# failed sets (Step 11 fix): [subset⊆blocked (sound proof-demotion — provably
# also-blocked), max-Jaccard-with-failed (hint)]. Domain-agnostic set relations. The
# unsound "blocked⊊subset ⇒ prefer longer" cue is deliberately excluded — it helps s3
# but misleads easy strata.
N_OVERLAP = 2


@dataclass
class SpectreBatch:
    """Padded tensors for one batch of episodes (0 = pad; see ``dataset``).

    The leading fields are the static scene/candidate/label tensors; the trailing
    ``rec_*`` fields carry the failure-record tokens and default to ``None``, so a batch
    built without them is exactly the static batch. Record tags are **role-separated**:
    ``rec_arg_tags`` holds the objects the failing query was *about* and
    ``rec_culprit_tags`` the objects observed to block it -- pooling both into one slot
    would tell the net "these objects are associated with this failure" without saying
    which was the target and which the obstacle.
    """

    # scene (objects)
    obj_tags: Tensor  # (B, N) long — episode-local tag ids (0 = pad)
    obj_boundary: Tensor  # (B, N, P, 2) float — resampled boundary ring, item frame
    obj_pose: Tensor  # (B, N, 3) float — (x, y, theta), normalized
    obj_rel: Tensor  # (B, N, d_rel) float — per-object anchor-free scene scalars (3)
    obj_is_goal: Tensor  # (B, N) float — 1 for an object named by the goal atoms
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
    # short-first). Known before any refinement — the scorer treats geometry as a
    # residual correction on this prior (init-toward-prior); ``None`` disables it.
    cand_prior: Optional[Tensor] = None  # (B, K, N_PRIOR) float
    # structural evidence features per candidate vs the observed failed sets
    # (Step 11 fix); 0 when no facts / static path. Lets the ranker use proofs by
    # set-containment.
    cand_overlap: Optional[Tensor] = None  # (B, K, N_OVERLAP) float
    # failure-record tokens (all None on the static path). See MAX_RECORD_* in model.py.
    rec_schema_ids: Optional[Tensor] = None  # (B, R) long — 0 = pad
    rec_arg_tags: Optional[Tensor] = None  # (B, R, MAX_RECORD_ARGS) long
    rec_culprit_tags: Optional[Tensor] = None  # (B, R, MAX_RECORD_CULPRITS) long
    rec_scalars: Optional[Tensor] = None  # (B, R, N_RECORD_SCALARS) float
    rec_mask: Optional[Tensor] = None  # (B, R) bool — real record
    # `s_j - s_0` per record. Role axis is [added, deleted] — kept apart for the same
    # reason arg-tags and culprit-tags are: "the prefix put o1 on the buffer" and "the
    # prefix took o1 out of the drawer" are different claims about o1.
    rec_delta_pred_ids: Optional[Tensor] = None  # (B, R, 2, MAX_DELTA_ATOMS) long
    rec_delta_arg_tags: Optional[Tensor] = None  # (B, R, 2, MAX_DELTA_ATOMS, A) long
    # PointSetEncoder inputs (doc pointset_encoder_upgrade.md); None on the config-off
    # path (the FootprintEncoder v1 path ignores them). Trailing-nullable like rec_*.
    point_feats: Optional[Tensor] = None  # (B, N, P, C_pt) float — per-point features
    knn_idx: Optional[Tensor] = None  # (B, N, P, k) long — Euclidean-kNN indices
    # init/goal atom profiles (doc spectre_atom_input_guide.md); None on the config-off
    # path (atom_mode="off"). Consumed by AtomProfileEncoder: `pred` = vocab id +1
    # (0 = pad), `arg_tags` = object tags in the scene tag namespace (0 = PAD_TAG).
    init_atom_pred: Optional[Tensor] = None  # (B, A_i) long
    init_atom_arg_tags: Optional[Tensor] = None  # (B, A_i, M) long
    goal_atom_pred: Optional[Tensor] = None  # (B, A_g) long
    goal_atom_arg_tags: Optional[Tensor] = None  # (B, A_g, M) long
    # Rung-1 evidence-step stream (docs/failed_records_fix.md F-A); None unless
    # record_mode="steps". Steps are in the candidate namespace (op id + position + arg
    # tags) so the shared CandidateEncoder embeds them; the enrichment channels
    # (status/attempt/culprit/scalars) are added on top. See RecordStepEncoder.
    rec_step_op_ids: Optional[Tensor] = None  # (B, S) long — 0 = pad
    rec_step_arg_tags: Optional[Tensor] = None  # (B, S, A) long
    rec_step_pos: Optional[Tensor] = None  # (B, S) long — step index in its plan
    rec_step_status: Optional[Tensor] = None  # (B, S) long — 1 failed / 2 established
    rec_step_attempt: Optional[Tensor] = None  # (B, S) long — within-context attempt id
    rec_step_culprit_tags: Optional[Tensor] = None  # (B, S, MAX_RECORD_CULPRITS) long
    rec_step_culprit_counts: Optional[Tensor] = (
        None  # (B, S, MAX_RECORD_CULPRITS) float
    )
    rec_step_scalars: Optional[Tensor] = None  # (B, S, N_STEP_SCALARS) float
    rec_step_mask: Optional[Tensor] = None  # (B, S) bool — real step

    def to(self, device) -> "SpectreBatch":
        return SpectreBatch(
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

    def __init__(self, dropout_p: float = DROPOUT, point_dim: int = 2) -> None:
        super().__init__()
        # ``point_dim`` is 2 for the boundary-ring (DD2D/SB2D) path and 3 for a Restock3D
        # point cloud. Only the input Linear's in-features change; the symmetric
        # point-set pool below is dimension-agnostic. Persisted via ``SpectreConfig`` so
        # a checkpoint is bound to its trained width. See docs/decisions 2026-08-18.
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, D_POINT),
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


def point_feat_dim(use_pca_feats: bool, point_dim: int) -> int:
    """Per-point feature width ``C_pt`` the tensorizer emits and the encoder consumes.

    With PCA features: coordinates + oriented normal + (2D signed curvature / 3D pad) +
    flatness = ``2*point_dim + 2`` (6 in 2D, 8 in 3D). Without: raw coordinates only
    (= ``point_dim``). A pure function of the two persisted config fields, so the
    tensorizer and the model always agree on the lift-MLP input width.
    """
    return (2 * point_dim + 2) if use_pca_feats else point_dim


class EdgeConv(nn.Module):
    """One DGCNN EdgeConv layer over a fixed, precomputed kNN graph (doc §3).

    ``msg = mlp([h_i ; h_j - h_i])`` max-aggregated over the k neighbors, then a
    **zero-initialized** residual projection so the branch is near-identity at init (the
    function class nests the no-EdgeConv model up to the retained residual LayerNorm).
    The graph is the tensorizer's coordinate-space kNN, not recomputed in feature space
    -- one layer, deterministic, sufficient at P=32.
    """

    def __init__(self, dim: int = D_MODEL, dropout_p: float = DROPOUT) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
        )
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.ln = nn.LayerNorm(dim)

    def forward(self, h: Tensor, knn_idx: Tensor, pmask: Tensor) -> Tensor:
        # h (Bn, P, D); knn_idx (Bn, P, k) long; pmask (Bn, P) bool (True = real point).
        bn, _, _ = h.shape
        k = knn_idx.shape[-1]
        batch_ix = torch.arange(bn, device=h.device)[:, None, None]  # (Bn, 1, 1)
        h_j = h[batch_ix, knn_idx]  # (Bn, P, k, D) — advanced-index gather
        h_i = h.unsqueeze(2).expand(-1, -1, k, -1)  # (Bn, P, k, D)
        msg = self.mlp(torch.cat([h_i, h_j - h_i], dim=-1))  # (Bn, P, k, D)
        nbr_valid = pmask[batch_ix, knn_idx]  # (Bn, P, k)
        msg = msg.masked_fill(~nbr_valid.unsqueeze(-1), float("-inf"))
        agg = msg.max(dim=2).values  # (Bn, P, D) — DGCNN max-aggregation
        # A point with no valid neighbor would max to -inf; zero it (never fires at P=32,
        # lives for 3D partial clouds, doc §2.6/§3 note 3).
        agg = torch.where(
            nbr_valid.any(dim=2, keepdim=True), agg, torch.zeros_like(agg)
        )
        return self.ln(h + self.out_proj(agg))  # residual (out_proj zero-init)


class PointSetEncoder(nn.Module):
    """Upgraded per-object descriptor (doc pointset_encoder_upgrade.md).

    ``lift(C_pt→32→64) → [EdgeConv] → [SAB] → MultiSeedPMA(seeds) →
    Linear(64·seeds→32)``. Selected by :class:`SceneEncoder` only when a switch is on;
    config-off keeps the v1 :class:`FootprintEncoder`. Output width is
    ``D_DESCRIPTOR=32`` so the scene-token slot is unchanged. Dimension-generic:
    ``c_pt`` alone carries 2D vs 3D.
    """

    def __init__(
        self,
        c_pt: int,
        use_edgeconv: bool = False,
        use_point_sab: bool = False,
        pma_seeds: int = 1,
        dropout_p: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.pma_seeds = pma_seeds
        self.lift = nn.Sequential(
            nn.Linear(c_pt, 32),
            nn.GELU(),
            nn.Linear(32, D_MODEL),
        )
        self.edge = EdgeConv(D_MODEL, dropout_p) if use_edgeconv else None
        self.sab = (
            SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p)
            if use_point_sab
            else None
        )
        self.pool = MultiSeedPMA(
            dim=D_MODEL, n_heads=N_HEADS, n_seeds=pma_seeds, dropout_p=dropout_p
        )
        self.out = nn.Linear(D_MODEL * pma_seeds, D_DESCRIPTOR)

    def forward(
        self, point_feats: Tensor, knn_idx: Optional[Tensor], obj_mask: Tensor
    ) -> Tensor:
        # point_feats (B, N, P, C_pt); knn_idx (B, N, P, k) | None; obj_mask (B, N).
        b, n, p, _ = point_feats.shape
        h = self.lift(point_feats).reshape(b * n, p, D_MODEL)  # (B*N, P, D)
        # Per-point mask: all-real today (P fixed); threaded so a real 3D partial-cloud
        # mask is a one-line change (doc §2.6).
        pmask = torch.ones(b * n, p, dtype=torch.bool, device=point_feats.device)
        if self.edge is not None:
            assert knn_idx is not None, "use_edgeconv=True requires knn_idx"
            h = self.edge(h, knn_idx.reshape(b * n, p, -1), pmask)
        if self.sab is not None:
            h = self.sab(h, pmask)
        pooled = self.pool(h, pmask).reshape(b * n, -1)  # (B*N, D_MODEL*seeds)
        desc = self.out(pooled).reshape(b, n, D_DESCRIPTOR)  # (B, N, 32)
        return desc * obj_mask.unsqueeze(-1)


class SceneEncoder(nn.Module):
    """Object tokens = [tag ; footprint descriptor ; pose ; rel-to-target ; is-target],
    projected to D_MODEL, then two Set-Attention layers (objects attend to each other).
    """

    def __init__(
        self,
        max_tags: int = MAX_TAGS_DEFAULT,
        dropout_p: float = DROPOUT,
        d_rel: int = D_REL,
        point_dim: int = 2,
        pose_dim: int = 3,
        use_pca_feats: bool = False,
        use_edgeconv: bool = False,
        use_point_sab: bool = False,
        pma_seeds: int = 1,
    ) -> None:
        super().__init__()
        # ``d_rel`` is the width of ``obj_rel``: the anchor-free ``[area, sinθ, cosθ]``
        # triple (3). ``point_dim``/``pose_dim`` are 2/3 for the 2D footprint path and
        # 3/4 for a Restock3D point cloud + (x, y, z, yaw) pose. All carried per instance
        # so the widths are bound to the checkpoint they were trained on.
        self.d_rel = d_rel
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        # Descriptor module: v1 ``FootprintEncoder`` when the PointSet upgrade is off
        # (exact byte-identity / D-8 compat), else ``PointSetEncoder``. Exactly one is
        # built -- the conditional-submodule pattern (cf. SpectreModel ``records``) so
        # config-off adds no state-dict keys and old checkpoints load ``strict``.
        self.use_pointset = (
            use_pca_feats or use_edgeconv or use_point_sab or pma_seeds > 1
        )
        if self.use_pointset:
            c_pt = point_feat_dim(use_pca_feats, point_dim)
            self.pointset = PointSetEncoder(
                c_pt, use_edgeconv, use_point_sab, pma_seeds, dropout_p
            )
        else:
            self.footprint = FootprintEncoder(dropout_p, point_dim=point_dim)
        self.pose_proj = nn.Linear(pose_dim, D_POSE)
        self.rel_proj = nn.Linear(d_rel, d_rel)
        in_dim = D_TAG + D_DESCRIPTOR + D_POSE + d_rel + 1
        self.proj = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.LayerNorm(D_MODEL))
        self.sab1 = SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p)
        self.sab2 = SetAttentionBlock(dim=D_MODEL, n_heads=N_HEADS, dropout_p=dropout_p)

    def forward(
        self, batch: SpectreBatch, atom_obj_add: Optional[Tensor] = None
    ) -> Tensor:
        tag = self.tag_emb(batch.obj_tags)  # (B, N, D_TAG)
        if self.use_pointset:
            desc = self.pointset(
                batch.point_feats, batch.knn_idx, batch.obj_mask
            )  # (B, N, D_DESC)
        else:
            desc = self.footprint(batch.obj_boundary, batch.obj_mask)  # (B, N, D_DESC)
        pose = self.pose_proj(batch.obj_pose)
        rel = self.rel_proj(batch.obj_rel)
        tgt = batch.obj_is_goal.unsqueeze(-1)
        # `self.proj` is Linear -> LayerNorm; the (zero-init) atom-profile add goes in
        # between so object-object attention conditions on atoms. `None` ⇒ the exact
        # `self.proj(cat)` expression, byte-identical to the pre-atom-input path.
        h = self.proj[0](torch.cat([tag, desc, pose, rel, tgt], dim=-1))
        if atom_obj_add is not None:
            h = h + atom_obj_add
        tok = self.proj[1](h)
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

    def encode_steps(self, op_ids: Tensor, pos: Tensor, arg_tags: Tensor) -> Tensor:
        """Per-step token embeddings ``[op + position + projected args]``, shape
        ``(..., L, D)``.

        Factored out of :meth:`forward` (byte-identical to the old inline computation) so
        the **same weights** can encode a failed skeleton's evidence steps for rung-1
        enrichment — a failed ``place_short(b)`` and the current candidate's
        ``place_short(b)`` then become identical vectors by construction. Leading dims are
        arbitrary: ``(B, K, L)`` for candidates, ``(B, S)`` for a flat evidence-step
        stream.
        """
        op = self.op_emb(op_ids)
        pos_e = self.pos_emb(pos.clamp(max=63))
        args = self.tag_emb(arg_tags).reshape(
            *arg_tags.shape[:-1], self.max_arity * D_TAG
        )
        return self.step_ln(op + pos_e + self.arg_proj(args))

    def pool_steps(self, step: Tensor, step_mask: Tensor, pool_mask: Tensor) -> Tensor:
        """PMA-pool per-candidate step tokens ``(B, K, L, D)`` to ``(B, K, D)``."""
        b, k, ell = step.shape[0], step.shape[1], step.shape[2]
        emb = self.pool(
            step.reshape(b * k, ell, D_MODEL), step_mask.reshape(b * k, ell)
        ).reshape(b, k, D_MODEL)
        return emb * pool_mask.unsqueeze(-1)

    def forward(self, batch: SpectreBatch) -> Tensor:
        step = self.encode_steps(
            batch.cand_op_ids, batch.cand_pos, batch.cand_arg_tags
        )  # (B, K, L, D)
        return self.pool_steps(step, batch.cand_step_mask, batch.pool_mask)


class FactEncoder(nn.Module):
    """Typed-fact token = [fact-type emb ; tier emb ; mean-pooled argument-tag emb] →
    D_MODEL.

    The fact carries **identity** through its argument tags (the same episode-local
    tags as the scene/candidate tokens), so scrambling those tags changes the token —
    the property the live scramble gauge exploits. Empty-arg facts contribute only
    type/tier.
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


class AtomProfileEncoder(nn.Module):
    """Object-centric profiles of the init-state / goal atoms (Rung A of the atom-input
    guide, ``docs/spectre_atom_input_guide.md``).

    For each object *o* it sums ``predEmb[p] + slotEmb[k]`` over every atom that names
    *o* at argument slot *k*, forming a fixed-width bag-of-``(predicate, slot)`` summary
    -- separately for the initial abstract state and the goal (kept apart: "true now" vs
    "wanted"). Sum pooling keeps atom *count* as information; the slot embedding keeps
    ``On(a, b)`` distinct from ``On(b, a)``. 0-ary atoms (no object argument) pool into
    a separate global term. Both injections leave the module through **zero-
    initialized** Linears, so a freshly-built ``atom_mode="profiles"`` model is
    functionally identical to ``atom_mode="off"`` at step 0 (the state-delta branch
    discipline, one module over); ``SceneEncoder`` and the scorer gain no parameters of
    their own.
    """

    def __init__(
        self,
        n_predicates: int,
        max_pred_arity: int,
        max_tags: int,
        use_init: bool = True,
        use_goal: bool = True,
    ) -> None:
        # No dropout: the profiles feed two zero-init Linears (opt-in-by-gradient), so a
        # dropout layer would add nothing at init and only noise a tiny, additive signal.
        super().__init__()
        if n_predicates <= 0:
            raise ValueError(
                "atom_mode='profiles' needs n_predicates from the vocab; a 1-row "
                "embedding table would train silently and mean nothing"
            )
        self.max_tags = max_tags
        self.arity = max(max_pred_arity, 1)
        self.use_init = use_init
        self.use_goal = use_goal
        self.pred_emb = nn.Embedding(n_predicates + 1, D_ATOM, padding_idx=0)
        self.slot_emb = nn.Embedding(self.arity, D_ATOM)
        # INIT and GOAL profiles are concatenated (2*D_ATOM) before projection, so
        # use_init/use_goal only zero a half and never change the state dict.
        self.obj_proj = nn.Linear(2 * D_ATOM, D_MODEL)
        self.glob_proj_atom = nn.Linear(2 * D_ATOM, D_MODEL)
        for proj in (self.obj_proj, self.glob_proj_atom):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _profiles(
        self, pred: Tensor, arg_tags: Tensor, obj_tags: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``(obj (B, N, D_ATOM), glob (B, D_ATOM))`` for one provenance.

        Per-object profiles are built by scatter-summing each atom's ``predEmb +
        slotEmb`` into a per-tag buffer, then gathering by each object's tag (tags are
        distinct per object). PAD slots contribute exactly zero, so ``buf[:, 0]`` (the
        pad tag) and padded objects stay zero. 0-ary atoms (all-PAD arg row) skip the
        scatter and pool into the global term instead.
        """
        b, _ = obj_tags.shape
        m = arg_tags.shape[-1]
        present = pred.ne(0)  # (B, A) real atom
        slot_present = arg_tags.ne(PAD_TAG)  # (B, A, M) real object slot
        p = self.pred_emb(pred)  # (B, A, D)
        slot_w = self.slot_emb.weight[:m].view(1, 1, m, D_ATOM)
        contrib = (p.unsqueeze(2) + slot_w) * slot_present.unsqueeze(-1).to(p.dtype)
        buf = p.new_zeros(b, self.max_tags + 1, D_ATOM)
        idx = arg_tags.reshape(b, -1)  # (B, A*M) tag ids (0 = PAD)
        buf.scatter_add_(
            1,
            idx.unsqueeze(-1).expand(-1, -1, D_ATOM),
            contrib.reshape(b, -1, D_ATOM),
        )
        obj = buf.gather(1, obj_tags.unsqueeze(-1).expand(-1, -1, D_ATOM))  # (B, N, D)
        is_global = (present & ~slot_present.any(dim=-1)).unsqueeze(-1).to(p.dtype)
        glob = (p * is_global).sum(dim=1)  # (B, D) — 0-ary atoms only
        return obj, glob

    def forward(self, batch: SpectreBatch) -> tuple[Tensor, Tensor]:
        obj_tags = batch.obj_tags  # (B, N) long
        b, n = obj_tags.shape
        dtype = self.pred_emb.weight.dtype
        zeros_obj = torch.zeros(b, n, D_ATOM, device=obj_tags.device, dtype=dtype)
        zeros_glob = torch.zeros(b, D_ATOM, device=obj_tags.device, dtype=dtype)

        def _one(
            pred: Optional[Tensor], arg: Optional[Tensor], use: bool
        ) -> tuple[Tensor, Tensor]:
            # Zero-substitute an absent/off provenance (like RecordEncoder's missing
            # delta), so a batch with atoms and one without encode identically.
            if not use or pred is None or arg is None:
                return zeros_obj, zeros_glob
            return self._profiles(pred, arg, obj_tags)

        obj_i, glob_i = _one(
            batch.init_atom_pred, batch.init_atom_arg_tags, self.use_init
        )
        obj_g, glob_g = _one(
            batch.goal_atom_pred, batch.goal_atom_arg_tags, self.use_goal
        )
        obj_add = self.obj_proj(torch.cat([obj_i, obj_g], dim=-1))
        obj_add = obj_add * batch.obj_mask.unsqueeze(-1)
        glob_add = self.glob_proj_atom(torch.cat([glob_i, glob_g], dim=-1))
        return obj_add, glob_add


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
            # Additive default-order residual: the geometry head is init-to-zero, and
            # the prior gate is init to trust the default-order column, so an untrained
            # scorer ranks ≈ default-order and the head only learns the correction
            # (init-toward-prior).
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
        glob_extra: Tensor | None = None,  # (B, D) — 0-ary atom profile (zero-init)
    ) -> Tensor:
        b, k, _ = cand_emb.shape
        # `glob_extra` carries the 0-ary init/goal atom profile (zero-init), so `None`
        # reduces this to the pre-atom-input `glob_proj(glob_feats)` exactly.
        glob = self.glob_proj(glob_feats)
        if glob_extra is not None:
            glob = glob + glob_extra
        glob = glob.unsqueeze(1)  # (B, 1, D)
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
