"""SPECTRE v3 ranker -- the same job as v2.2, re-derived from three ideas.

v3 keeps v2.2's contribution (a relational, tag-joined, object-centric geometric encoder
scored listwise) and removes the accretions around it: the data-dependent prior knob, the
five bespoke fact types, the inert packing certificate, the part-zeroed global token.
See ``docs/SPECTRE_v3_proposal.md``.

**The exact-absence invariant (D-8).** Every v3 feature is behind a flag on
:class:`V3Config`, and with every flag off this model *is* deployed v2.2 -- not
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

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from alphatamp.approaches.spectre.model import D_MODEL, FFN_DIM, N_HEADS
from alphatamp.approaches.spectre.model_v2 import (
    D_DESCRIPTOR,
    D_POSE,
    D_REL,
    D_TAG,
    MAX_TAGS_DEFAULT,
    AuxHead,
    CandidateEncoder,
    CrossAttentionScorer,
    FactEncoder,
    SceneEncoder,
    SpectreV2Batch,
)
from alphatamp.approaches.spectre.tags import PAD_TAG

DROPOUT = 0.1

# Record-token dims. `MAX_RECORD_ARGS` / `MAX_RECORD_CULPRITS` cap how many objects one
# record names in each role; DD2D queries are unary and a grasp is blocked by a handful of
# objects, so these are generous.
MAX_RECORD_ARGS = 4
MAX_RECORD_CULPRITS = 8
D_SCHEMA = 32
N_RECORD_SCALARS = 4  # [depth j/L, effort (log1p, scaled), exhausted, effort_is_total]


@dataclass
class SpectreV3Batch(SpectreV2Batch):
    """The v2.2 batch plus v3's failure-record tokens.

    Record fields are trailing and optional, so a batch built without them *is* a v2.2
    batch and the compat path is unaffected. They replace the five bespoke `fact_*`
    tensors, which stay present so the legacy encoder remains selectable (D-8).

    Tags are **role-separated**: `rec_arg_tags` holds the objects the failing query was
    *about*, `rec_culprit_tags` the objects observed to block it. v2.2 kept that
    distinction only implicitly, by giving `grasp-witness` its own fact type; pooling both
    roles into one slot would tell the net "these objects are associated with this
    failure" without saying which was the target and which the obstacle.
    """

    rec_schema_ids: Optional[Tensor] = None  # (B, R) long — 0 = pad
    rec_arg_tags: Optional[Tensor] = None  # (B, R, MAX_RECORD_ARGS) long
    rec_culprit_tags: Optional[Tensor] = None  # (B, R, MAX_RECORD_CULPRITS) long
    rec_scalars: Optional[Tensor] = None  # (B, R, N_RECORD_SCALARS) float
    rec_mask: Optional[Tensor] = None  # (B, R) bool — real record
    obj_evidence: Optional[Tensor] = None  # (B, N, N_OBJ_EVIDENCE) float

    def to(self, device) -> "SpectreV3Batch":
        return SpectreV3Batch(
            **{  # type: ignore[arg-type]
                k: (v.to(device) if v is not None else None)
                for k, v in self.__dict__.items()
            }
        )


class RecordEncoder(nn.Module):
    """One observed failure -> one token, with the object roles kept apart.

    Replaces `FactEncoder`'s hand-built type vocabulary with the domain's own operator
    schemas, and finally consumes the scalars v2.2 harvested and then dropped on the floor
    (`Fact.scalars` never reached the tensorizer). No tier embedding: only hint-tier
    evidence ever entered the network, so it was a constant column.
    """

    def __init__(
        self, n_schemas: int, max_tags: int, dropout_p: float = DROPOUT
    ) -> None:
        super().__init__()
        self.schema_emb = nn.Embedding(n_schemas + 1, D_SCHEMA, padding_idx=0)
        self.tag_emb = nn.Embedding(max_tags + 1, D_TAG, padding_idx=PAD_TAG)
        self.proj = nn.Sequential(
            nn.Linear(D_SCHEMA + 2 * D_TAG + N_RECORD_SCALARS, D_MODEL),
            nn.Dropout(dropout_p),
            nn.LayerNorm(D_MODEL),
        )

    @staticmethod
    def _pool(emb: Tensor, ids: Tensor) -> Tensor:
        """Masked mean over a role's tag slots; zeros when the role is empty."""
        present = (ids != PAD_TAG).float().unsqueeze(-1)
        return (emb * present).sum(dim=2) / present.sum(dim=2).clamp(min=1.0)

    def forward(
        self,
        schema_ids: Tensor,
        arg_tags: Tensor,
        culprit_tags: Tensor,
        scalars: Tensor,
        mask: Tensor,
    ) -> Tensor:
        parts = [
            self.schema_emb(schema_ids),
            self._pool(self.tag_emb(arg_tags), arg_tags),
            self._pool(self.tag_emb(culprit_tags), culprit_tags),
            scalars,
        ]
        return self.proj(torch.cat(parts, dim=-1)) * mask.unsqueeze(-1)


def sinusoidal_positions(pos: Tensor, dim: int) -> Tensor:
    """Standard transformer sinusoidal encoding evaluated at arbitrary integer positions.

    Returns ``(*pos.shape, dim)``. Unlike a learned table this is *defined* at every
    position, which is the whole point: the absolute ``nn.Embedding(64, D)`` it replaces
    has untrained rows beyond the longest plan seen in training, so a model trained on
    s0-s2 (plans of <= 5 operators) and deployed on s3 (7) would read randomly-initialized
    vectors at steps 5 and 6 -- and the length-generalization experiment would be
    measuring initialization noise rather than generalization.
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=pos.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    ang = pos.unsqueeze(-1).float() * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class CrossAttentionScorerV3(CrossAttentionScorer):
    """Scorer with a **separate attention channel for evidence**.

    v2.2 concatenates scene tokens, the global token and the evidence tokens into one
    memory and runs a single cross-attention over it. That is the architectural reason
    the record tokens end up inert, and it is a competition the evidence cannot win:

    - **One softmax must split its mass.** With ~10 scene tokens against up to 2045 record
      tokens, the geometry that actually determines feasibility is outnumbered ~200:1;
      with aggregation it is still ~3:1 and grows with |F|.
    - **Geometry is reliably useful, evidence is noisy.** The loss-minimizing policy for a
      *shared* attention budget is therefore to spend it on geometry and ignore evidence
      -- exactly what the ``suppress_records`` diagnostic measured (16.17 -> 16.40, i.e.
      the trained model had already learned to discard its own records).

    Two channels remove the competition: the candidate attends over ``[scene ; global]``
    and, independently, over the evidence memory, and the head sees both. Evidence can now
    be attended to *without* giving up geometry, so a useful record no longer has to
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
    ) -> Tensor:
        b, k, _ = cand_emb.shape
        glob = self.glob_proj(glob_feats).unsqueeze(1)
        memory = torch.cat([scene_tok, glob], dim=1)
        key_pad = torch.cat(
            [~obj_mask, torch.zeros(b, 1, dtype=torch.bool, device=obj_mask.device)],
            dim=1,
        )
        attended, _ = self.attn(cand_emb, memory, memory, key_padding_mask=key_pad)

        ev = cand_emb.new_zeros(b, k, D_MODEL)
        if fact_tok is not None and fact_tok.shape[1] > 0 and fact_mask is not None:
            # A batch row with no records would be an all-True key-padding mask, which
            # makes MultiheadAttention emit NaN rather than an empty result. Attend under
            # a mask that always leaves one key live, then zero those rows afterwards --
            # the same guard the v1 encoder uses.
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


N_OBJ_EVIDENCE = 5
"""Per-object evidence summary width; see :class:`SceneEncoderV3`."""


class SceneEncoderV3(SceneEncoder):
    """:class:`SceneEncoder` plus a per-object summary of the failures observed so far.

    **Why here, and not as more tokens.** Measured on the G6b checkpoint: deploying a
    records-trained model with its evidence memory emptied at every step moves it by 0.23
    FP (16.17 -> 16.40). The model had learned to *ignore* the per-failure tokens. What it
    does use is `cand_overlap` -- two compact scalars per candidate summarising the same
    failure set. So the failure is not "evidence is useless" but "free-floating tokens are
    the wrong shape for this architecture": the scorer's strength is the tag join between
    objects and candidate arguments, and a record token participates in that join only
    weakly, through pooled tag slots.

    This routes the same observations onto the objects they *name*, where the tag join
    already lives. Four scalars per object, all in [0, 1], all zero when no failure has
    been observed yet:

    ``[frac of failed candidates that manipulate o,
       frac of hint records naming o as an argument,
       frac of hint records naming o as a culprit,
       mean normalized depth of the records naming o]``

    Domain-agnostic by construction -- set membership over record fields, no geometry and
    no per-environment predicate (C1). Proof-tier records stay excluded exactly as they are
    from the token path, so nothing here re-imports the "blocked sets are large, prefer
    longer" correlate that L4 warns about.
    """

    def __init__(
        self, max_tags: int = MAX_TAGS_DEFAULT, dropout_p: float = DROPOUT
    ) -> None:
        super().__init__(max_tags, dropout_p)
        in_dim = D_TAG + D_DESCRIPTOR + D_POSE + D_REL + 1 + N_OBJ_EVIDENCE
        self.proj = nn.Sequential(nn.Linear(in_dim, D_MODEL), nn.LayerNorm(D_MODEL))

    def forward(self, batch: SpectreV2Batch) -> Tensor:
        tag = self.tag_emb(batch.obj_tags)
        desc = self.footprint(batch.obj_boundary, batch.obj_mask)
        pose = self.pose_proj(batch.obj_pose)
        rel = self.rel_proj(batch.obj_rel)
        tgt = batch.obj_is_target.unsqueeze(-1)
        ev = getattr(batch, "obj_evidence", None)
        if ev is None:
            ev = torch.zeros(
                *batch.obj_tags.shape,
                N_OBJ_EVIDENCE,
                device=tag.device,
                dtype=tag.dtype,
            )
        tok = self.proj(torch.cat([tag, desc, pose, rel, tgt, ev], dim=-1))
        tok = self.sab1(tok, batch.obj_mask)
        tok = self.sab2(tok, batch.obj_mask)
        return tok * batch.obj_mask.unsqueeze(-1)


class CandidateEncoderV3(CandidateEncoder):
    """:class:`CandidateEncoder` with the learned absolute position table removed.

    Subclassed rather than edited in place because v2 modules are frozen (D-7). The
    ``pos_emb`` submodule is *deleted*, not merely bypassed, so it leaves the state dict
    -- which is exactly why enabling this retires the D-8 equivalence oracle: a v2.2
    checkpoint can no longer load ``strict=True`` into this model. That is planned (G9 is
    the last architectural change), not accidental.
    """

    def __init__(
        self, n_ops: int, max_tags: int, max_arity: int, dropout_p: float = DROPOUT
    ) -> None:
        super().__init__(n_ops, max_tags, max_arity, dropout_p)
        del self.pos_emb

    def forward(self, batch: SpectreV2Batch) -> Tensor:
        b, k, ell = batch.cand_op_ids.shape
        op = self.op_emb(batch.cand_op_ids)
        pos = sinusoidal_positions(batch.cand_pos, D_MODEL)
        args = self.tag_emb(batch.cand_arg_tags)
        args = args.reshape(b, k, ell, self.max_arity * D_TAG)
        step = self.step_ln(op + pos + self.arg_proj(args))
        step = step.reshape(b * k, ell, D_MODEL)
        smask = batch.cand_step_mask.reshape(b * k, ell)
        emb = self.pool(step, smask).reshape(b, k, D_MODEL)
        return emb * batch.pool_mask.unsqueeze(-1)


@dataclass(frozen=True)
class V3Config:
    """Architecture switches. Every v3 feature defaults **off**, so the default config
    reproduces deployed v2.2 exactly (D-8).

    ``n_prior_feats`` is retained only so a pre-v3 checkpoint that *was* trained with the
    short-first prior still loads for comparison. The deployed dd2d_v3 model has it off,
    and v3 does not reintroduce it: the prior was a per-dataset hand switch that
    diverged training on the easier collection (``decisions.md`` 2026-07-25). Note the
    v2 scorer couples ``n_prior_feats > 0`` to a zero-init of the head's output layer, so
    prior-on and prior-off differ in initialization as well as in features -- a confound
    to remember when reading any historical prior on/off delta.
    """

    n_overlap_feats: int = 0
    n_prior_feats: int = 0
    max_tags: int = MAX_TAGS_DEFAULT
    dropout_p: float = DROPOUT
    # --- v3 feature switches (added by later gates; all no-ops here) ---
    use_records: bool = False  # G6: role-separated FailureRecord tokens
    use_necessity: bool = False  # G8: necessity head + its candidate features
    # G9: sinusoidal step positions instead of the learned absolute table. Turning this
    # on RETIRES the D-8 equivalence oracle (pos_emb leaves the state dict), so it is the
    # last architectural change by design.
    sinusoidal_pos: bool = False
    # Per-object evidence summary on the scene tokens (see SceneEncoderV3). Changes the
    # scene projection's input width, so it also retires the D-8 oracle.
    use_obj_evidence: bool = False
    # Give evidence its own cross-attention channel instead of making it compete with
    # the scene inside one softmax. See CrossAttentionScorerV3.
    evidence_attn: bool = False

    @classmethod
    def from_v2_checkpoint_cfg(cls, cfg: dict) -> "V3Config":
        """Build the compat config that matches a stored ``train_v2`` checkpoint.

        ``train_v2`` records ``use_prior`` / ``use_overlap`` as booleans and the scorer
        sizes itself from the corresponding widths, so a checkpoint cannot be loaded
        without consulting them (``strict=True`` would fail on the head shape).
        """
        return cls(
            n_overlap_feats=2 if cfg.get("use_overlap") else 0,
            n_prior_feats=2 if cfg.get("use_prior") else 0,
            max_tags=int(cfg.get("max_tags", MAX_TAGS_DEFAULT)),
            dropout_p=float(cfg.get("dropout_p", DROPOUT)),
        )


class SpectreV3Model(nn.Module):
    """The v3 listwise re-ranker.

    Submodule names (``scene`` / ``cands`` / ``facts`` / ``scorer`` / ``aux``) are fixed
    by the compat contract above -- renaming one silently breaks checkpoint loading, so
    ``test_v3_equivalence.py`` pins the key set.
    """

    def __init__(
        self,
        n_ops: int,
        max_arity: int,
        cfg: Optional[V3Config] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or V3Config()
        c = self.cfg
        scene_cls = SceneEncoderV3 if c.use_obj_evidence else SceneEncoder
        self.scene = scene_cls(c.max_tags, c.dropout_p)
        cand_cls = CandidateEncoderV3 if c.sinusoidal_pos else CandidateEncoder
        self.cands = cand_cls(n_ops, c.max_tags, max_arity, c.dropout_p)
        self.facts = FactEncoder(c.max_tags, c.dropout_p)
        scorer_cls = CrossAttentionScorerV3 if c.evidence_attn else CrossAttentionScorer
        self.scorer = scorer_cls(c.n_overlap_feats, c.n_prior_feats, c.dropout_p)
        self.aux = AuxHead()
        # Additive by construction: the record encoder only exists when asked for, so a
        # default-config state dict is byte-identical to v2.2's (D-8) and the equivalence
        # oracle keeps loading.
        self.records = (
            RecordEncoder(n_ops, c.max_tags, c.dropout_p) if c.use_records else None
        )
        if c.use_necessity:  # pragma: no cover - cut from v3 scope, see decisions.md
            raise NotImplementedError(
                "necessity conditioning was cut from v3 (decisions.md 2026-07-26): D2 "
                "showed the s2 deficit is within-length, which it does not address"
            )

    def forward(
        self, batch: SpectreV3Batch, overlap: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """``(logits (B, K), aux (B, M, 2))`` -- the v2.2 contract, unchanged.

        Logits are ``-inf`` at unavailable candidates (pads, and during a rollout the
        already-tried ones), so ``argmax`` is the next attempt.
        """
        scene_tok = self.scene(batch)
        cand_emb = self.cands(batch)
        # Evidence memory: v3 record tokens when enabled, else the legacy fact tokens.
        # Never both -- they encode the same failures, so stacking them would double-count
        # the evidence and make the increment unattributable.
        fact_tok = None
        fact_mask = batch.fact_mask
        if (
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
        prior = batch.cand_prior if self.cfg.n_prior_feats else None
        if overlap is None and self.cfg.n_overlap_feats:
            overlap = batch.cand_overlap
        logits = self.scorer(
            cand_emb,
            scene_tok,
            batch.obj_mask,
            batch.glob_feats,
            overlap,
            fact_tok,
            fact_mask,
            prior,
        )
        avail = batch.avail_mask if batch.avail_mask is not None else batch.pool_mask
        logits = logits.masked_fill(~avail, float("-inf"))
        return logits, self.aux(scene_tok)


def load_v2_checkpoint(path, map_location="cpu") -> tuple[SpectreV3Model, dict]:
    """Load a ``train_v2`` checkpoint into a compat-mode :class:`SpectreV3Model`.

    Returns ``(model, cfg)``. ``strict=True`` on purpose: a silent key mismatch would
    produce a randomly-initialized submodule that still runs and still emits plausible
    logits, which is exactly the failure the equivalence oracle exists to catch.
    """
    import torch

    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["cfg"]
    model = SpectreV3Model(
        n_ops=int(ck["n_ops"]),
        max_arity=_max_arity_from_state_dict(ck["state_dict"]),
        cfg=V3Config.from_v2_checkpoint_cfg(cfg),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, cfg


def _max_arity_from_state_dict(state_dict) -> int:
    """Recover ``max_arity`` from the candidate encoder's argument projection.

    ``CandidateEncoder.arg_proj`` is ``Linear(max_arity * D_TAG, D_MODEL)``, and
    ``max_arity`` is a property of the *vocab*, not of the checkpoint's cfg dict -- so it
    is not stored and must be read back off the weights or the load fails on a shape
    mismatch that reads like a corrupted file.
    """
    from alphatamp.approaches.spectre.model_v2 import D_TAG

    w = state_dict["cands.arg_proj.weight"]
    assert w.shape[0] == D_MODEL, w.shape
    return int(w.shape[1] // D_TAG)
