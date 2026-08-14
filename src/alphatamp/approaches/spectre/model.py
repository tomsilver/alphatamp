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


class RecordEncoder(nn.Module):
    """One observed failure -> one token, with the object roles kept apart.

    Replaces `FactEncoder`'s hand-built type vocabulary with the domain's own operator
    schemas, and finally consumes the scalars v2.2 harvested and then dropped on the
    floor (`Fact.scalars` never reached the tensorizer). No tier embedding: only
    hint-tier evidence ever entered the network, so it was a constant column.
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
    """Architecture switches. Every v3 feature defaults **off**, so the default config
    reproduces deployed v2.2 exactly (D-8).

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
        self.scene = SceneEncoder(c.max_tags, c.dropout_p, d_rel=c.d_rel)
        self.cands = CandidateEncoder(n_ops, c.max_tags, max_arity, c.dropout_p)
        self.facts = FactEncoder(c.max_tags, c.dropout_p)
        scorer_cls = (
            EvidenceCrossAttentionScorer if c.evidence_attn else CrossAttentionScorer
        )
        self.scorer = scorer_cls(c.n_overlap_feats, c.n_prior_feats, c.dropout_p)
        self.aux = AuxHead()
        # Additive by construction: the record encoder only exists when asked for, so a
        # default-config state dict is byte-identical to v2.2's (D-8) and the
        # equivalence oracle keeps loading.
        self.records = (
            RecordEncoder(
                n_ops,
                c.max_tags,
                c.dropout_p,
                c.n_predicates,
                c.max_pred_arity,
                c.use_state_delta,
            )
            if c.use_records
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
        scene_tok = self.scene(batch)
        cand_emb = self.cands(batch)
        # Evidence memory: v3 record tokens when enabled, else the legacy fact tokens.
        # Never both -- they encode the same failures, so stacking them would
        # double-count the evidence and make the increment unattributable.
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
