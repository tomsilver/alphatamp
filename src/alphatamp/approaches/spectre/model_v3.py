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

from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn

from alphatamp.approaches.spectre.model import D_MODEL
from alphatamp.approaches.spectre.model_v2 import (
    MAX_TAGS_DEFAULT,
    AuxHead,
    CandidateEncoder,
    CrossAttentionScorer,
    FactEncoder,
    SceneEncoder,
    SpectreV2Batch,
)

# The v3 batch is the v2 batch until a gate adds a field to it; aliased rather than
# re-declared so the tensorizers and the model cannot drift apart.
SpectreV3Batch = SpectreV2Batch

DROPOUT = 0.1


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
        self.scene = SceneEncoder(c.max_tags, c.dropout_p)
        self.cands = CandidateEncoder(n_ops, c.max_tags, max_arity, c.dropout_p)
        self.facts = FactEncoder(c.max_tags, c.dropout_p)
        self.scorer = CrossAttentionScorer(
            c.n_overlap_feats, c.n_prior_feats, c.dropout_p
        )
        self.aux = AuxHead()
        if c.use_records or c.use_necessity:  # pragma: no cover - later gates
            raise NotImplementedError(
                "use_records (G6) / use_necessity (G8) are not built yet; "
                "they must land as additional submodules, never by mutating compat mode"
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
        fact_tok = None
        if batch.fact_type_ids is not None and batch.fact_type_ids.shape[1] > 0:
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
            batch.fact_mask,
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
