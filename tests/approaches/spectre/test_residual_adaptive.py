"""X2: the record/evidence channel as a zero-init, |F|-gated residual over a frozen,
warm-started static trunk (``ResidualEvidenceScorer``, ``--residual-adaptive``).

The load-bearing guarantee is **"cannot be worse than static at init"**: with the static
half warm-started + frozen and the residual output zero-initialized, a residual model built
from a static checkpoint reproduces the static ranker bit-for-bit before any training. The
tests pin that, plus the usual additive-branch invariants (off adds no residual keys / static
checkpoints load ``strict=True``; the switch round-trips through the checkpoint; the |F| gate
reads the context size from ``avail_mask``).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from alphatamp.approaches.spectre.dataset import build_example, collate
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model import (
    ResidualEvidenceScorer,
    SpectreConfig,
    SpectreModel,
)
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
_needs_data = pytest.mark.skipif(not _V4.is_dir(), reason="dd2d_v4 collection absent")

# Lean pure-static backbone (no overlap, no scalars) -- the X2 base. Point-set/atoms off to
# keep the unit test fast; the static/residual split is independent of the backbone width.
_STATIC = dict(n_overlap_feats=0, max_tags=32, dropout_p=0.0)


def _vocab() -> Vocab:
    return Vocab.from_json(_VOCAB)


def _static_cfg() -> SpectreConfig:
    return SpectreConfig(**_STATIC)  # type: ignore[arg-type]


def _residual_cfg() -> SpectreConfig:
    return SpectreConfig(**_STATIC, use_records=True, residual_adaptive=True)  # type: ignore[arg-type]


def _model(cfg: SpectreConfig, vocab: Vocab, seed: int = 0) -> SpectreModel:
    torch.manual_seed(seed)
    return SpectreModel(
        n_ops=len(vocab.operators),
        max_arity=vocab.max_operator_arity,
        cfg=cfg,
    ).eval()


def _episode_with_evidence():
    """A test episode + a non-empty failure context (stride, never truncate)."""
    paths = list_episodes(_V4)
    for path in paths[:: max(1, len(paths) // 8)]:
        episode = load_episode(path)
        fails = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        if fails:
            return episode, frozenset(fails[:30])
    pytest.skip("no test episode with a failure")


def _batch(episode, vocab, ctx) -> Any:
    example, records = build_example(
        episode,
        vocab,
        rng=None,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        overlap_mode="jaccard",
        aggregate_records=True,
    )
    return (
        collate(
            [example],
            max_arity=vocab.max_operator_arity,
            records=[records],
            max_pred_arity=vocab.max_predicate_arity,
        ),
        ctx,
    )


# ----------------------------------------------------------------- off is exact absence


def test_residual_off_builds_no_residual_scorer() -> None:
    """Default config keeps the ordinary scorer -- no adaptive_head / gate keys."""
    vocab = _vocab()
    off = _model(_static_cfg(), vocab)
    keys = set(off.state_dict())
    assert not any("adaptive_head" in k or "scorer.gate" in k for k in keys)
    assert not isinstance(off.scorer, ResidualEvidenceScorer)

    on = _model(_residual_cfg(), vocab)
    assert isinstance(on.scorer, ResidualEvidenceScorer)
    new = set(on.state_dict()) - keys
    # The residual adds exactly its own submodules (records + the two residual heads +
    # the evidence attention); the static scorer.head keeps its 2*D_MODEL shape.
    assert any("scorer.adaptive_head" in k for k in new)
    assert any("scorer.gate" in k for k in new)
    assert any("scorer.evid_attn" in k for k in new)
    assert on.state_dict()["scorer.head.0.weight"].shape == (
        off.state_dict()["scorer.head.0.weight"].shape
    )


def test_residual_output_heads_are_zero_init() -> None:
    """adjustment output and gate output are zero-initialized (step-0 residual is 0)."""
    scorer = ResidualEvidenceScorer(0, 0, 0.0)
    for seq in (scorer.adaptive_head, scorer.gate):
        last = seq[-1]
        assert isinstance(last, torch.nn.Linear)
        assert torch.count_nonzero(last.weight) == 0
        assert torch.count_nonzero(last.bias) == 0


# ----------------------------------------------------- the "cannot be worse than static"


@_needs_data
def test_warm_started_residual_equals_static_at_init() -> None:
    """A residual model warm-started from a static one scores IDENTICALLY at init.

    This is the X2 guarantee: static half frozen+warm-started, residual zero-init -> the
    adjustment is exactly 0, so ``logit = static_logit`` before any training. Same static
    weights + a `+0.0`, so the logits are bit-equal (not merely close).
    """
    vocab = _vocab()
    static = _model(_static_cfg(), vocab, seed=0)
    resid = _model(_residual_cfg(), vocab, seed=1)  # different init draw on purpose

    # Warm-start: copy the shared static keys into the residual model (as train.py does).
    static_sd = static.state_dict()
    incompat = resid.load_state_dict(static_sd, strict=False)
    assert not incompat.unexpected_keys  # static is a subset of the residual model
    # The only unloaded keys are the residual additions.
    for k in incompat.missing_keys:
        assert k.startswith("records.") or k.startswith(
            ("scorer.evid_attn", "scorer.adaptive_head", "scorer.gate")
        )

    episode, ctx = _episode_with_evidence()
    batch, _ = _batch(episode, vocab, ctx)
    with torch.no_grad():
        logits_static, _ = static(batch)
        logits_resid, _ = resid(batch)
    finite = torch.isfinite(logits_static)
    assert finite.any()
    assert torch.equal(logits_static[finite], logits_resid[finite])


@_needs_data
def test_freeze_partitions_static_and_residual_params() -> None:
    """Freezing the loaded static keys leaves exactly the residual params trainable."""
    vocab = _vocab()
    static_sd = _model(_static_cfg(), vocab).state_dict()
    resid = _model(_residual_cfg(), vocab, seed=1)
    resid.load_state_dict(static_sd, strict=False)

    frozen_keys = set(static_sd)
    for name, p in resid.named_parameters():
        if name in frozen_keys:
            p.requires_grad_(False)

    trainable = {n for n, p in resid.named_parameters() if p.requires_grad}
    # Nothing static trains; every trainable param is a residual one.
    assert trainable and all(
        n.startswith("records.")
        or n.startswith(("scorer.evid_attn", "scorer.adaptive_head", "scorer.gate"))
        for n in trainable
    )
    # And the static trunk (scene/cands/scorer.head) is fully frozen.
    assert all(
        not p.requires_grad
        for n, p in resid.named_parameters()
        if n.startswith(("scene.", "cands.", "scorer.head", "scorer.attn"))
    )


@_needs_data
def test_gate_reads_context_size_from_avail_mask() -> None:
    """|F| = number of in-context candidates, derived from avail_mask (no new field)."""
    vocab = _vocab()
    episode, ctx = _episode_with_evidence()
    batch, ctx = _batch(episode, vocab, ctx)
    avail = batch.avail_mask if batch.avail_mask is not None else batch.pool_mask
    n_ctx = int(((~avail) & batch.pool_mask).sum())
    assert n_ctx == len(ctx)


def test_checkpoint_round_trips_residual_adaptive(tmp_path) -> None:
    """A residual checkpoint saved via asdict(TrainConfig) reloads strict with the scorer."""
    from alphatamp.approaches.spectre.inference import load_checkpoint
    from alphatamp.approaches.spectre.train import TrainConfig

    vocab = _vocab()
    model = _model(_residual_cfg(), vocab)
    cfg = TrainConfig(
        use_overlap=False,
        use_records=True,
        evidence_attn=True,
        residual_adaptive=True,
    )
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "n_ops": len(vocab.operators),
            "selected": "raw",
        },
        ckpt,
    )
    loaded, _ = load_checkpoint(ckpt, vocab, "cpu")
    assert isinstance(loaded.scorer, ResidualEvidenceScorer)
    assert loaded.cfg.residual_adaptive
