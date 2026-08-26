"""W2 probe: `evidence_context` / `evidence_cap_k` caps the failure EVIDENCE the model
conditions on (record + hint-fact tokens) to a subset of the context, WITHOUT changing which
candidates remain available (`avail_mask`) or, in the X2 residual, what |F| the gate reads.

Eval-only diagnostic (docs/failed_records_fix_part2.md §2). Off (`None`) is byte-identical to
today; on, it shrinks the evidence memory. Re-try prevention is the rollout's `_TRIED` sentinel,
so capping the evidence context never lets a tried candidate be picked again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphatamp.approaches.spectre.dataset import build_example
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
_needs_data = pytest.mark.skipif(not _V4.is_dir(), reason="dd2d_v4 collection absent")


def _episode(min_fails: int = 10):
    for p in list_episodes(_V4):
        ep = load_episode(p)
        fails = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if len(fails) >= min_fails:
            return ep, frozenset(fails[:min_fails])
    pytest.skip("no test episode with enough failures")


def _build(ep, vocab, ctx, ev_ctx):
    return build_example(
        ep,
        vocab,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        spec=spec_for(ep.provenance.env_variant),
        aggregate_records=True,
        evidence_context=ev_ctx,
    )


@_needs_data
def test_evidence_context_none_is_byte_identical() -> None:
    """Not passing evidence_context == passing None == passing the full ctx."""
    vocab = Vocab.from_json(_VOCAB)
    ep, ctx = _episode()
    ex0, r0 = build_example(
        ep,
        vocab,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        spec=spec_for(ep.provenance.env_variant),
        aggregate_records=True,
    )
    ex1, r1 = _build(ep, vocab, ctx, None)
    ex2, r2 = _build(ep, vocab, ctx, ctx)
    assert r0 == r1 == r2
    assert ex0.avail == ex1.avail == ex2.avail


@_needs_data
def test_cap_shrinks_records_but_not_avail() -> None:
    """Capping the evidence context reduces record tokens; avail_mask is untouched."""
    vocab = Vocab.from_json(_VOCAB)
    ep, ctx = _episode()
    ex_full, r_full = _build(ep, vocab, ctx, None)
    for k in (1, 2, 4):
        ev = frozenset(sorted(ctx)[-k:])
        ex_k, r_k = _build(ep, vocab, ctx, ev)
        assert len(r_k) <= len(r_full)
        assert len(r_k) <= k  # at most one aggregated record per capped candidate
        # avail (re-try mask) is on the FULL context regardless of the evidence cap.
        assert ex_k.avail == ex_full.avail


@_needs_data
def test_rollout_evidence_cap_runs_and_prevents_retry() -> None:
    """A capped rollout completes and never re-tries a candidate (the _TRIED sentinel)."""

    from alphatamp.approaches.spectre.inference import (
        deployed_rollout_traced,
        load_checkpoint,
    )

    ckpt = (
        _ROOT
        / "data"
        / "spectre"
        / "checkpoints_spectre_noov_atoms_residual_records"
        / "dd2d_v4"
        / "seed_0"
        / "best.pt"
    )
    if not ckpt.is_file():
        pytest.skip("X2 residual seed-0 checkpoint absent")
    vocab = Vocab.from_json(_VOCAB)
    model, deploy = load_checkpoint(ckpt, vocab, "cpu")
    ep, _ = _episode()
    for k in (None, 1, 4):
        attempts, trace = deployed_rollout_traced(
            model, ep, vocab, "cpu", evidence_cap_k=k, **deploy
        )
        assert attempts >= 1
        assert len(trace.order) == len(set(trace.order))  # no candidate tried twice
