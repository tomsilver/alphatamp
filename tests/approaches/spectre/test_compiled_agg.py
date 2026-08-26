"""X1: compiled evidence aggregation (docs/failed_records_fix_part2.md §3).

The residual reads the failure records with either the soft `evid_attn` (default "attention")
or the compiled `CompiledEvidenceAgg` -- a learned per-(candidate, record) read reduced by a
**hand-fixed** `sum`/`max` (the quantifier soft attention cannot reliably induce). "attention" is
byte-identical to the X2 residual; the compiled modes swap `evid_attn` for `compiled_agg` and,
because the adaptive_head is still zero-init, keep the X2 "≡ static at init" guarantee.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from alphatamp.approaches.spectre.dataset import build_example, collate
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model import (
    CompiledEvidenceAgg,
    ResidualEvidenceScorer,
    SpectreConfig,
    SpectreModel,
)
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
_needs_data = pytest.mark.skipif(not _V4.is_dir(), reason="dd2d_v4 collection absent")
_STATIC = dict(n_overlap_feats=0, max_tags=32, dropout_p=0.0)


def _vocab() -> Vocab:
    return Vocab.from_json(_VOCAB)


def _model(cfg: SpectreConfig, vocab: Vocab, seed: int = 0) -> SpectreModel:
    torch.manual_seed(seed)
    return SpectreModel(
        n_ops=len(vocab.operators), max_arity=vocab.max_operator_arity, cfg=cfg
    ).eval()


def _resid_cfg(agg: str) -> SpectreConfig:
    return SpectreConfig(**_STATIC, use_records=True, residual_adaptive=True, evidence_agg=agg)  # type: ignore[arg-type]


def test_evidence_agg_selects_the_right_module() -> None:
    """attention builds evid_attn; sum/max build compiled_agg instead (never both)."""
    vocab = _vocab()
    att = _model(_resid_cfg("attention"), vocab).scorer
    assert isinstance(att, ResidualEvidenceScorer)
    assert att.evid_attn is not None and att.compiled_agg is None
    for agg in ("sum", "max"):
        sc = _model(_resid_cfg(agg), vocab).scorer
        assert isinstance(sc, ResidualEvidenceScorer)
        assert sc.evid_attn is None and isinstance(sc.compiled_agg, CompiledEvidenceAgg)


def test_compiled_agg_masks_no_record_rows_to_zero_and_is_finite() -> None:
    """A batch row with no valid record reduces to 0 (both sum and max); all finite."""
    torch.manual_seed(0)
    b, k, r, d = 2, 3, 4, 64
    cand = torch.randn(b, k, d)
    fact = torch.randn(b, r, d)
    mask = torch.ones(b, r, dtype=torch.bool)
    mask[1] = False  # second episode has no valid records
    for agg in ("sum", "max"):
        ev = CompiledEvidenceAgg(agg, 0.0).eval()(cand, fact, mask)
        assert ev.shape == (b, k, d)
        assert torch.isfinite(ev).all()
        assert torch.count_nonzero(ev[1]) == 0  # no-record row collapses to 0


@_needs_data
def test_x1_warm_started_equals_static_at_init() -> None:
    """A compiled-agg (sum) residual warm-started from static scores identically at init."""
    vocab = _vocab()
    static = _model(SpectreConfig(**_STATIC), vocab, seed=0)  # type: ignore[arg-type]
    resid = _model(_resid_cfg("sum"), vocab, seed=1)
    resid.load_state_dict(static.state_dict(), strict=False)

    for path in list_episodes(_V4):
        ep = load_episode(path)
        fails = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if fails:
            ctx = frozenset(fails[:20])
            break
    else:
        pytest.skip("no failures")
    ex, recs = build_example(
        ep,
        vocab,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        spec=spec_for(ep.provenance.env_variant),
        aggregate_records=True,
    )
    batch = collate(
        [ex],
        max_arity=vocab.max_operator_arity,
        records=[recs],
        max_pred_arity=vocab.max_predicate_arity,
    )
    with torch.no_grad():
        ls, _ = static(batch)
        lr, _ = resid(batch)
    finite = torch.isfinite(ls)
    assert finite.any()
    assert torch.equal(ls[finite], lr[finite])


def test_x1_checkpoint_round_trips_evidence_agg(tmp_path) -> None:
    """A compiled-agg checkpoint reloads strict with the same aggregation mode."""
    from alphatamp.approaches.spectre.inference import load_checkpoint
    from alphatamp.approaches.spectre.train import TrainConfig

    vocab = _vocab()
    model = _model(_resid_cfg("max"), vocab)
    cfg = TrainConfig(
        use_overlap=False, use_records=True, residual_adaptive=True, evidence_agg="max"
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
    assert loaded.scorer.compiled_agg is not None
    assert loaded.scorer.compiled_agg.reduce == "max"
