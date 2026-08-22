"""Rung-1 evidence-step pathway (docs/failed_records_fix.md F-A / F-B2).

Guards the additive, checkpoint-safe contract of ``record_mode="steps"`` +
``use_step_join``:

- **off is byte-structural-identical** — a summary / no-join model builds no
  ``rec_steps.*`` / ``step_join.*`` params (so old checkpoints load ``strict=True``);
- **the shared CandidateEncoder is not double-registered** (no ``rec_steps.cands.*``);
- **the step-join is a zero-init no-op at step 0** — a steps+join model equals a steps-only
  model before any training;
- **the on path runs** forward + backward with finite logits at available candidates;
- **the checkpoint round-trips** the new switches through ``load_checkpoint``.

The structural checks are data-free; the forward / round-trip checks tensorize one real
episode and skip when the collection is absent.
"""

from __future__ import annotations

import glob
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from alphatamp.approaches.spectre.dataset import build_example, collate
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference import load_checkpoint
from alphatamp.approaches.spectre.io import load_episode
from alphatamp.approaches.spectre.model import SpectreConfig, SpectreModel
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[3]
_VARIANT = "restock3d_v3"  # has mid-plan failures with culprits -> establishing steps


def _cfg(**kw) -> SpectreConfig:
    return SpectreConfig(
        use_records=True,
        evidence_attn=True,
        n_predicates=4,
        max_pred_arity=2,
        max_tags=32,
        **kw,
    )


def _model(seed=0, **kw) -> SpectreModel:
    torch.manual_seed(seed)
    return SpectreModel(n_ops=6, max_arity=2, cfg=_cfg(**kw))


def test_off_path_builds_no_rung1_params() -> None:
    keys = set(_model().state_dict())
    assert not any(k.startswith("rec_steps.") for k in keys)
    assert not any(k.startswith("step_join.") for k in keys)


def test_steps_mode_swaps_records_for_step_encoder() -> None:
    keys = set(_model(record_mode="steps", use_step_join=True).state_dict())
    assert any(k.startswith("rec_steps.") for k in keys)
    assert any(k.startswith("step_join.") for k in keys)
    # steps mode replaces the summary RecordEncoder (no dead params) ...
    assert not any(k.startswith("records.") for k in keys)
    # ... and reuses self.cands without double-registering its weights.
    assert not any(k.startswith("rec_steps.cands") for k in keys)


def _episode_batch():
    raw = REPO / "data" / "spectre" / "raw" / _VARIANT / "train" / "episodes"
    vpath = REPO / "data" / "spectre" / "derived" / _VARIANT / "train_vocab.json"
    paths = sorted(glob.glob(str(raw / "*")))
    if not paths or not vpath.exists():
        pytest.skip(f"no collection for {_VARIANT}")
    vocab = Vocab.from_json(vpath)
    spec = spec_for(_VARIANT)
    for p in paths:
        ep = load_episode(Path(p))
        fail = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if len(fail) >= 3 and ep.scene_geometry is not None:
            ex, recs = build_example(
                ep,
                vocab,
                evidence=True,
                context_f=frozenset(fail[:6]),
                spec=spec,
                record_mode="steps",
                aggregate_records=True,
            )
            batch = collate(
                [ex],
                max_arity=vocab.max_operator_arity,
                records=[recs],
                max_pred_arity=vocab.max_predicate_arity,
            )
            return batch, vocab
    pytest.skip("no failure-bearing episode")


def _real_model(vocab, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = SpectreConfig(
        use_records=True,
        evidence_attn=True,
        n_predicates=len(vocab.predicates),
        max_pred_arity=vocab.max_predicate_arity,
        max_tags=32,
        **kw,
    )
    return SpectreModel(
        n_ops=len(vocab.operators), max_arity=vocab.max_operator_arity, cfg=cfg
    )


def test_steps_forward_backward_finite() -> None:
    batch, vocab = _episode_batch()
    m = _real_model(vocab, record_mode="steps", use_step_join=True)
    logits, _ = m(batch)
    avail = batch.avail_mask if batch.avail_mask is not None else batch.pool_mask
    assert torch.isfinite(logits[avail]).all()
    logits[avail].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_step_join_is_zero_init_noop() -> None:
    batch, vocab = _episode_batch()
    m_join = _real_model(vocab, record_mode="steps", use_step_join=True).eval()
    m_no = _real_model(vocab, record_mode="steps", use_step_join=False).eval()
    m_no.load_state_dict(
        {
            k: v
            for k, v in m_join.state_dict().items()
            if not k.startswith("step_join.")
        },
        strict=False,
    )
    avail = batch.avail_mask if batch.avail_mask is not None else batch.pool_mask
    lj, _ = m_join(batch)
    ln, _ = m_no(batch)
    assert torch.allclose(lj[avail], ln[avail], atol=1e-6)


def test_checkpoint_roundtrips_rung1_switches(tmp_path) -> None:
    batch, vocab = _episode_batch()
    m = _real_model(vocab, record_mode="steps", use_step_join=True)
    cfg_dict = asdict(m.cfg)
    cfg_dict["record_mode"] = (
        "steps"  # emission switch persisted by TrainConfig normally
    )
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "cfg": cfg_dict,
            "n_ops": len(vocab.operators),
            "state_dict": m.state_dict(),
        },
        ckpt,
    )
    _loaded, deploy = load_checkpoint(ckpt, vocab, "cpu")
    assert deploy["record_mode"] == "steps"
    assert any(k.startswith("step_join.") for k in _loaded.state_dict())
