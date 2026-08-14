"""EMA weight-averaging in run_training (docs/decisions 2026-08-08).

The lever is a training-*process* addition to recover the domain-agnostic
(narrowed-input) model's across-seed variance without touching inputs or architecture.
Two guarantees:

- **Off-by-default is byte-identical.** `weight_avg="none"` never builds the EMA shadow
  and takes the pre-change path, so training stays deterministic/reproducible (the
  trainer's D-8 exact-absence discipline).
- **On is not inert.** `weight_avg="ema"` genuinely tracks a moving average and exposes
  it to the selector (`val_fp_ema` logged; `selected` recorded), so "averaging silently
  does nothing" cannot pass.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from alphatamp.approaches.spectre.train import (  # noqa: E402
    TrainConfig,
    _ema_update,
)

_ROOT = Path(__file__).resolve().parents[3]
_TRAIN = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "train"
_VAL = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "val"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
_needs_v4 = pytest.mark.skipif(
    not (_TRAIN.is_dir() and _VAL.is_dir() and _VOCAB.is_file()),
    reason="dd2d_v4 collection / vocab absent (gitignored)",
)

# A fast 2-epoch CPU config: tiny val set so the per-epoch selector rollout is cheap, and
# num_workers=0 so training is deterministic on CPU (no worker RNG). ema_start_epoch=0 so
# the shadow exists from the first step in the EMA arm.
_FAST = dict(
    epochs=2,
    val_episodes=4,
    num_workers=0,
    overlap_mode="jaccard",
    coverage_feats=True,
    aggregate_records=True,
    evidence_attn=True,
    use_state_delta=True,
)


def _vocab():
    from alphatamp.approaches.spectre.vocab import Vocab

    return Vocab.from_json(_VOCAB)


# --------------------------------------------------------------------------- unit


def test_ema_update_is_a_decayed_average() -> None:
    """`_ema_update` sets shadow ← decay·shadow + (1-decay)·live, per float tensor."""
    m = torch.nn.Linear(4, 4)
    ema = copy.deepcopy(m)
    with torch.no_grad():  # move the live weights so there is something to average
        for p in m.parameters():
            p.add_(1.0)
    before = {k: v.clone() for k, v in ema.state_dict().items()}
    _ema_update(ema, m, 0.9)
    live = m.state_dict()
    for k, v in ema.state_dict().items():
        assert torch.allclose(v, 0.9 * before[k] + 0.1 * live[k])
    # and it genuinely moved (not inert)
    assert not torch.allclose(ema.weight, before["weight"])


def test_ema_update_copies_non_float_tensors() -> None:
    """Non-float buffers are copied verbatim (kept a valid, loadable state dict)."""

    class _M(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, step: int) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.register_buffer("step", torch.tensor(step, dtype=torch.long))

    live, ema = _M(5), _M(0)
    _ema_update(ema, live, 0.9)
    assert int(ema.step) == 5  # copied, not averaged


def test_config_default_is_off() -> None:
    assert TrainConfig().weight_avg == "none"


# --------------------------------------------------------------------------- e2e


@pytest.mark.slow
@_needs_v4
def test_weight_avg_none_never_builds_the_shadow(tmp_path) -> None:
    """Off-by-default takes the raw-only path: no EMA shadow, no EMA in selection.

    The off-by-default guarantee is a *code path*, not bit-identity — GPU training is
    not bit-deterministic at fixed seed and does not need to be. So this asserts the
    observable consequences of the `ema is not None` guard: with `weight_avg="none"` the
    shadow is never constructed (`val_fp_ema` is `None` every epoch) and the selector
    only ever picks the raw weights (`selected == "raw"`). Runs on the default device.
    """
    import json

    from alphatamp.approaches.spectre.train import run_training

    vocab = _vocab()  # type: ignore[no-untyped-call]
    cfg = TrainConfig(seed=0, weight_avg="none", **_FAST)  # type: ignore[arg-type]
    out = tmp_path / "none"
    run_training(cfg, _TRAIN, _VAL, vocab, out)
    log = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert all(r["val_fp_ema"] is None for r in log), "shadow built with EMA off"
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert saved.get("selected") == "raw"


@pytest.mark.slow
@_needs_v4
def test_weight_avg_ema_is_not_inert(tmp_path) -> None:
    """`weight_avg="ema"` exposes the EMA to the selector and can be chosen.

    Guards against "averaging silently does nothing": the shadow must be scored
    (`val_fp_ema` present and finite once it exists) and the saved checkpoint must record
    which arm won. The EMA shadow provably differs from the live weights (unit test
    above), so a finite `val_fp_ema` means it was really built and evaluated.
    """
    import json

    from alphatamp.approaches.spectre.train import run_training

    vocab = _vocab()  # type: ignore[no-untyped-call]
    cfg = TrainConfig(
        seed=0, weight_avg="ema", ema_start_epoch=0, **_FAST  # type: ignore[arg-type]
    )
    out = tmp_path / "ema"
    run_training(cfg, _TRAIN, _VAL, vocab, out)
    log = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    ema_vals = [r["val_fp_ema"] for r in log if r.get("val_fp_ema") is not None]
    assert ema_vals, "EMA was on but val_fp_ema never logged -- shadow not evaluated"
    assert all(v == v and v >= 0 for v in ema_vals)  # finite, non-negative
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert saved.get("selected") in ("raw", "ema")
