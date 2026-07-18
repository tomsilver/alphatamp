"""Tests for the optional wandb hook in ``spectre.train``.

Uses a fake duck-typed run (no network, no wandb import) to confirm per-epoch
metrics are logged with the expected flat keys, and that the ``wandb_run=None``
path is an exact no-op.
"""

from __future__ import annotations

import math
from pathlib import Path

from _fixtures import write_toy_split

from alphatamp.approaches.spectre.train import (
    TrainingConfig,
    _flatten_for_wandb,
    train,
)
from alphatamp.approaches.spectre.vocab import extract_vocab


class _FakeRun:
    """Records ``.log()`` calls and exposes a ``.summary`` dict like wandb.Run."""

    def __init__(self) -> None:
        self.logs: list[tuple[int | None, dict[str, float]]] = []
        self.summary: dict[str, object] = {}

    def log(self, data: dict[str, float], step: int | None = None) -> None:
        """Record a logged metrics dict, mirroring ``wandb.Run.log``."""
        self.logs.append((step, dict(data)))


def _toy_cfg() -> TrainingConfig:
    # val_loss selection + no rollout eval keeps the eda/rollout path (and its
    # heavy imports) out of this unit test.
    return TrainingConfig(
        epochs=2,
        batch_size=2,
        warmup_steps=1,
        num_workers=0,
        augment=False,
        num_f_samples_per_epoch=2,
        num_f_samples_per_val_episode=1,
        rollout_eval_each_epoch=False,
        checkpoint_metric="val_loss",
        early_stop_patience=0,
    )


def _write_toy_splits(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    write_toy_split(
        train_dir,
        [
            ("fail", "success"),
            ("success", "fail"),
            ("fail", "fail", "success"),
            ("success", "fail"),
        ],
    )
    write_toy_split(val_dir, [("fail", "success"), ("success", "fail")])
    vocab = extract_vocab(train_dir, config_hash="abc")
    return train_dir, val_dir, vocab


def test_flatten_for_wandb_skips_none_and_nan() -> None:
    """Flatten produces slashed scalar keys and drops None / NaN entries."""
    record = {
        "train_loss": 1.5,
        "val_loss": 2.0,
        "lr": 1e-4,
        "global_step": 10,
        "auroc": {"0": 0.6, "3": None},
        "top1": {"0": 0.5},
        "val_rollout": {"mean_attempts": float("nan"), "std_attempts": 1.0},
    }
    flat = _flatten_for_wandb(record)
    assert flat["train/loss"] == 1.5
    assert flat["val/loss"] == 2.0
    assert flat["val/auroc_0"] == 0.6
    assert "val/auroc_3" not in flat  # None dropped
    assert "val_rollout/mean_attempts" not in flat  # NaN dropped
    assert flat["val_rollout/std_attempts"] == 1.0
    assert all(isinstance(v, float) for v in flat.values())


def test_train_logs_to_fake_run(tmp_path: Path) -> None:
    """Train() logs one wandb record per epoch with the expected keys + summary."""
    train_dir, val_dir, vocab = _write_toy_splits(tmp_path)
    run = _FakeRun()
    cfg = _toy_cfg()

    train(
        cfg=cfg,
        train_dir=train_dir,
        val_dir=val_dir,
        vocab=vocab,
        type_aug_policy=None,
        out_dir=tmp_path / "out",
        wandb_run=run,
    )

    # One log per epoch, stepped by epoch index.
    assert len(run.logs) == cfg.epochs
    assert [step for step, _ in run.logs] == list(range(cfg.epochs))
    first = run.logs[0][1]
    for key in ("train/loss", "val/loss", "lr", "checkpoint/is_best"):
        assert key in first
        assert not math.isnan(first[key])
    # Summary carries the epoch-0 random-Φ baseline and the best-checkpoint info.
    assert "random_phi_baseline" in run.summary
    assert run.summary["best/checkpoint_metric"] == "val_loss"
    assert isinstance(run.summary["best/epoch"], int)


def test_train_without_run_is_noop(tmp_path: Path) -> None:
    """wandb_run=None trains normally and writes best.pt."""
    train_dir, val_dir, vocab = _write_toy_splits(tmp_path)
    best_path = train(
        cfg=_toy_cfg(),
        train_dir=train_dir,
        val_dir=val_dir,
        vocab=vocab,
        type_aug_policy=None,
        out_dir=tmp_path / "out",
        wandb_run=None,
    )
    assert best_path.exists()
