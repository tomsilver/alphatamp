"""Training loop for the SPECTRE model on RT2D (and any kinder env).

Implements ``SPECTRE_RT2D_METHOD_SPEC.md`` §8: AdamW + cosine LR with linear
warmup, gradient clipping, F-subsample multiplier, prior dropout, per-epoch
validation with PL loss + AUROC(t) for ``t ∈ {0, 1, 2, 3}``.

Public surface:

- :class:`TrainingConfig` — all hyperparameters in one frozen-ish dataclass.
- :func:`train` — runs the full loop, writes ``best.pt`` + ``log.jsonl`` +
  ``model_meta.json`` under ``out_dir``.

The Hydra entrypoint at ``experiments/spectre_train.py`` constructs a
``TrainingConfig`` from cfg and calls :func:`train`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from alphatamp.approaches.spectre.dataset import (
    FSamplingConfig,
    SpectreBatch,
    SpectreDataset,
    SpectreTrainingExample,
    collate_spectre_batch,
)
from alphatamp.approaches.spectre.loss import plackett_luce_loss
from alphatamp.approaches.spectre.model import SpectreModel
from alphatamp.approaches.spectre.priors import BasePrior, ZeroPrior
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class TrainingConfig:
    """Hyperparameters for one SPECTRE training run."""

    # Optimizer + schedule
    lr: float = 3e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 16
    epochs: int = 20
    warmup_steps: int = 500
    lr_min: float = 1e-5
    grad_clip: float = 1.0

    # Regularization
    prior_dropout_p: float = 0.2
    augment: bool = True

    # F-sampling (spec §8.2 default mix weights)
    f_sampling_mode: str = "rollout_aligned_mix"
    f_sampling_mix_weights: tuple[float, float, float] = (0.25, 0.25, 0.5)
    f_sampling_log_normal_mu: float = 0.0
    f_sampling_log_normal_sigma: float = 1.0

    # F-subsample multiplier (spec §8.1 fix #5)
    num_f_samples_per_epoch: int = 8
    num_f_samples_per_val_episode: int = 4

    # Reproducibility
    seed: int = 0

    # Validation logging knobs
    auroc_t_max: int = 3  # AUROC(0..auroc_t_max)

    # DataLoader
    num_workers: int = 0  # 0 = single-process (LRU cache works simply)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Linear blend between min_lr and base_lr; LambdaLR multiplies the
        # base_lr passed to AdamW.
        scale = (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cos
        return scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_batch(batch: SpectreBatch, device: torch.device) -> SpectreBatch:
    """Shallow copy with every tensor moved to ``device``."""
    fields: dict[str, object] = {}
    for name, val in batch.__dict__.items():
        if isinstance(val, torch.Tensor):
            fields[name] = val.to(device)
        else:
            fields[name] = val
    return SpectreBatch(**fields)  # type: ignore[arg-type]


def _collate_with_vocab(vocab: Vocab):
    def _collate(batch: list[SpectreTrainingExample]) -> SpectreBatch:
        return collate_spectre_batch(batch, vocab)

    return _collate


def _build_f_sampling_config(cfg: TrainingConfig) -> FSamplingConfig:
    return FSamplingConfig(
        mode=cfg.f_sampling_mode,
        mix_weights=cfg.f_sampling_mix_weights,
        log_normal_mu=cfg.f_sampling_log_normal_mu,
        log_normal_sigma=cfg.f_sampling_log_normal_sigma,
    )


# ---------------------------------------------------------------------------
# Validation: PL loss + AUROC(t) + top-1 hit rate
# ---------------------------------------------------------------------------


def _safe_auroc(scores: list[float], labels: list[int]) -> float | None:
    """Return AUROC, or ``None`` if undefined (single-class labels).

    Uses a no-sklearn-dependency rank-sum implementation for portability.
    """
    if not labels:
        return None
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    # Mann-Whitney U / (|pos|*|neg|).
    ranks = _rank_with_ties(scores)
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _rank_with_ties(scores: list[float]) -> list[float]:
    """Average-rank handling for ties; ranks are 1-indexed."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@dataclass
class EvalReport:
    """Per-epoch validation metrics returned by :func:`_evaluate`."""

    val_loss: float
    auroc_by_t: dict[int, float | None]
    top1_by_t: dict[int, float | None]
    per_t_count: dict[int, int]


def _evaluate(
    model: SpectreModel,
    val_dataset: SpectreDataset,
    vocab: Vocab,
    cfg: TrainingConfig,
    device: torch.device,
) -> EvalReport:
    model.eval()
    loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_collate_with_vocab(vocab),
        num_workers=cfg.num_workers,
    )
    losses: list[float] = []
    # Per-|F| stratified scores/labels for AUROC(t); top-1 hit rate counts.
    scores_by_t: dict[int, list[float]] = defaultdict(list)
    labels_by_t: dict[int, list[int]] = defaultdict(list)
    top1_correct_by_t: dict[int, int] = defaultdict(int)
    top1_total_by_t: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = plackett_luce_loss(logits, batch.r_success_mask, batch.r_mask)
            losses.append(float(loss.item()))
            f_sizes = batch.f_mask.sum(dim=-1).cpu().tolist()
            r_mask = batch.r_mask.cpu()
            r_succ = batch.r_success_mask.cpu()
            logits_cpu = logits.cpu()
            for ex_idx, t in enumerate(f_sizes):
                t = int(t)
                if t > cfg.auroc_t_max:
                    continue
                row_logits = logits_cpu[ex_idx]
                row_mask = r_mask[ex_idx]
                row_succ = r_succ[ex_idx]
                # AUROC over R-valid slots only.
                for j in range(row_mask.numel()):
                    if not row_mask[j].item():
                        continue
                    scores_by_t[t].append(float(row_logits[j].item()))
                    labels_by_t[t].append(1 if row_succ[j].item() else 0)
                # Top-1 hit rate: argmax index over R-valid slots.
                masked = row_logits.clone()
                masked[~row_mask] = -float("inf")
                pick = int(masked.argmax().item())
                top1_total_by_t[t] += 1
                if row_succ[pick].item():
                    top1_correct_by_t[t] += 1

    val_loss = float(np.mean(losses)) if losses else float("nan")
    auroc_by_t: dict[int, float | None] = {
        t: _safe_auroc(scores_by_t[t], labels_by_t[t])
        for t in range(cfg.auroc_t_max + 1)
    }
    top1_by_t: dict[int, float | None] = {
        t: (top1_correct_by_t[t] / top1_total_by_t[t]) if top1_total_by_t[t] else None
        for t in range(cfg.auroc_t_max + 1)
    }
    per_t_count = {t: top1_total_by_t[t] for t in range(cfg.auroc_t_max + 1)}
    return EvalReport(
        val_loss=val_loss,
        auroc_by_t=auroc_by_t,
        top1_by_t=top1_by_t,
        per_t_count=per_t_count,
    )


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def train(
    cfg: TrainingConfig,
    train_dir: Path,
    val_dir: Path,
    vocab: Vocab,
    type_aug_policy: dict[str, bool] | None,
    out_dir: Path,
    prior: BasePrior | None = None,
    device: Optional[torch.device | str] = None,
) -> Path:
    """Run a SPECTRE training session and return the path of ``best.pt``.

    ``out_dir`` will gain ``best.pt``, ``last.pt``, ``log.jsonl``, and
    ``model_meta.json``.
    """
    if prior is None:
        prior = ZeroPrior()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if not isinstance(device, torch.device) else device

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    log_handle = open(log_path, "a", encoding="utf-8")

    _set_seed(cfg.seed)

    f_cfg = _build_f_sampling_config(cfg)
    train_dataset = SpectreDataset(
        split_dir=train_dir,
        prior=prior,
        seed=cfg.seed,
        f_sampling=f_cfg,
        augment=cfg.augment,
        type_aug_policy=type_aug_policy,
        num_f_samples_per_epoch=cfg.num_f_samples_per_epoch,
    )
    val_dataset = SpectreDataset(
        split_dir=val_dir,
        prior=prior,
        seed=cfg.seed + 10_000,  # different stream from train
        f_sampling=f_cfg,
        augment=False,
        type_aug_policy=type_aug_policy,
        num_f_samples_per_epoch=cfg.num_f_samples_per_val_episode,
    )

    model = SpectreModel(vocab, prior_dropout_p=cfg.prior_dropout_p).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate_with_vocab(vocab),
        num_workers=cfg.num_workers,
    )
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = _cosine_with_warmup(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
        base_lr=cfg.lr,
        min_lr=cfg.lr_min,
    )

    # Snapshot training config + vocab metadata so a downstream eval driver
    # can verify it's loading a checkpoint trained against the right vocab.
    meta = {
        "config": asdict(cfg),
        "vocab_config_hash": vocab.config_hash,
        "type_aug_policy": dict(type_aug_policy or {}),
        "num_episodes_train": train_dataset.num_episodes,
        "num_episodes_val": val_dataset.num_episodes,
        "num_train_examples_per_epoch": len(train_dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    best_loss = float("inf")
    best_auroc3: float | None = None
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    global_step = 0
    for epoch in range(cfg.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = plackett_luce_loss(logits, batch.r_success_mask, batch.r_mask)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
            global_step += 1

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        report = _evaluate(model, val_dataset, vocab, cfg, device)

        log_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": report.val_loss,
            "lr": scheduler.get_last_lr()[0],
            "auroc": {str(k): v for k, v in report.auroc_by_t.items()},
            "top1": {str(k): v for k, v in report.top1_by_t.items()},
            "per_t_count": {str(k): v for k, v in report.per_t_count.items()},
        }
        log_handle.write(json.dumps(log_record) + "\n")
        log_handle.flush()
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f}"
            f" val_loss={report.val_loss:.4f}"
            f" auroc0={report.auroc_by_t.get(0)}"
            f" auroc3={report.auroc_by_t.get(3)}"
        )

        # Save last; checkpoint best by (val_loss, auroc3 tie-break).
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "vocab_config_hash": vocab.config_hash,
            },
            last_path,
        )
        cur_auroc3 = report.auroc_by_t.get(3)
        is_better = report.val_loss < best_loss
        if (
            (not is_better)
            and math.isclose(report.val_loss, best_loss)
            and cur_auroc3 is not None
            and (best_auroc3 is None or cur_auroc3 > best_auroc3)
        ):
            is_better = True
        if is_better:
            best_loss = report.val_loss
            best_auroc3 = cur_auroc3
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "vocab_config_hash": vocab.config_hash,
                },
                best_path,
            )

    log_handle.close()
    return best_path
