"""Training loop for the v2.2 geometry-aware static model (Step 9).

Additive to v1's ``train.py``. Plackett–Luce loss over the pool (v1's hardest-won correct
choice) + a small auxiliary BCE on ``necessary``/``relevant`` (ignored where the target is
-1, i.e. no aux labels). Checkpoint by val PL loss for the static model; a rollout-based
selection can be layered on later (the static-model ladder gate in §9 is score-based).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from alphatamp.approaches.spectre.dataset_v2 import SpectreV2Dataset, make_collate
from alphatamp.approaches.spectre.loss import plackett_luce_loss
from alphatamp.approaches.spectre.model_v2 import SpectreV2Model
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class TrainV2Config:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 8
    max_tags: int = 32
    aux_weight: float = 0.2
    dropout_p: float = 0.1
    augment: bool = True
    exclude_marginal: bool = False
    seed: int = 0
    warmup_epochs: int = 2


def _aux_loss(aux_logits, aux_nec, aux_rel, obj_mask) -> torch.Tensor:
    """Masked BCE on necessary/relevant, ignoring -1 (no-label) and padding."""
    bce = nn.functional.binary_cross_entropy_with_logits
    tot = aux_logits.new_zeros(())
    for i, tgt in enumerate((aux_nec, aux_rel)):
        m = obj_mask & (tgt >= 0.0)
        if m.any():
            tot = tot + bce(aux_logits[..., i][m], tgt[m])
    return tot


def _pl_over_batch(logits, success_mask, pool_mask) -> torch.Tensor:
    """PL loss averaged over rows that have >= 1 success (others carry no signal)."""
    rows = success_mask.any(dim=1)
    if not rows.any():
        return logits.new_zeros(())
    return plackett_luce_loss(logits[rows], success_mask[rows], pool_mask[rows])


def _run_epoch(model, loader, device, aux_weight: float, opt=None) -> float:
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logits, aux = model(batch)
            loss = _pl_over_batch(logits, batch.success_mask, batch.pool_mask)
            if train:
                loss = loss + aux_weight * _aux_loss(
                    aux, batch.aux_necessary, batch.aux_relevant, batch.obj_mask
                )
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def train_v2(
    cfg: TrainV2Config,
    train_dir: Path,
    val_dir: Path,
    vocab: Vocab,
    out_dir: Path,
    device: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds = SpectreV2Dataset(
        train_dir,
        vocab,
        cfg.max_tags,
        augment=cfg.augment,
        seed=cfg.seed,
        exclude_marginal=cfg.exclude_marginal,
    )
    val_ds = SpectreV2Dataset(
        val_dir, vocab, cfg.max_tags, augment=False, seed=cfg.seed
    )
    collate = make_collate(vocab.max_operator_arity)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate)

    n_ops = len(vocab.operators)
    model = SpectreV2Model(
        n_ops=n_ops,
        max_arity=vocab.max_operator_arity,
        max_tags=cfg.max_tags,
        dropout_p=cfg.dropout_p,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_at(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(cfg.warmup_epochs, 1)
        prog = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    log = []
    for epoch in range(cfg.epochs):
        for grp in opt.param_groups:
            grp["lr"] = cfg.lr * lr_at(epoch)
        train_ds.set_epoch(epoch)
        tr = _run_epoch(model, train_loader, device, cfg.aux_weight, opt)
        va = _run_epoch(model, val_loader, device, cfg.aux_weight, None)
        log.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val:
            best_val = va
            torch.save(
                {"state_dict": model.state_dict(), "cfg": asdict(cfg), "n_ops": n_ops},
                out_dir / "best.pt",
            )
    (out_dir / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    return {"best_val_loss": best_val, "epochs": len(log), "n_train": len(train_ds)}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train the v2.2 geometry-aware model")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    vocab = Vocab.from_json(root / "derived" / a.env / "train_vocab.json")
    cfg = TrainV2Config(epochs=a.epochs, seed=a.seed)
    out = root / "checkpoints_v2" / a.env / f"seed_{a.seed}"
    res = train_v2(
        cfg,
        root / "raw" / a.env / "train",
        root / "raw" / a.env / "val",
        vocab,
        out,
    )
    print(f"train_v2 done: {res} -> {out}/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
