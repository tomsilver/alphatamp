"""Training loop for the v2.2 geometry-aware static model (Step 9).

Additive to v1's ``train.py``. Plackett–Luce loss over the pool (v1's hardest-won correct
choice) + a small auxiliary BCE on ``necessary``/``relevant`` (ignored where the target is
-1, i.e. no aux labels). Checkpoint by val PL loss for the static model; a rollout-based
selection can be layered on later (the static-model ladder gate in §9 is score-based).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from alphatamp.approaches.spectre.dataset_v2 import (
    SpectreV2Dataset,
    build_v2_example,
    collate_v2,
    make_collate,
)
from alphatamp.approaches.spectre.evidence import scramble_gauge
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.loss import plackett_luce_loss
from alphatamp.approaches.spectre.model_v2 import SpectreV2Batch, SpectreV2Model
from alphatamp.approaches.spectre.vocab import Vocab


def _build_gauge_batch(
    val_dir: Path, vocab: Vocab, cfg: "TrainV2Config", n_eps: int = 24
) -> Optional[SpectreV2Batch]:
    """A fixed val batch of evidence examples with a nonempty failed context ``F`` (up
    to 3 fails/episode), for the scramble gauge.

    None if no val episode has a usable context.
    """
    exs = []
    for p in list_episodes(val_dir)[:n_eps]:
        ep = load_episode(p)
        if ep.scene_geometry is None:
            continue
        fails = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if not fails:
            continue
        exs.append(
            build_v2_example(
                ep,
                vocab,
                rng=None,
                max_tags=cfg.max_tags,
                evidence=True,
                context_f=frozenset(fails[:3]),
                augment_tags=False,
            )
        )
    if not exs:
        return None
    return collate_v2(exs, max_arity=vocab.max_operator_arity)


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
    evidence: bool = (
        False  # Step 11: train the typed-evidence pathway (F-context sampling)
    )


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
            loss.backward()  # type: ignore[no-untyped-call]
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
        evidence=cfg.evidence,
    )
    # Val loss (checkpoint selection) is the STATIC ranking quality at t=0 — the deployment
    # start the static pathway must own (P-D); the evidence use is tracked by the scramble
    # gauge, not the selection metric.
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

    gauge_batch = _build_gauge_batch(val_dir, vocab, cfg) if cfg.evidence else None
    gauge_rng = np.random.default_rng(cfg.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    log = []
    print(
        f"[train_v2] seed={cfg.seed} device={device} n_train={len(train_ds)} "
        f"n_val={len(val_ds)} epochs={cfg.epochs} batch={cfg.batch_size} "
        f"evidence={cfg.evidence}",
        flush=True,
    )
    t_start = time.time()
    for epoch in range(cfg.epochs):
        for grp in opt.param_groups:
            grp["lr"] = cfg.lr * lr_at(epoch)
        train_ds.set_epoch(epoch)
        tr = _run_epoch(model, train_loader, device, cfg.aux_weight, opt)
        va = _run_epoch(model, val_loader, device, cfg.aux_weight, None)
        gauge = (
            scramble_gauge(model, gauge_batch, device, gauge_rng)
            if gauge_batch is not None
            else 0.0
        )
        log.append(
            {"epoch": epoch, "train_loss": tr, "val_loss": va, "scramble_gauge": gauge}
        )
        improved = va < best_val
        if improved:
            best_val = va
            torch.save(
                {"state_dict": model.state_dict(), "cfg": asdict(cfg), "n_ops": n_ops},
                out_dir / "best.pt",
            )
        # Periodic heartbeat so a long run is never mistaken for a hang: every 5 epochs,
        # plus the first and last, with train/val loss, best-so-far, scramble gauge, ETA.
        if epoch == 0 or epoch == cfg.epochs - 1 or (epoch + 1) % 5 == 0:
            elapsed = time.time() - t_start
            per_epoch = elapsed / (epoch + 1)
            eta = per_epoch * (cfg.epochs - epoch - 1)
            gauge_str = f" gauge={gauge:.3f}" if cfg.evidence else ""
            print(
                f"[train_v2] seed={cfg.seed} epoch {epoch + 1}/{cfg.epochs} "
                f"train={tr:.4f} val={va:.4f} best={best_val:.4f}"
                f"{' *' if improved else ''}{gauge_str} | "
                f"{per_epoch:.1f}s/ep ETA {eta / 60:.1f}m",
                flush=True,
            )
    (out_dir / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    final_gauge = log[-1]["scramble_gauge"] if log else 0.0
    return {
        "best_val_loss": best_val,
        "epochs": len(log),
        "n_train": len(train_ds),
        "final_scramble_gauge": final_gauge,
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train the v2.2 geometry-aware model")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--evidence", action="store_true", help="train the Step-11 pathway")
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    vocab = Vocab.from_json(root / "derived" / a.env / "train_vocab.json")
    cfg = TrainV2Config(epochs=a.epochs, seed=a.seed, evidence=a.evidence)
    sub = "checkpoints_v2_evidence" if a.evidence else "checkpoints_v2"
    out = root / sub / a.env / f"seed_{a.seed}"
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
