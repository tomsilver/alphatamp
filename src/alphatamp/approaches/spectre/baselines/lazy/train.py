"""Behaviour-cloning training for the LAZY policy, with val rollout-FP selection.

Runnable as ``python -m alphatamp.approaches.spectre.baselines.lazy.train`` or via the
thin entry point ``experiments/spectre/lazy_train.py``. Selection metric is val
rollout-FP (the project arbiter), NOT the BC cross-entropy. The fitted feasibility prior
ϕ is saved *with* the checkpoint so the deployment rollout uses the exact train-fit ϕ.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from alphatamp.approaches.spectre.baselines.lazy.dataset import (
    build_bc_examples,
    load_structs,
)
from alphatamp.approaches.spectre.baselines.lazy.domain import make_lazy_domain
from alphatamp.approaches.spectre.baselines.lazy.eval import mean_rollout_fp
from alphatamp.approaches.spectre.baselines.lazy.feasibility import fit_phi
from alphatamp.approaches.spectre.baselines.lazy.graph import build_feature_spec
from alphatamp.approaches.spectre.baselines.lazy.model import AttentionPolicy, bc_loss
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[6]


def _default_out_dir(env_variant: str, seed: int) -> Path:
    return REPO / "data" / "spectre" / "checkpoints" / env_variant / f"lazy_s{seed}"


def train_lazy(
    env_variant: str,
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
    d: int = 64,
    heads: int = 4,
    dropout: float = 0.1,
    max_demos_per_episode: int = 16,
    keep_strata: set[int] | None = None,
    patience: int = 10,
    out_dir: Path | None = None,
    tiny: bool = False,
) -> dict:
    """Train one seed and save ``ckpt.pt``; returns the metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        device = "cpu"

    domain = make_lazy_domain(env_variant)
    vocab = Vocab.from_json(domain.vocab_path)
    spec = build_feature_spec(vocab)

    t0 = time.perf_counter()
    train_structs = load_structs(
        domain.split_dir("train"),
        vocab,
        spec,
        domain.frame_extent,
        domain.shape_max,
        keep_strata=keep_strata,
    )
    if tiny:
        # Stride across the split so the overfit gate spans strata (episodes are stored
        # in seed order = stratum bands; a prefix would be one trivial stratum).
        step = max(1, len(train_structs) // 20)
        train_structs = train_structs[::step][:20]
    val_structs = load_structs(
        domain.split_dir("val"),
        vocab,
        spec,
        domain.frame_extent,
        domain.shape_max,
        keep_strata=keep_strata,
    )
    if tiny:
        val_structs = train_structs  # overfit gate: val == train
    phi_prior = fit_phi(st.episode for st in train_structs)
    examples = build_bc_examples(
        train_structs, vocab, spec, max_demos_per_episode=max_demos_per_episode
    )
    print(
        f"[lazy {env_variant} s{seed}] train_eps={len(train_structs)} "
        f"val_eps={len(val_structs)} bc_examples={len(examples)} "
        f"phi_keys={len(phi_prior)} setup={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    loader = DataLoader(
        examples, batch_size=batch_size, shuffle=True, follow_batch=["act_op"]
    )

    model = AttentionPolicy(
        node_dim=spec.node_dim,
        edge_dim=spec.edge_dim,
        op_vocab=len(vocab.operators),
        max_arity=spec.max_arity,
        d=d,
        heads=heads,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_fp = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    since_improve = 0
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        nb = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = bc_loss(model, batch)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        val_fp = mean_rollout_fp(model, val_structs, vocab, spec, phi_prior, device)
        improved = val_fp < best_fp - 1e-9
        if improved:
            best_fp = val_fp
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            since_improve = 0
        else:
            since_improve += 1
        print(
            f"[lazy {env_variant} s{seed}] epoch {epoch:02d} "
            f"train_ce={tot/max(1,nb):.4f} val_fp={val_fp:.3f} "
            f"best={best_fp:.3f}@{best_epoch}",
            flush=True,
        )
        if since_improve >= patience:
            print(f"[lazy {env_variant} s{seed}] early stop @ {epoch}", flush=True)
            break

    out_dir = Path(out_dir) if out_dir else _default_out_dir(env_variant, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "node_dim": spec.node_dim,
        "edge_dim": spec.edge_dim,
        "op_vocab": len(vocab.operators),
        "max_arity": spec.max_arity,
        "d": d,
        "heads": heads,
        "dropout": dropout,
        "spec": asdict(spec),
        # ϕ prior is part of the deployed method; torch.save pickles the tuple keys.
        "phi_prior": phi_prior,
        "vocab_config_hash": vocab.config_hash,
        "env_variant": env_variant,
        "seed": seed,
        "metrics": {"val_fp": best_fp, "best_epoch": best_epoch},
    }
    torch.save(ckpt, out_dir / "ckpt.pt")
    (out_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "val_fp": best_fp,
                "best_epoch": best_epoch,
                "env": env_variant,
                "seed": seed,
                "bc_examples": len(examples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[lazy {env_variant} s{seed}] DONE best_val_fp={best_fp:.3f} -> "
        f"{out_dir/'ckpt.pt'}",
        flush=True,
    )
    return {"val_fp": best_fp, "best_epoch": best_epoch}


def _parse_strata(s: str | None) -> set[int] | None:
    if not s:
        return None
    return {int(x) for x in s.split(",") if x.strip() != ""}


def main() -> None:
    """CLI: train one LAZY seed for an env-variant and save ``ckpt.pt``."""
    ap = argparse.ArgumentParser(description="Train the LAZY baseline policy.")
    ap.add_argument("--env-variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-demos", type=int, default=16)
    ap.add_argument("--keep-strata", default=None, help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tiny", action="store_true", help="20-episode overfit gate")
    args = ap.parse_args()
    train_lazy(
        env_variant=args.env_variant,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_demos_per_episode=args.max_demos,
        keep_strata=_parse_strata(args.keep_strata),
        patience=args.patience,
        out_dir=args.out_dir,
        tiny=args.tiny,
    )


if __name__ == "__main__":
    main()
