"""Training loop for SPECTRE v3.

Same objective as v2.2 -- listwise Plackett-Luce over the pool, plus the same loss
restricted to within plan-length buckets so length cannot be used as a shortcut. What
changes is the evidence pathway (v3 failure-record tokens instead of five bespoke fact
types) and the checkpoint selector.

**Selection is deployed-val-FP, not `relrank`.** v2.2 selected on a
difficulty-normalized rank statistic that turned out to be miscalibrated on dd2d_v3
(never below 1, i.e. never better than random) and could pick an underfit epoch. v3
selects on the quantity actually reported: mean failed attempts before the first
success, from the real deployed rollout on val -- model scores plus sound demotion.
Three guards, each from a specific failure:

- the **rule used for selection is frozen** at ``permissive`` regardless of the mode a
  gate deploys with, so a change to the demotion rule cannot silently move the selector
  underneath a comparison;
- selection uses a **3-epoch moving average**, because a single val pass is noisy and
  ``argmin`` over 30 epochs is a maximization-biased estimator;
- selection is **uncensored and over the whole val split**, because a budget is a
  ceiling on the statistic and the models differ in the tail above it. G6 shipped with
  the selector censored at 30 attempts over a 50-episode subsample: it scored v2.2 at
  11.12 and v3 at 11.40 -- indistinguishable -- while the same two models were 4+ FP
  apart uncensored on test, because s2/s3 episodes routinely need 30-40+ attempts and
  every one was clipped to the same number. A selector blind to the region where models
  differ ranks epochs by noise. Cheaper recoveries if this ever costs too much: run it
  every K epochs, or uncensored on a stride -- never censored below the separating tail.

The aux head is *not* trained: no collection populates ``aux_labels``, so v2.2's masked
BCE contributed exactly zero, and necessity conditioning was cut from v3
(``decisions.md`` 2026-07-26). Pretending to train it would be theatre.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alphatamp.approaches.spectre.dataset_v3 import (
    build_v3_example,
    collate_v3,
    sample_context,
)
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.inference_v3 import deployed_rollout_v3_traced
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.loss import plackett_luce_loss, within_length_pl_loss
from alphatamp.approaches.spectre.model_v3 import SpectreV3Model, V3Config
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class TrainV3Config:
    """Hyperparameters for :func:`train_v3`, persisted into every checkpoint."""

    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 8
    max_tags: int = 32
    dropout_p: float = 0.1
    augment: bool = True
    seed: int = 0
    warmup_epochs: int = 2
    within_length_weight: float = 1.0
    use_overlap: bool = True
    use_records: bool = True
    select_window: int = 3
    # Uncensored, whole-split selection. `select_budget=None` means "run to the pool
    # cap", the same convention reporting uses (`decisions.md` 2026-06-07). See module
    # docstring for why censoring here was silently fatal.
    val_episodes: int = 100
    select_budget: Optional[int] = None
    num_workers: int = 4
    # "both" | "jaccard" | "dead" | "none" -- which cand_overlap columns the net sees.
    # Dropping `dead` is C5 hygiene (the sound rule stays outside the net as demotion);
    # see the note in `dataset_v3.build_v3_example`.
    overlap_mode: str = "both"
    # >0 switches on rollout-aligned |F| sampling out to this size. v2.2's inherited cap
    # of 8 never shows the model the |F| ~ 20-40 regime an s3 rollout actually spends
    # most of its attempts in.
    tail_max_f: int = 0
    sinusoidal_pos: bool = False
    # Collapse a candidate's failures to one record per (schema, args). The refiner
    # emits one per failed *sample*, which lets one unlucky candidate contribute
    # hundreds of tokens; §6.1 defines a record per failing *query*.
    aggregate_records: bool = False
    # Per-object evidence summary on scene tokens (SceneEncoderV3).
    use_obj_evidence: bool = False
    # G9: restrict the *training* split to these strata (empty = all). This is experiment
    # design, not a model input -- C2 bans stratum as an input or a test-time gate, and
    # this is neither: it decides which episodes exist during training, exactly as the
    # proposal's "train s0-s2, deploy s3" protocol (§7.4 A4) requires.
    train_strata: tuple[int, ...] = ()


class SpectreV3Dataset(Dataset):
    """Episodes -> ``(_V2Example, record arrays)``.

    Loads **raw** and lets ``build_v3_example`` canonicalize once. That is not
    incidental: ``canonicalize_episode`` is not idempotent, and feeding it an
    already-canonical episode silently changes the object->tag binding (the bug that
    skewed every cached comparison number until 2026-07-26).
    """

    def __init__(
        self,
        split_dir: Path,
        vocab: Vocab,
        cfg: TrainV3Config,
        spec: Optional[DomainSpec] = None,
    ) -> None:
        self.vocab = vocab
        self.cfg = cfg
        self.spec = spec
        self.epoch = 0
        self._paths = [
            p
            for p in list_episodes(split_dir)
            if _keep(load_episode(p), cfg.train_strata)
        ]

    @property
    def paths(self) -> list[Path]:
        """Episode paths, in split order.

        Public because the selector needs to *stride* this list rather than take a
        prefix: the collector fills strata in seed bands, so a prefix is the easy half.
        """
        return self._paths

    def set_epoch(self, epoch: int) -> None:
        """Reseed the per-epoch F-subset sampling, so each epoch draws new contexts."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        episode = load_episode(self._paths[idx])
        spec = self.spec or spec_for(episode.provenance.env_variant)
        rng = np.random.default_rng((self.cfg.seed, idx, self.epoch))
        fail_idx = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        ctx, hide = sample_context(fail_idx, rng, tail_max_f=self.cfg.tail_max_f)
        example, records = build_v3_example(
            episode,
            self.vocab,
            rng=rng,
            max_tags=self.cfg.max_tags,
            evidence=True,
            context_f=ctx,
            hide_facts=hide,
            augment_tags=self.cfg.augment,
            spec=spec,
            overlap_mode=self.cfg.overlap_mode,
            aggregate_records=self.cfg.aggregate_records,
        )
        if not self.cfg.use_records:
            records = []
        return example, records


def _trainable(ep) -> bool:
    return (
        ep.scene_geometry is not None
        and ep.summary.num_success >= 1
        and len(ep.skeleton_pool) >= 2
    )


def _keep(ep, strata: tuple[int, ...]) -> bool:
    """``_trainable`` plus the optional G9 stratum restriction on the training split."""
    if not _trainable(ep):
        return False
    if not strata:
        return True
    from alphatamp.approaches.spectre.dd2d_compare import stratum_of

    return stratum_of(int(ep.provenance.problem_id)) in strata


def _make_collate(max_arity: int):
    def _collate(items):
        examples = [e for e, _ in items]
        records = [r for _, r in items]
        return collate_v3(examples, max_arity=max_arity, records=records)

    return _collate


@torch.no_grad()
def deployed_val_fp(
    model: SpectreV3Model,
    episodes: list,
    vocab: Vocab,
    device: str,
    spec: DomainSpec,
    max_tags: int,
    budget: Optional[int] = None,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
) -> float:
    """Mean failed attempts before first success, on the real deployed loop.

    The demotion rule is pinned to ``permissive`` so the selector measures the *model*,
    not whichever rule a gate happens to deploy with. ``budget=None`` runs to the pool
    cap, i.e. uncensored -- the same convention reporting uses, and the only setting
    under which this statistic can see the s2/s3 tail where models actually differ.
    """
    model.eval()
    fps = []
    for ep in episodes:
        attempts, _ = deployed_rollout_v3_traced(
            model,
            ep,
            vocab,
            device,
            spec=spec,
            max_tags=max_tags,
            mode="permissive",
            max_attempts=budget,
            overlap_mode=overlap_mode,
            aggregate_records=aggregate_records,
        )
        fps.append(float(attempts) - 1.0)
    return float(np.mean(fps)) if fps else float("inf")


def _run_epoch(model, loader, device, wl_weight: float, opt=None) -> float:
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logits, _ = model(batch)
            loss = plackett_luce_loss(logits, batch.success_mask, batch.pool_mask)
            if train and wl_weight and batch.cand_prior is not None:
                loss = loss + wl_weight * within_length_pl_loss(
                    logits,
                    batch.success_mask,
                    batch.pool_mask,
                    batch.cand_prior[:, :, 1],
                )
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def train_v3(
    cfg: TrainV3Config,
    train_dir: Path,
    val_dir: Path,
    vocab: Vocab,
    out_dir: Path,
    device: Optional[str] = None,
) -> dict:
    """Train one v3 ranker, writing ``best.pt`` and ``log.jsonl`` under ``out_dir``.

    Returns the run summary (best selection score, epochs, training-set size).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = load_episode(list_episodes(train_dir)[0])
    spec = spec_for(probe.provenance.env_variant)

    train_ds = SpectreV3Dataset(train_dir, vocab, cfg, spec)
    val_ds = SpectreV3Dataset(val_dir, vocab, cfg, spec)
    collate = _make_collate(vocab.max_operator_arity)
    # Tensorization is ~79% of a training step (measured) and the model is tiny, so the
    # loader is the bottleneck, not the GPU. `persistent_workers` stays off on purpose:
    # workers are re-forked each epoch and so pick up `set_epoch`, which drives both the
    # tag permutation and the failure-context sampling.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )
    # Stride, never truncate. The collector fills strata in seed bands and episodes are
    # stored in seed order, so `[:50]` would hand the selector only strata 0-1 -- the
    # easy half -- and it would happily pick a checkpoint hopeless on s2/s3.
    _val_paths = val_ds.paths
    _stride = max(1, len(_val_paths) // max(cfg.val_episodes, 1))
    val_episodes = [load_episode(p) for p in _val_paths[::_stride]][: cfg.val_episodes]

    model = SpectreV3Model(
        n_ops=len(vocab.operators),
        max_arity=vocab.max_operator_arity,
        cfg=V3Config(
            n_overlap_feats=2 if cfg.use_overlap else 0,
            n_prior_feats=0,
            max_tags=cfg.max_tags,
            dropout_p=cfg.dropout_p,
            use_records=cfg.use_records,
            sinusoidal_pos=cfg.sinusoidal_pos,
            use_obj_evidence=cfg.use_obj_evidence,
        ),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    print(
        f"[train_v3] seed={cfg.seed} device={device} n_train={len(train_ds)} "
        f"n_val={len(val_ds)} epochs={cfg.epochs} records={cfg.use_records} "
        f"overlap={cfg.overlap_mode if cfg.use_overlap else 'off'} "
        f"tail_max_f={cfg.tail_max_f or 'off'} "
        f"selection=deployed-val-FP(ma{cfg.select_window}, "
        f"n={len(val_episodes)}, "
        f"budget={cfg.select_budget if cfg.select_budget else 'uncensored'})",
        flush=True,
    )

    log: list[dict] = []
    best = float("inf")
    t0 = time.time()
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = _lr_at(epoch, cfg)
        tr = _run_epoch(model, train_loader, device, cfg.within_length_weight, opt)
        va = _run_epoch(model, val_loader, device, 0.0, None)
        fp = deployed_val_fp(
            model,
            val_episodes,
            vocab,
            device,
            spec,
            cfg.max_tags,
            cfg.select_budget,
            cfg.overlap_mode,
            cfg.aggregate_records,
        )
        log.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "val_fp": fp})
        # moving average: a single 100-episode val pass is noisy, and argmin over 30
        # epochs would systematically pick the luckiest one rather than the best model
        window = [r["val_fp"] for r in log[-cfg.select_window :]]
        smoothed = float(np.mean(window))
        improved = smoothed < best
        if improved:
            best = smoothed
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "n_ops": len(vocab.operators),
                },
                out_dir / "best.pt",
            )
        if epoch == 0 or epoch == cfg.epochs - 1 or (epoch + 1) % 5 == 0:
            per = (time.time() - t0) / (epoch + 1)
            print(
                f"[train_v3] seed={cfg.seed} epoch {epoch + 1}/{cfg.epochs} "
                f"train={tr:.4f} val={va:.4f} val_fp={fp:.2f} ma={smoothed:.2f} "
                f"best={best:.2f}{' *' if improved else ''} | "
                f"{per:.1f}s/ep ETA {per * (cfg.epochs - epoch - 1) / 60:.1f}m",
                flush=True,
            )
    (out_dir / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    return {"best_val_fp": best, "epochs": len(log), "n_train": len(train_ds)}


def _lr_at(epoch: int, cfg: TrainV3Config) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(argv=None) -> int:
    """CLI entry point; see the module docstring for the selection protocol."""
    ap = argparse.ArgumentParser(description="Train the SPECTRE v3 ranker")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=TrainV3Config.lr)
    ap.add_argument("--wl-weight", type=float, default=1.0)
    ap.add_argument("--no-records", action="store_true", help="ablate record tokens")
    ap.add_argument("--no-overlap", action="store_true", help="ablate [dead, jaccard]")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--val-episodes",
        type=int,
        default=TrainV3Config.val_episodes,
        help="val episodes used by the selector (strided, never truncated)",
    )
    ap.add_argument(
        "--select-budget",
        type=int,
        default=None,
        help="censor the selector's rollout at N attempts; omit for uncensored",
    )
    ap.add_argument(
        "--overlap-mode",
        default="both",
        choices=["both", "jaccard", "dead", "none"],
        help="which cand_overlap columns the net sees; the sound rule is applied "
        "outside the net as demotion regardless",
    )
    ap.add_argument(
        "--tail-max-f",
        type=int,
        default=0,
        help="rollout-aligned |F| sampling out to this size (0 = v2.2's cap of 8)",
    )
    ap.add_argument(
        "--obj-evidence",
        action="store_true",
        help="summarise failures onto scene tokens via the tag join (SceneEncoderV3)",
    )
    ap.add_argument(
        "--aggregate-records",
        action="store_true",
        help="one record token per (schema, args) instead of per failed sample",
    )
    ap.add_argument(
        "--sinusoidal-pos",
        action="store_true",
        help="sinusoidal step positions (G9); retires the D-8 equivalence oracle",
    )
    ap.add_argument(
        "--train-strata",
        type=int,
        nargs="*",
        default=[],
        help="restrict the TRAINING split to these strata, e.g. --train-strata 0 1 2",
    )
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args(argv)

    root = Path(a.data_root)
    vocab = Vocab.from_json(root / "derived" / a.env / "train_vocab.json")
    cfg = TrainV3Config(
        epochs=a.epochs,
        seed=a.seed,
        lr=a.lr,
        within_length_weight=a.wl_weight,
        use_records=not a.no_records,
        use_overlap=not a.no_overlap,
        num_workers=a.num_workers,
        val_episodes=a.val_episodes,
        select_budget=a.select_budget,
        overlap_mode=a.overlap_mode,
        tail_max_f=a.tail_max_f,
        aggregate_records=a.aggregate_records,
        use_obj_evidence=a.obj_evidence,
        sinusoidal_pos=a.sinusoidal_pos,
        train_strata=tuple(a.train_strata),
    )
    sub = "checkpoints_v3"
    if a.no_records:
        sub += "_norec"
    if a.no_overlap:
        sub += "_noov"
    sub += a.out_suffix
    out = root / sub / a.env / f"seed_{a.seed}"
    res = train_v3(
        cfg, root / "raw" / a.env / "train", root / "raw" / a.env / "val", vocab, out
    )
    print(f"train_v3 done: {res} -> {out}/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
