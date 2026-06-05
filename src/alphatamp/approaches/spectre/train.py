"""Training loop for the SPECTRE model on RT2D (and any kinder env).

Implements ``docs/archive/SPECTRE_RT2D_METHOD_SPEC.md`` §8: AdamW + cosine LR with linear
warmup, gradient clipping, F-subsample multiplier, prior dropout, per-epoch
validation with PL loss + AUROC(t) for ``t ∈ {0, 1, 2, 3}``.

Public surface:

- :class:`TrainingConfig` — all hyperparameters in one frozen-ish dataclass.
- :func:`train` — runs the full loop, writes ``best.pt`` + ``log.jsonl`` +
  ``model_meta.json`` under ``out_dir``.

The Hydra entrypoint at ``experiments/spectre/spectre_train.py`` constructs a
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
from alphatamp.approaches.spectre.priors import BasePrior, make_prior
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
    # ``prior_type``: which BasePrior subclass to use for π(s).
    #   "zero" — ZeroPrior (the original baseline; π ≡ 0).
    #   "heuristic" — HeuristicPrior (per-episode z-score of the negated
    #     pyperplan FF trajectory cost; RT2D-only). Same FF score the new
    #     B2 baseline uses, surfaced into the model so σ has a warm start.
    # Recorded in ``model_meta.json`` so eval/inference can reconstruct
    # the matching prior at test time.
    prior_type: str = "zero"
    # Per-module dropout (attention, FFN, transformer, scorer MLP).
    # Threaded into every nn.Dropout / MultiheadAttention dropout in
    # ``model.py``. Default 0.1 matches the original spec; bump to
    # 0.2–0.3 to fight overfitting (Tier 1 of the latest plan).
    dropout_p: float = 0.1

    # Architecture toggles
    # ``use_atom_sab2``: include the 2nd SAB in Φ_s atom-pool. Default True
    # preserves current behavior; set False to ablate against the 1-SAB
    # baseline (spec §4.3 original).
    use_atom_sab2: bool = True

    # ``use_static_tag_pool``: F3-B-(1) predicate-type-conditioned pooling.
    # When True, atoms whose predicate-name is in
    # ``env_registry.get_static_tag_predicates(env_variant)`` are routed
    # through a dedicated SAB+PMA stream that does not compete with the
    # fluent atom pool. Caller must pass ``static_tag_predicates`` to
    # :func:`train`; if absent the model silently falls back to the
    # single-pool path.
    use_static_tag_pool: bool = False

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

    # Early stopping
    # Stop when the configured ``checkpoint_metric`` has not improved for
    # ``early_stop_patience`` epochs, provided we have already trained at
    # least ``early_stop_min_epochs``. Set ``early_stop_patience = 0`` to
    # disable.
    early_stop_patience: int = 0
    early_stop_min_epochs: int = 5

    # Per-epoch deployment-style rollout evaluation. When enabled the
    # trainer runs ``eda.spectre_evaluate`` on the full train and val
    # splits each epoch and logs ``train/val_rollout_attempts`` plus
    # standard deviation and censoring rate. ~10s per epoch overhead on
    # CPU; required when ``checkpoint_metric == "val_rollout_attempts"``.
    rollout_eval_each_epoch: bool = True

    # ``checkpoint_metric``: how to pick ``best.pt``.
    #   "val_rollout_attempts": min mean-attempts on val rollout (lower is
    #     better), with val_loss as tiebreak (lower is better). Most
    #     directly tracks the deployment metric.
    #   "val_loss": legacy — min val PL loss with AUROC(3) tiebreak (higher
    #     auroc3 is better).
    # Early stopping uses the same metric.
    checkpoint_metric: str = "val_rollout_attempts"

    # ``rollout_attempt_budget``: attempt budget used inside the per-epoch
    # rollout eval. Mirrors the test-time budget the EDA notebook uses
    # so train-time and reported numbers are directly comparable.
    rollout_attempt_budget: int = 20


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


def _accumulate_auroc_buckets(
    logits: torch.Tensor,
    batch: SpectreBatch,
    t_max: int,
    scores_by_t: dict[int, list[float]],
    labels_by_t: dict[int, list[int]],
    top1_correct_by_t: dict[int, int],
    top1_total_by_t: dict[int, int],
) -> None:
    """Stratify-by-|F| accumulator for AUROC(t) + top-1 hit rate.

    Mutates the four passed-in dicts in place. Shared between train and val loops so on-
    the-fly train AUROC matches the validation definition.
    """
    f_sizes = batch.f_mask.sum(dim=-1).cpu().tolist()
    r_mask = batch.r_mask.cpu()
    r_succ = batch.r_success_mask.cpu()
    logits_cpu = logits.detach().cpu()
    for ex_idx, t in enumerate(f_sizes):
        t = int(t)
        if t > t_max:
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


def _finalize_auroc(
    scores_by_t: dict[int, list[float]],
    labels_by_t: dict[int, list[int]],
    top1_correct_by_t: dict[int, int],
    top1_total_by_t: dict[int, int],
    t_max: int,
) -> tuple[dict[int, float | None], dict[int, float | None], dict[int, int]]:
    auroc_by_t: dict[int, float | None] = {
        t: _safe_auroc(scores_by_t[t], labels_by_t[t]) for t in range(t_max + 1)
    }
    top1_by_t: dict[int, float | None] = {
        t: (top1_correct_by_t[t] / top1_total_by_t[t]) if top1_total_by_t[t] else None
        for t in range(t_max + 1)
    }
    per_t_count = {t: top1_total_by_t[t] for t in range(t_max + 1)}
    return auroc_by_t, top1_by_t, per_t_count


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
            _accumulate_auroc_buckets(
                logits,
                batch,
                cfg.auroc_t_max,
                scores_by_t,
                labels_by_t,
                top1_correct_by_t,
                top1_total_by_t,
            )

    val_loss = float(np.mean(losses)) if losses else float("nan")
    auroc_by_t, top1_by_t, per_t_count = _finalize_auroc(
        scores_by_t,
        labels_by_t,
        top1_correct_by_t,
        top1_total_by_t,
        cfg.auroc_t_max,
    )
    return EvalReport(
        val_loss=val_loss,
        auroc_by_t=auroc_by_t,
        top1_by_t=top1_by_t,
        per_t_count=per_t_count,
    )


# ---------------------------------------------------------------------------
# Per-epoch deployment-style rollout eval (test-time attempt loop on val/train)
# ---------------------------------------------------------------------------


@dataclass
class RolloutSummary:
    """Mean / std / censoring rate over per-episode attempts.

    Mirrors the columns the EDA notebook reports for B1–B5 + SPECTRE so the per-epoch
    console output is directly comparable.
    """

    mean_attempts: float
    std_attempts: float
    censoring_rate: float
    n_episodes: int

    def to_dict(self) -> dict[str, float | int]:
        """Return the summary as a JSON-serializable dict (for log.jsonl)."""
        return {
            "mean_attempts": self.mean_attempts,
            "std_attempts": self.std_attempts,
            "censoring_rate": self.censoring_rate,
            "n_episodes": self.n_episodes,
        }


def _summarize_rollout(
    arr_attempts: np.ndarray, arr_censored: np.ndarray
) -> RolloutSummary:
    if arr_attempts.size == 0:
        return RolloutSummary(
            mean_attempts=float("nan"),
            std_attempts=float("nan"),
            censoring_rate=float("nan"),
            n_episodes=0,
        )
    return RolloutSummary(
        mean_attempts=float(arr_attempts.mean()),
        std_attempts=float(arr_attempts.std()),
        censoring_rate=float(arr_censored.mean()),
        n_episodes=int(arr_attempts.size),
    )


def _checkpoint_metric_tuple(
    cfg: TrainingConfig,
    val_loss: float,
    val_auroc3: float | None,
    val_rollout_attempts: float | None,
) -> tuple[float, float]:
    """Return a tuple whose lexicographic min is the "best" checkpoint.

    Lex-min semantics let us encode any (primary, tiebreak) ordering with "lower is
    better" by negating quantities where higher is better.
    """
    if cfg.checkpoint_metric == "val_rollout_attempts":
        if val_rollout_attempts is None or math.isnan(val_rollout_attempts):
            # Defensive: fall back to val_loss if rollout produced no
            # measurement (e.g., no trainable val episodes).
            return (float("inf"), val_loss)
        return (val_rollout_attempts, val_loss)
    if cfg.checkpoint_metric == "val_loss":
        # Primary: min val_loss. Tiebreak: max auroc3 → use -auroc3.
        tiebreak = -val_auroc3 if val_auroc3 is not None else 0.0
        return (val_loss, tiebreak)
    raise ValueError(
        f"Unknown checkpoint_metric={cfg.checkpoint_metric!r};"
        " expected one of {'val_rollout_attempts', 'val_loss'}"
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
    static_tag_predicates: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Run a SPECTRE training session and return the path of ``best.pt``.

    ``out_dir`` will gain ``best.pt``, ``last.pt``, ``log.jsonl``, and
    ``model_meta.json``.
    """
    if prior is None:
        prior = make_prior(cfg.prior_type)
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

    # Resolve the static-tag predicate list: only honored when
    # ``cfg.use_static_tag_pool`` is set; otherwise pass None so the model
    # uses the single-pool path even if a list was supplied by the caller.
    resolved_static_tags = (
        list(static_tag_predicates)
        if (cfg.use_static_tag_pool and static_tag_predicates)
        else None
    )
    model = SpectreModel(
        vocab,
        prior_dropout_p=cfg.prior_dropout_p,
        use_atom_sab2=cfg.use_atom_sab2,
        static_tag_predicates=resolved_static_tags,
        dropout_p=cfg.dropout_p,
    ).to(device)
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
        "static_tag_predicates": list(resolved_static_tags or []),
        "num_episodes_train": train_dataset.num_episodes,
        "num_episodes_val": val_dataset.num_episodes,
        "num_train_examples_per_epoch": len(train_dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Random-Φ + zero-init head baseline. The scorer's head.weight is
    # zero-initialized, so the untrained logit is identically 0 regardless of
    # prior_type — this should produce ~0.5 AUROC at every t (uniform
    # ranking), confirming the architecture initializes as the spec
    # describes. Anything materially off 0.5 means the init drifted.
    random_phi_model = SpectreModel(
        vocab,
        prior_dropout_p=cfg.prior_dropout_p,
        use_atom_sab2=cfg.use_atom_sab2,
        dropout_p=cfg.dropout_p,
    ).to(device)
    random_phi_report = _evaluate(random_phi_model, val_dataset, vocab, cfg, device)
    random_phi_baseline = {
        "val_loss": random_phi_report.val_loss,
        "auroc": {str(k): v for k, v in random_phi_report.auroc_by_t.items()},
        "top1": {str(k): v for k, v in random_phi_report.top1_by_t.items()},
        "per_t_count": {str(k): v for k, v in random_phi_report.per_t_count.items()},
    }
    del random_phi_model

    # Load LoadedSplit objects once for per-epoch deployment-style rollout
    # eval. Lazy import to avoid pulling eda's heavy video-rendering
    # imports into the training startup path. The rollout itself runs
    # ``inference.init_inference_state`` + ``select_next_skeleton`` per
    # episode — same code path the notebook uses for SPECTRE evaluation.
    rollout_train_split = None
    rollout_val_split = None
    if cfg.rollout_eval_each_epoch or cfg.checkpoint_metric == "val_rollout_attempts":
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre import eda as _eda

        print("Loading splits for per-epoch rollout eval...")
        rollout_train_split = _eda.load_split_episodes(train_dir)
        rollout_val_split = _eda.load_split_episodes(val_dir)
        print(
            f"  rollout: train={len(rollout_train_split.episodes)} eps,"
            f" val={len(rollout_val_split.episodes)} eps"
        )

    best_metric: tuple[float, float] = (float("inf"), float("inf"))
    epochs_since_improve = 0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    global_step = 0
    for epoch in range(cfg.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        # Train-side AUROC accumulators, mirroring _evaluate. Shapes drift
        # across the epoch as parameters update; this is a "smoothed"
        # in-loop snapshot used only to detect train-vs-val divergence.
        train_scores_by_t: dict[int, list[float]] = defaultdict(list)
        train_labels_by_t: dict[int, list[int]] = defaultdict(list)
        train_top1_correct: dict[int, int] = defaultdict(int)
        train_top1_total: dict[int, int] = defaultdict(int)
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
            _accumulate_auroc_buckets(
                logits,
                batch,
                cfg.auroc_t_max,
                train_scores_by_t,
                train_labels_by_t,
                train_top1_correct,
                train_top1_total,
            )

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        train_auroc_by_t, train_top1_by_t, train_per_t_count = _finalize_auroc(
            train_scores_by_t,
            train_labels_by_t,
            train_top1_correct,
            train_top1_total,
            cfg.auroc_t_max,
        )
        report = _evaluate(model, val_dataset, vocab, cfg, device)

        # Per-epoch deployment-style rollout eval on train + val. Same code
        # path the notebook uses for SPECTRE evaluation, so the numbers are
        # directly comparable to the EDA summary table. Train-vs-val gap is
        # the primary overfitting signal at the deployment metric.
        train_rollout_summary: RolloutSummary | None = None
        val_rollout_summary: RolloutSummary | None = None
        if rollout_train_split is not None and rollout_val_split is not None:
            # pylint: disable=import-outside-toplevel
            from alphatamp.approaches.spectre import eda as _eda

            train_rollout = _eda.spectre_evaluate(
                rollout_train_split,
                model,
                vocab,
                attempt_budget=cfg.rollout_attempt_budget,
                prior=prior,
                device=device,
                name="train_rollout",
            )
            val_rollout = _eda.spectre_evaluate(
                rollout_val_split,
                model,
                vocab,
                attempt_budget=cfg.rollout_attempt_budget,
                prior=prior,
                device=device,
                name="val_rollout",
            )
            train_rollout_summary = _summarize_rollout(
                train_rollout.attempts, train_rollout.censored
            )
            val_rollout_summary = _summarize_rollout(
                val_rollout.attempts, val_rollout.censored
            )

        log_record: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": report.val_loss,
            "lr": scheduler.get_last_lr()[0],
            "auroc": {str(k): v for k, v in report.auroc_by_t.items()},
            "top1": {str(k): v for k, v in report.top1_by_t.items()},
            "per_t_count": {str(k): v for k, v in report.per_t_count.items()},
            "train_auroc": {str(k): v for k, v in train_auroc_by_t.items()},
            "train_top1": {str(k): v for k, v in train_top1_by_t.items()},
            "train_per_t_count": {str(k): v for k, v in train_per_t_count.items()},
        }
        if train_rollout_summary is not None:
            log_record["train_rollout"] = train_rollout_summary.to_dict()
        if val_rollout_summary is not None:
            log_record["val_rollout"] = val_rollout_summary.to_dict()
        if epoch == 0:
            log_record["random_phi_baseline"] = random_phi_baseline
        log_handle.write(json.dumps(log_record) + "\n")
        log_handle.flush()

        rollout_str = ""
        if train_rollout_summary is not None and val_rollout_summary is not None:
            att_gap = (
                val_rollout_summary.mean_attempts - train_rollout_summary.mean_attempts
            )
            rollout_str = (
                f" train_att={train_rollout_summary.mean_attempts:.2f}±"
                f"{train_rollout_summary.std_attempts:.2f}"
                f" val_att={val_rollout_summary.mean_attempts:.2f}±"
                f"{val_rollout_summary.std_attempts:.2f}"
                f" gap={att_gap:+.2f}"
            )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f}"
            f" val_loss={report.val_loss:.4f}"
            f" auroc0={report.auroc_by_t.get(0)}"
            f" auroc3={report.auroc_by_t.get(3)}"
            f"{rollout_str}"
        )

        # Save last; pick best.pt via the configured metric.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "vocab_config_hash": vocab.config_hash,
                "static_tag_predicates": list(resolved_static_tags or []),
            },
            last_path,
        )
        val_rollout_attempts = (
            val_rollout_summary.mean_attempts if val_rollout_summary else None
        )
        cur_metric = _checkpoint_metric_tuple(
            cfg,
            val_loss=report.val_loss,
            val_auroc3=report.auroc_by_t.get(3),
            val_rollout_attempts=val_rollout_attempts,
        )
        is_better = cur_metric < best_metric
        if is_better:
            best_metric = cur_metric
            epochs_since_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "vocab_config_hash": vocab.config_hash,
                    "static_tag_predicates": list(resolved_static_tags or []),
                    "checkpoint_metric": cfg.checkpoint_metric,
                    "checkpoint_metric_value": list(cur_metric),
                },
                best_path,
            )
        else:
            epochs_since_improve += 1

        if (
            cfg.early_stop_patience > 0
            and epoch + 1 >= cfg.early_stop_min_epochs
            and epochs_since_improve >= cfg.early_stop_patience
        ):
            print(
                f"early stop: no {cfg.checkpoint_metric} improvement for"
                f" {epochs_since_improve} epochs (patience={cfg.early_stop_patience})"
            )
            break

    log_handle.close()
    return best_path
