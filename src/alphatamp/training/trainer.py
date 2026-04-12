"""Belief encoder training loop with rollout-consistent prefixes.

Orchestrates end-to-end training of the belief encoder pipeline:
SkeletonEncoder → TokenBuilder → BeliefEncoder → prediction heads,
using PrefixGenerator for rollout-consistent training data and a
DAgger-style schedule that mixes teacher-forced, epsilon-random,
and on-policy prefix generation.

Validation computes per-step NLL bucketed by history size to verify
that the belief encoder integrates history (monotonically decreasing NLL).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alphatamp.data.skeleton_dataset import SkeletonDataset
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.losses import PredictionNLLLoss
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder
from alphatamp.training.prefix_generator import PrefixGenerator, PrefixStep

__all__ = ["BeliefTrainer"]

logger = logging.getLogger(__name__)


def _pad_op_sequences(
    dataset: SkeletonDataset,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad variable-length op sequences into uniform tensors.

    Returns
    -------
    op_type_ids : (M, L_max) int
    obj_ids : (M, L_max, P_max) int
    lengths : (M,) int
    """
    M = dataset.M
    seqs = dataset.op_sequences

    L_max = max(s.length for s in seqs) if M > 0 else 0
    L_max = max(L_max, 1)  # avoid zero-dim tensors

    P_max = max(
        (s.obj_ids.shape[1] if s.length > 0 else 0) for s in seqs
    ) if M > 0 else 0
    P_max = max(P_max, 1)

    op_type_ids = torch.zeros(M, L_max, dtype=torch.int32)
    obj_ids = torch.full((M, L_max, P_max), -1, dtype=torch.int32)
    lengths = torch.zeros(M, dtype=torch.int64)

    for j, seq in enumerate(seqs):
        L_j = seq.length
        lengths[j] = L_j
        if L_j > 0:
            op_type_ids[j, :L_j] = seq.op_type_ids
            P_j = seq.obj_ids.shape[1]
            obj_ids[j, :L_j, :P_j] = seq.obj_ids

    return (
        op_type_ids.to(device),
        obj_ids.to(device),
        lengths.to(device),
    )


class BeliefTrainer:
    """Orchestrate training of the belief encoder pipeline.

    Parameters
    ----------
    skeleton_encoder, token_builder, belief_encoder:
        Model components.
    y_head, f_head, t_head, joint_y_head:
        Prediction heads.
    loss_fn:
        PredictionNLLLoss instance.
    train_dataset, val_dataset:
        HDF5-backed skeleton datasets (should be preloaded for speed).
    lr, weight_decay, num_epochs, batch_size, grad_clip_norm:
        Standard training hyperparameters.
    lr_min_fraction:
        Cosine schedule floor as fraction of initial lr.
    warmup_epochs:
        Epochs of epsilon-random warmup before DAgger mixing.
    epsilon:
        Random deviation probability for epsilon_random mode.
    device:
        Torch device for training.
    seed:
        Random seed for reproducibility.
    checkpoint_dir:
        Directory for saving checkpoints.
    log_every_steps:
        Print training loss every N batches.
    """

    def __init__(
        self,
        skeleton_encoder: SkeletonEncoder,
        token_builder: TokenBuilder,
        belief_encoder: BeliefEncoder,
        y_head: YHead,
        f_head: FHead,
        t_head: THead,
        joint_y_head: JointYHead,
        loss_fn: PredictionNLLLoss,
        train_dataset: SkeletonDataset,
        val_dataset: SkeletonDataset,
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 50,
        batch_size: int = 64,
        grad_clip_norm: float = 1.0,
        lr_min_fraction: float = 0.01,
        warmup_epochs: int = 5,
        epsilon: float = 0.1,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        checkpoint_dir: str = "checkpoints",
        log_every_steps: int = 20,
    ) -> None:
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.warmup_epochs = warmup_epochs
        self.epsilon = epsilon
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every_steps = log_every_steps

        # Store model components
        self.skeleton_encoder = skeleton_encoder.to(device)
        self.token_builder = token_builder.to(device)
        self.belief_encoder = belief_encoder.to(device)
        self.y_head = y_head.to(device)
        self.f_head = f_head.to(device)
        self.t_head = t_head.to(device)
        self.joint_y_head = joint_y_head.to(device)
        self.loss_fn = loss_fn.to(device)

        self._modules: list[nn.Module] = [
            self.skeleton_encoder, self.token_builder, self.belief_encoder,
            self.y_head, self.f_head, self.t_head, self.joint_y_head,
        ]

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimizer and scheduler
        all_params = []
        for m in self._modules:
            all_params.extend(m.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params, lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr * lr_min_fraction,
        )

        # Prepare skeleton inputs (padded op sequences, stored on device)
        self._prepare_skeleton_inputs()

    # ------------------------------------------------------------------
    # Skeleton input preparation
    # ------------------------------------------------------------------

    def _prepare_skeleton_inputs(self) -> None:
        """Pad op sequences and store on device. Called once at init."""
        self._op_type_ids, self._obj_ids, self._skel_lengths_enc = (
            _pad_op_sequences(self.train_dataset, self.device)
        )
        # Skeleton lengths for TokenBuilder (float, from dataset attribute)
        self._skel_lengths_tb = (
            self.train_dataset.skeleton_lengths.float().to(self.device)
        )  # (M,)

    # ------------------------------------------------------------------
    # Skeleton embedding computation
    # ------------------------------------------------------------------

    def _compute_skel_embeds(self, *, no_grad: bool = False) -> Tensor:
        """Compute (M, d_skel) skeleton embeddings.

        When no_grad=True, wraps in torch.no_grad() (for on-policy scoring).
        """
        if no_grad:
            with torch.no_grad():
                return self.skeleton_encoder(
                    self._op_type_ids, self._obj_ids, self._skel_lengths_enc,
                )
        return self.skeleton_encoder(
            self._op_type_ids, self._obj_ids, self._skel_lengths_enc,
        )

    # ------------------------------------------------------------------
    # Prefix step collation
    # ------------------------------------------------------------------

    @staticmethod
    def _collate_prefix_steps(steps: list[PrefixStep]) -> dict[str, Tensor]:
        """Stack a list of PrefixSteps into batched tensors.

        Returns dict with (B, M) tensors and (B,) step_indices.
        """
        return {
            "applicability": torch.stack([s.applicability for s in steps]),
            "revealed_mask": torch.stack([s.revealed_mask for s in steps]),
            "y": torch.stack([s.revealed_outcomes["y"] for s in steps]),
            "f": torch.stack([s.revealed_outcomes["f"] for s in steps]),
            "t": torch.stack([s.revealed_outcomes["t"] for s in steps]),
            "y_true": torch.stack([s.y_true for s in steps]),
            "f_true": torch.stack([s.f_true for s in steps]),
            "t_true": torch.stack([s.t_true for s in steps]),
            "lengths": torch.stack([s.lengths for s in steps]),
            "step_indices": torch.tensor(
                [s.step_index for s in steps], dtype=torch.int64,
            ),
        }

    # ------------------------------------------------------------------
    # Score function factory (for on_policy mode)
    # ------------------------------------------------------------------

    def _build_score_fn(self, skel_embeds: Tensor) -> Callable[[PrefixStep], Tensor]:
        """Return a scoring closure for on_policy PrefixGenerator.

        The closure runs a no-grad forward pass and returns
        -(t_dist.mean / sigmoid(y_logits).clamp(min=1e-6)) as (M,) scores
        so that argmax picks the best candidate per the index rule.
        """
        def score_fn(step: PrefixStep) -> Tensor:
            with torch.no_grad():
                # Unsqueeze to (1, M, ...) batch dim
                se = skel_embeds.unsqueeze(0)  # (1, M, d_skel)
                app = step.applicability.unsqueeze(0).to(self.device)
                rm = step.revealed_mask.unsqueeze(0).to(self.device)
                y = step.revealed_outcomes["y"].unsqueeze(0).to(self.device)
                f = step.revealed_outcomes["f"].unsqueeze(0).to(self.device)
                t = step.revealed_outcomes["t"].unsqueeze(0).to(self.device)
                lengths = step.lengths.unsqueeze(0).to(self.device)

                tokens = self.token_builder(se, app, rm, y, f, t, lengths)
                pad_mask = torch.zeros(1, tokens.shape[1], dtype=torch.bool,
                                       device=self.device)
                ctx, _ = self.belief_encoder(tokens, pad_mask)

                marginal_logits = self.y_head(ctx, pad_mask)
                y_logits = self.joint_y_head(ctx, marginal_logits, pad_mask)
                t_dist = self.t_head(ctx, pad_mask)

                p_success = torch.sigmoid(y_logits).clamp(min=1e-6)
                scores = -(t_dist.mean / p_success)  # (1, M)
                return scores.squeeze(0)  # (M,)

        return score_fn

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_batch(
        self,
        batch: dict[str, Tensor],
        skel_embeds: Tensor,
    ) -> dict[str, Tensor]:
        """Run forward pass on a collated batch, return loss dict.

        Parameters
        ----------
        batch:
            Collated prefix steps with (B, M) tensors.
        skel_embeds:
            (M, d_skel) skeleton embeddings (with gradient graph).
        """
        B = batch["applicability"].shape[0]
        M = skel_embeds.shape[0]

        # Expand skeleton data to batch
        skel_embeds_b = skel_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_skel)
        skel_lengths_b = self._skel_lengths_tb.unsqueeze(0).expand(B, -1)  # (B, M)

        # Token builder
        tokens = self.token_builder(
            skel_embeds_b,
            batch["applicability"],
            batch["revealed_mask"],
            batch["y"],
            batch["f"],
            batch["t"],
            skel_lengths_b,
        )  # (B, M, d_token)

        # Belief encoder (no padding — all M slots are real)
        pad_mask = torch.zeros(B, M, dtype=torch.bool, device=self.device)
        ctx, _ = self.belief_encoder(tokens, pad_mask)  # (B, M, d_model)

        # Prediction heads
        marginal_logits = self.y_head(ctx, pad_mask)  # (B, M)
        y_logits = self.joint_y_head(ctx, marginal_logits, pad_mask)  # (B, M)
        f_dist = self.f_head(ctx, pad_mask)  # Beta(B, M)
        t_dist = self.t_head(ctx, pad_mask)  # LogNormal(B, M)

        # Loss — CRITICAL: pass ~revealed_mask so active = applicable & ~revealed = C_t
        applicable_mask = batch["applicability"] > 0.5
        loss_dict = self.loss_fn(
            y_logits, f_dist, t_dist,
            batch["y_true"], batch["f_true"], batch["t_true"],
            applicable_mask,
            ~batch["revealed_mask"],  # invert: prediction targets are unrevealed
        )

        return loss_dict

    # ------------------------------------------------------------------
    # Prefix generation schedule
    # ------------------------------------------------------------------

    def _generate_epoch_prefixes(
        self,
        dataset: SkeletonDataset,
        epoch: int,
    ) -> list[PrefixStep]:
        """Generate all prefix steps for one epoch.

        Schedule:
        - Epochs 0 to warmup_epochs-1: epsilon_random for all instances
        - Epochs warmup_epochs+: per-instance random mix:
            50% teacher_forced, 25% epsilon_random, 25% on_policy
        """
        rng = torch.Generator().manual_seed(self.seed + epoch)
        all_steps: list[PrefixStep] = []

        # Build generators
        gen_teacher = PrefixGenerator("teacher_forced")
        gen_eps = PrefixGenerator("epsilon_random", epsilon=self.epsilon)

        # For on_policy: need model in eval mode + score_fn
        score_fn = None
        if epoch >= self.warmup_epochs:
            for m in self._modules:
                m.eval()
            skel_embeds = self._compute_skel_embeds(no_grad=True)
            score_fn = self._build_score_fn(skel_embeds)
            for m in self._modules:
                m.train()

        for i in range(len(dataset)):
            item = dataset[i]
            app = item.applicability
            suc = item.success
            scf = item.steps_completed_fraction
            rt = item.refinement_time
            lengths = self._skel_lengths_tb.cpu()

            if epoch < self.warmup_epochs:
                steps = gen_eps.generate(
                    app, suc, scf, rt, lengths, rng=rng,
                )
            else:
                # Per-instance random choice
                r = torch.rand(1, generator=rng).item()
                if r < 0.50:
                    steps = gen_teacher.generate(
                        app, suc, scf, rt, lengths, rng=rng,
                    )
                elif r < 0.75:
                    steps = gen_eps.generate(
                        app, suc, scf, rt, lengths, rng=rng,
                    )
                else:
                    steps = PrefixGenerator("on_policy").generate(
                        app, suc, scf, rt, lengths,
                        score_fn=score_fn, rng=rng,
                    )

            all_steps.extend(steps)

        return all_steps

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch. Returns mean losses."""
        for m in self._modules:
            m.train()

        all_steps = self._generate_epoch_prefixes(self.train_dataset, epoch)

        # Shuffle
        rng = torch.Generator().manual_seed(self.seed + epoch + 10000)
        perm = torch.randperm(len(all_steps), generator=rng)
        all_steps = [all_steps[i] for i in perm]

        # Accumulate losses
        total_loss = 0.0
        total_loss_y = 0.0
        total_loss_f = 0.0
        total_loss_t = 0.0
        n_batches = 0

        for start in range(0, len(all_steps), self.batch_size):
            batch_steps = all_steps[start : start + self.batch_size]
            batch = self._collate_prefix_steps(batch_steps)
            # Move to device
            batch = {
                k: v.to(self.device) for k, v in batch.items()
            }

            # Fresh skeleton embeddings with gradients each batch
            skel_embeds = self._compute_skel_embeds(no_grad=False)

            loss_dict = self._forward_batch(batch, skel_embeds)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
            nn.utils.clip_grad_norm_(
                [p for m in self._modules for p in m.parameters()],
                self.grad_clip_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_y += loss_dict["loss_y"].item()
            total_loss_f += loss_dict["loss_f"].item()
            total_loss_t += loss_dict["loss_t"].item()
            n_batches += 1

            if n_batches % self.log_every_steps == 0:
                logger.info(
                    "  epoch %d  step %d/%d  loss=%.4f",
                    epoch, n_batches,
                    math.ceil(len(all_steps) / self.batch_size),
                    loss.item(),
                )

        n_batches = max(n_batches, 1)
        return {
            "train_loss": total_loss / n_batches,
            "train_loss_y": total_loss_y / n_batches,
            "train_loss_f": total_loss_f / n_batches,
            "train_loss_t": total_loss_t / n_batches,
        }

    # ------------------------------------------------------------------
    # Validation + informativeness curve
    # ------------------------------------------------------------------

    def _validate(self) -> dict[str, Any]:
        """Validate on val set. Returns metrics including informativeness curve."""
        for m in self._modules:
            m.eval()

        gen = PrefixGenerator("teacher_forced")
        all_steps = []
        for i in range(len(self.val_dataset)):
            item = self.val_dataset[i]
            lengths = self._skel_lengths_tb.cpu()
            steps = gen.generate(
                item.applicability, item.success,
                item.steps_completed_fraction, item.refinement_time,
                lengths,
            )
            all_steps.extend(steps)

        if not all_steps:
            return {
                "val_loss": 0.0, "val_loss_y": 0.0,
                "val_loss_f": 0.0, "val_loss_t": 0.0,
            }

        # Forward all batches
        total_loss = 0.0
        total_loss_y = 0.0
        total_loss_f = 0.0
        total_loss_t = 0.0
        n_batches = 0

        # Informativeness curve: bucket Y BCE by step_index
        # Buckets: 0, 1, 2, 3, 5+ (step_index=4 excluded)
        bucket_losses: dict[str, list[float]] = {
            "nll_ht_0": [], "nll_ht_1": [], "nll_ht_2": [],
            "nll_ht_3": [], "nll_ht_5+": [],
        }

        with torch.no_grad():
            skel_embeds = self._compute_skel_embeds(no_grad=True)

            for start in range(0, len(all_steps), self.batch_size):
                batch_steps = all_steps[start : start + self.batch_size]
                batch = self._collate_prefix_steps(batch_steps)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss_dict = self._forward_batch(batch, skel_embeds)

                total_loss += loss_dict["loss"].item()
                total_loss_y += loss_dict["loss_y"].item()
                total_loss_f += loss_dict["loss_f"].item()
                total_loss_t += loss_dict["loss_t"].item()
                n_batches += 1

                # Per-step Y BCE for informativeness curve
                B, M = batch["applicability"].shape
                skel_embeds_b = skel_embeds.unsqueeze(0).expand(B, -1, -1)
                skel_lengths_b = self._skel_lengths_tb.unsqueeze(0).expand(B, -1)
                tokens = self.token_builder(
                    skel_embeds_b, batch["applicability"],
                    batch["revealed_mask"], batch["y"], batch["f"],
                    batch["t"], skel_lengths_b,
                )
                pad_mask = torch.zeros(B, M, dtype=torch.bool, device=self.device)
                ctx, _ = self.belief_encoder(tokens, pad_mask)
                marginal_logits = self.y_head(ctx, pad_mask)
                y_logits = self.joint_y_head(ctx, marginal_logits, pad_mask)

                # C_t mask per step: applicable & ~revealed
                applicable = batch["applicability"] > 0.5
                candidate_mask = applicable & ~batch["revealed_mask"]

                step_indices = batch["step_indices"]  # (B,)

                for b_idx in range(B):
                    si = step_indices[b_idx].item()
                    cm = candidate_mask[b_idx]  # (M,)
                    if not cm.any():
                        continue

                    bce = F.binary_cross_entropy_with_logits(
                        y_logits[b_idx][cm],
                        batch["y_true"][b_idx][cm],
                        reduction="mean",
                    ).item()

                    if si == 0:
                        bucket_losses["nll_ht_0"].append(bce)
                    elif si == 1:
                        bucket_losses["nll_ht_1"].append(bce)
                    elif si == 2:
                        bucket_losses["nll_ht_2"].append(bce)
                    elif si == 3:
                        bucket_losses["nll_ht_3"].append(bce)
                    elif si >= 5:
                        bucket_losses["nll_ht_5+"].append(bce)
                    # si == 4 excluded per spec

        n_batches = max(n_batches, 1)
        metrics: dict[str, Any] = {
            "val_loss": total_loss / n_batches,
            "val_loss_y": total_loss_y / n_batches,
            "val_loss_f": total_loss_f / n_batches,
            "val_loss_t": total_loss_t / n_batches,
        }

        # Compute bucket means
        for key, vals in bucket_losses.items():
            metrics[key] = sum(vals) / len(vals) if vals else float("nan")

        # Check monotonic decrease
        curve_keys = ["nll_ht_0", "nll_ht_1", "nll_ht_2", "nll_ht_3", "nll_ht_5+"]
        curve_vals = [metrics[k] for k in curve_keys if not math.isnan(metrics[k])]
        if len(curve_vals) >= 2:
            for i in range(1, len(curve_vals)):
                if curve_vals[i] > curve_vals[i - 1]:
                    logger.warning(
                        "Informativeness curve NOT monotonically decreasing: %s",
                        {k: f"{metrics[k]:.4f}" for k in curve_keys
                         if not math.isnan(metrics[k])},
                    )
                    break

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_val_nll_ht3: float,
        val_metrics: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dicts": {
                    "skeleton_encoder": self.skeleton_encoder.state_dict(),
                    "token_builder": self.token_builder.state_dict(),
                    "belief_encoder": self.belief_encoder.state_dict(),
                    "y_head": self.y_head.state_dict(),
                    "f_head": self.f_head.state_dict(),
                    "t_head": self.t_head.state_dict(),
                    "joint_y_head": self.joint_y_head.state_dict(),
                },
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_nll_ht3": best_val_nll_ht3,
                "val_metrics": val_metrics,
            },
            path,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run full training. Returns final metrics."""
        torch.manual_seed(self.seed)

        best_val_nll_ht3 = float("inf")
        best_path = self.checkpoint_dir / "belief_best.pt"
        last_path = self.checkpoint_dir / "belief_last.pt"

        final_metrics: dict[str, Any] = {}

        for epoch in range(self.num_epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate()
            self.scheduler.step()

            # Log
            nll_0 = val_metrics.get("nll_ht_0", float("nan"))
            nll_1 = val_metrics.get("nll_ht_1", float("nan"))
            nll_2 = val_metrics.get("nll_ht_2", float("nan"))
            nll_3 = val_metrics.get("nll_ht_3", float("nan"))
            nll_5p = val_metrics.get("nll_ht_5+", float("nan"))

            logger.info(
                "epoch %d/%d  train_loss=%.4f  val_nll=%.4f",
                epoch, self.num_epochs,
                train_metrics["train_loss"], val_metrics["val_loss"],
            )
            logger.info(
                "  informativeness: |H|=0:%.4f  1:%.4f  2:%.4f  3:%.4f  5+:%.4f",
                nll_0, nll_1, nll_2, nll_3, nll_5p,
            )

            # Checkpoint best by val NLL at |H_t|=3
            current_nll_ht3 = val_metrics.get("nll_ht_3", float("inf"))
            if not math.isnan(current_nll_ht3) and current_nll_ht3 < best_val_nll_ht3:
                best_val_nll_ht3 = current_nll_ht3
                self._save_checkpoint(best_path, epoch, best_val_nll_ht3, val_metrics)
            self._save_checkpoint(last_path, epoch, best_val_nll_ht3, val_metrics)

            final_metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
                "best_val_nll_ht3": best_val_nll_ht3,
            }

        return final_metrics
