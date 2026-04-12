"""Train the belief encoder pipeline with rollout-consistent prefixes.

Hydra CLI entry point. Builds all model components from config, constructs
a BeliefTrainer, and runs the training loop.

Usage
-----
    uv run python experiments/train_belief_encoder.py \
        data.train_path=data/train.h5 data.val_path=data/val.h5
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

# Ensure project src is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from alphatamp.data.skeleton_dataset import SkeletonDataset
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.losses import PredictionNLLLoss
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder
from alphatamp.training.trainer import BeliefTrainer

logger = logging.getLogger(__name__)


def _select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


@hydra.main(
    config_path="conf",
    config_name="train_belief_encoder_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # 1. Seed
    torch.manual_seed(cfg.train.seed)

    # 2. Device
    device = _select_device(str(cfg.train.device))
    logger.info("Using device: %s", device)

    # 3. Load datasets
    logger.info("Loading train dataset: %s", cfg.data.train_path)
    train_ds = SkeletonDataset(cfg.data.train_path, preload=True)
    logger.info("Loading val dataset: %s", cfg.data.val_path)
    val_ds = SkeletonDataset(cfg.data.val_path, preload=True)

    logger.info("Train: N=%d, M=%d", train_ds.N, train_ds.M)
    logger.info("Val:   N=%d, M=%d", val_ds.N, val_ds.M)

    # 4. Extract vocab sizes
    num_op_types = len(train_ds.op_type_vocab)
    num_objects = len(train_ds.obj_vocab)
    logger.info("Vocab: %d op types, %d objects", num_op_types, num_objects)

    # 5. Build model components
    d_skel = cfg.skeleton_encoder.d_model
    d_out = cfg.token_builder.d_out

    skeleton_encoder = SkeletonEncoder(
        num_op_types=num_op_types,
        num_objects=num_objects,
        d_model=d_skel,
        n_heads=cfg.skeleton_encoder.n_heads,
        n_layers=cfg.skeleton_encoder.n_layers,
        max_seq_len=cfg.skeleton_encoder.max_seq_len,
        dropout=cfg.skeleton_encoder.dropout,
    )

    token_builder = TokenBuilder(
        d_skel=d_skel,
        d_out=d_out,
        dropout=cfg.skeleton_encoder.dropout,
    )

    belief_encoder = BeliefEncoder(
        d_token=cfg.belief_encoder.d_token,
        d_model=cfg.belief_encoder.d_model,
        n_heads=cfg.belief_encoder.n_heads,
        n_layers=cfg.belief_encoder.n_layers,
        ffn_dim=cfg.belief_encoder.ffn_dim,
        dropout=cfg.belief_encoder.dropout,
    )

    d_model = cfg.belief_encoder.d_model
    hidden_dim = cfg.heads.hidden_dim
    head_dropout = cfg.heads.dropout

    y_head = YHead(d_model, hidden_dim=hidden_dim, dropout=head_dropout)
    f_head = FHead(d_model, hidden_dim=hidden_dim, dropout=head_dropout)
    t_head = THead(d_model, hidden_dim=hidden_dim, dropout=head_dropout)
    joint_y_head = JointYHead(
        d_model,
        n_heads=cfg.heads.joint_y_n_heads,
        rank=cfg.heads.joint_y_rank,
        dropout=head_dropout,
    )

    loss_fn = PredictionNLLLoss(
        lambda_f=cfg.loss.lambda_f,
        lambda_t=cfg.loss.lambda_t,
    )

    # Count parameters
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [skeleton_encoder, token_builder, belief_encoder,
                  y_head, f_head, t_head, joint_y_head]
    )
    logger.info("Total trainable parameters: %s", f"{total_params:,}")

    # 6. Build trainer
    trainer = BeliefTrainer(
        skeleton_encoder=skeleton_encoder,
        token_builder=token_builder,
        belief_encoder=belief_encoder,
        y_head=y_head,
        f_head=f_head,
        t_head=t_head,
        joint_y_head=joint_y_head,
        loss_fn=loss_fn,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        num_epochs=cfg.train.num_epochs,
        batch_size=cfg.train.batch_size,
        grad_clip_norm=cfg.train.grad_clip_norm,
        lr_min_fraction=cfg.train.lr_min_fraction,
        warmup_epochs=cfg.train.warmup_epochs,
        epsilon=cfg.train.epsilon,
        device=device,
        seed=cfg.train.seed,
        checkpoint_dir=cfg.checkpoint.output_dir,
        log_every_steps=cfg.train.log_every_steps,
    )

    # 7. Train
    logger.info("Starting training for %d epochs", cfg.train.num_epochs)
    final_metrics = trainer.train()

    # 8. Summary
    logger.info("Training complete.")
    logger.info("Final metrics: %s", final_metrics)


if __name__ == "__main__":
    main()
