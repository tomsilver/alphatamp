"""Tests for BeliefTrainer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure project src and experiments are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from alphatamp.data.skeleton_dataset import SkeletonDataset, write_skeleton_dataset
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.losses import PredictionNLLLoss
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder
from alphatamp.training.prefix_generator import PrefixGenerator, PrefixStep
from alphatamp.training.trainer import BeliefTrainer
from build_synthetic_dataset import build_synthetic_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_SKEL = 16
D_OUT = 8
D_TOKEN = D_SKEL + D_OUT  # 24
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 1


@pytest.fixture(scope="module")
def synthetic_hdf5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a small synthetic dataset to HDF5."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "test.h5"
    dd = build_synthetic_dataset(N=20, M=8, rng_seed=0)
    write_skeleton_dataset(path, dd)
    return path


@pytest.fixture
def small_models(synthetic_hdf5: Path) -> dict:
    """Build small model components for fast tests."""
    ds = SkeletonDataset(synthetic_hdf5, preload=True)
    num_op_types = len(ds.op_type_vocab)
    num_objects = len(ds.obj_vocab)

    skeleton_encoder = SkeletonEncoder(
        num_op_types=num_op_types,
        num_objects=num_objects,
        d_model=D_SKEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
    )
    token_builder = TokenBuilder(d_skel=D_SKEL, d_out=D_OUT, dropout=0.0)
    belief_encoder = BeliefEncoder(
        d_token=D_TOKEN, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS,
        ffn_dim=D_MODEL * 2, dropout=0.0,
    )
    y_head = YHead(D_MODEL, dropout=0.0)
    f_head = FHead(D_MODEL, dropout=0.0)
    t_head = THead(D_MODEL, dropout=0.0)
    joint_y_head = JointYHead(D_MODEL, n_heads=N_HEADS, rank=4, dropout=0.0)
    loss_fn = PredictionNLLLoss()

    return {
        "skeleton_encoder": skeleton_encoder,
        "token_builder": token_builder,
        "belief_encoder": belief_encoder,
        "y_head": y_head,
        "f_head": f_head,
        "t_head": t_head,
        "joint_y_head": joint_y_head,
        "loss_fn": loss_fn,
    }


@pytest.fixture
def trainer(
    synthetic_hdf5: Path,
    small_models: dict,
    tmp_path: Path,
) -> BeliefTrainer:
    """Build a BeliefTrainer with small models for 2 epochs."""
    ds = SkeletonDataset(synthetic_hdf5, preload=True)
    return BeliefTrainer(
        **small_models,
        train_dataset=ds,
        val_dataset=ds,  # use same for testing
        num_epochs=2,
        batch_size=4,
        warmup_epochs=1,
        device=torch.device("cpu"),
        seed=42,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_every_steps=100,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collate_prefix_steps() -> None:
    """Collation of PrefixSteps produces correct shapes."""
    M = 6
    steps = []
    for si in range(3):
        steps.append(PrefixStep(
            applicability=torch.ones(M),
            lengths=torch.full((M,), 5.0),
            revealed_mask=torch.zeros(M, dtype=torch.bool),
            revealed_outcomes={
                "y": torch.zeros(M),
                "f": torch.zeros(M),
                "t": torch.zeros(M),
            },
            y_true=torch.rand(M),
            f_true=torch.rand(M),
            t_true=torch.rand(M),
            oracle_ranking=torch.full((M,), -1, dtype=torch.int64),
            step_index=si,
        ))

    batch = BeliefTrainer._collate_prefix_steps(steps)
    assert batch["applicability"].shape == (3, M)
    assert batch["revealed_mask"].shape == (3, M)
    assert batch["y_true"].shape == (3, M)
    assert batch["step_indices"].shape == (3,)
    assert batch["step_indices"].tolist() == [0, 1, 2]


def test_forward_batch_loss_keys(trainer: BeliefTrainer) -> None:
    """_forward_batch returns dict with expected loss keys, all finite."""
    # Generate a few prefix steps and run forward
    gen = PrefixGenerator("teacher_forced")
    item = trainer.train_dataset[0]
    lengths = trainer._skel_lengths_tb.cpu()
    steps = gen.generate(
        item.applicability, item.success,
        item.steps_completed_fraction, item.refinement_time,
        lengths,
    )

    batch = trainer._collate_prefix_steps(steps[:2])
    batch = {k: v.to(trainer.device) for k, v in batch.items()}

    with torch.no_grad():
        skel_embeds = trainer._compute_skel_embeds(no_grad=True)
        loss_dict = trainer._forward_batch(batch, skel_embeds)

    assert "loss" in loss_dict
    assert "loss_y" in loss_dict
    assert "loss_f" in loss_dict
    assert "loss_t" in loss_dict
    for key, val in loss_dict.items():
        assert torch.isfinite(val), f"{key} is not finite: {val}"


def test_loss_on_unrevealed_only(trainer: BeliefTrainer) -> None:
    """Loss is zero when C_t is empty (all applicable revealed)."""
    M = trainer.train_dataset.M
    # Construct a step where all applicable are revealed (C_t empty)
    app = trainer.train_dataset[0].applicability
    step_full = PrefixStep(
        applicability=app,
        lengths=trainer._skel_lengths_tb.cpu(),
        revealed_mask=torch.ones(M, dtype=torch.bool),  # all revealed
        revealed_outcomes={
            "y": trainer.train_dataset[0].success,
            "f": trainer.train_dataset[0].steps_completed_fraction,
            "t": trainer.train_dataset[0].refinement_time,
        },
        y_true=trainer.train_dataset[0].success,
        f_true=trainer.train_dataset[0].steps_completed_fraction,
        t_true=trainer.train_dataset[0].refinement_time,
        oracle_ranking=torch.full((M,), -1, dtype=torch.int64),
        step_index=99,
    )

    batch_full = trainer._collate_prefix_steps([step_full])
    batch_full = {k: v.to(trainer.device) for k, v in batch_full.items()}

    with torch.no_grad():
        skel_embeds = trainer._compute_skel_embeds(no_grad=True)
        loss_full = trainer._forward_batch(batch_full, skel_embeds)

    # C_t is empty → active = applicable & ~revealed = empty → loss = 0
    assert loss_full["loss"].item() == 0.0

    # Now a step with some unrevealed → loss should be > 0
    step_partial = PrefixStep(
        applicability=app,
        lengths=trainer._skel_lengths_tb.cpu(),
        revealed_mask=~(app > 0.5),  # only inapplicable revealed
        revealed_outcomes={
            "y": torch.zeros(M),
            "f": torch.zeros(M),
            "t": torch.zeros(M),
        },
        y_true=trainer.train_dataset[0].success,
        f_true=trainer.train_dataset[0].steps_completed_fraction,
        t_true=trainer.train_dataset[0].refinement_time,
        oracle_ranking=torch.full((M,), -1, dtype=torch.int64),
        step_index=0,
    )

    batch_partial = trainer._collate_prefix_steps([step_partial])
    batch_partial = {k: v.to(trainer.device) for k, v in batch_partial.items()}

    with torch.no_grad():
        loss_partial = trainer._forward_batch(batch_partial, skel_embeds)

    # With unrevealed candidates, loss should differ from zero
    # (NLL of continuous distributions can be negative, so check != 0)
    if (app > 0.5).any():
        assert loss_partial["loss"].item() != 0.0


def test_score_fn_returns_valid(trainer: BeliefTrainer) -> None:
    """_build_score_fn returns (M,) finite tensor."""
    trainer.skeleton_encoder.eval()
    trainer.token_builder.eval()
    trainer.belief_encoder.eval()
    trainer.y_head.eval()
    trainer.f_head.eval()
    trainer.t_head.eval()
    trainer.joint_y_head.eval()

    skel_embeds = trainer._compute_skel_embeds(no_grad=True)
    score_fn = trainer._build_score_fn(skel_embeds)

    item = trainer.train_dataset[0]
    M = trainer.train_dataset.M
    step = PrefixStep(
        applicability=item.applicability,
        lengths=trainer._skel_lengths_tb.cpu(),
        revealed_mask=~(item.applicability > 0.5),
        revealed_outcomes={
            "y": torch.zeros(M),
            "f": torch.zeros(M),
            "t": torch.zeros(M),
        },
        y_true=item.success,
        f_true=item.steps_completed_fraction,
        t_true=item.refinement_time,
        oracle_ranking=torch.full((M,), -1, dtype=torch.int64),
        step_index=0,
    )

    scores = score_fn(step)
    assert scores.shape == (M,)
    assert torch.isfinite(scores).all()


def test_gradient_through_skeleton_encoder(trainer: BeliefTrainer) -> None:
    """After one training step, SkeletonEncoder params have non-zero grads."""
    trainer.skeleton_encoder.train()

    gen = PrefixGenerator("teacher_forced")
    item = trainer.train_dataset[0]
    lengths = trainer._skel_lengths_tb.cpu()
    steps = gen.generate(
        item.applicability, item.success,
        item.steps_completed_fraction, item.refinement_time,
        lengths,
    )

    batch = trainer._collate_prefix_steps(steps[:2])
    batch = {k: v.to(trainer.device) for k, v in batch.items()}

    skel_embeds = trainer._compute_skel_embeds(no_grad=False)
    loss_dict = trainer._forward_batch(batch, skel_embeds)
    loss_dict["loss"].backward()

    has_grad = False
    for p in trainer.skeleton_encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "SkeletonEncoder params should have non-zero gradients"


def test_informativeness_curve_buckets(trainer: BeliefTrainer) -> None:
    """Validation returns expected informativeness curve bucket keys."""
    metrics = trainer._validate()

    assert "val_loss" in metrics
    assert "val_loss_y" in metrics
    # Bucket keys should exist (may be NaN if no steps at that depth)
    for key in ["nll_ht_0", "nll_ht_1", "nll_ht_2", "nll_ht_3", "nll_ht_5+"]:
        assert key in metrics, f"Missing bucket key: {key}"


def test_checkpoint_roundtrip(trainer: BeliefTrainer, tmp_path: Path) -> None:
    """Save + load checkpoint; model state dicts match."""
    ckpt_path = tmp_path / "test_ckpt.pt"
    trainer._save_checkpoint(ckpt_path, epoch=0, best_val_nll_ht3=1.0,
                             val_metrics={"val_loss": 0.5})

    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, weights_only=False)

    assert ckpt["epoch"] == 0
    assert ckpt["best_val_nll_ht3"] == 1.0
    assert "model_state_dicts" in ckpt

    expected_keys = {
        "skeleton_encoder", "token_builder", "belief_encoder",
        "y_head", "f_head", "t_head", "joint_y_head",
    }
    assert set(ckpt["model_state_dicts"].keys()) == expected_keys

    # Verify state dict can be loaded back
    trainer.skeleton_encoder.load_state_dict(
        ckpt["model_state_dicts"]["skeleton_encoder"]
    )


def test_two_epoch_integration(trainer: BeliefTrainer) -> None:
    """Full 2-epoch training completes with valid metrics and checkpoint."""
    final_metrics = trainer.train()

    assert "train_loss" in final_metrics
    assert "val_loss" in final_metrics
    assert "epoch" in final_metrics
    assert final_metrics["epoch"] == 1  # 0-indexed, 2 epochs → last epoch = 1

    # Best checkpoint should exist
    best_path = trainer.checkpoint_dir / "belief_best.pt"
    last_path = trainer.checkpoint_dir / "belief_last.pt"
    assert last_path.exists()
    # best_path exists if nll_ht_3 was non-NaN at some point
    # (depends on data; just check last exists)
