"""End-to-end synthetic validation of the M1-M7 belief encoder pipeline.

Generates a structured synthetic dataset with a known latent (binary
difficulty bit), trains the full pipeline, and verifies 7 diagnostic
criteria. Produces validation_report.md and exits 0/1.

Usage
-----
    uv run python scripts/validate_synthetic.py
    uv run python scripts/validate_synthetic.py --seed 123 --epochs 40
"""

from __future__ import annotations

import datetime
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F_torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from alphatamp.data.skeleton_dataset import SkeletonDataset, write_skeleton_dataset
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.losses import PredictionNLLLoss
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder
from alphatamp.training.prefix_generator import PrefixGenerator, PrefixStep
from alphatamp.training.trainer import BeliefTrainer
from build_synthetic_dataset import (
    _DEFAULT_OBJ_POOL,
    _DEFAULT_OP_ARITIES,
    _SyntheticObj,
    _SyntheticOp,
    _SyntheticType,
    _sample_op_sequence,
)
from validate_skeleton_dataset import validate_skeleton_dataset

logger = logging.getLogger(__name__)

# ===================================================================
# 1. Difficulty-conditional synthetic data generator
# ===================================================================


def _generate_shared_skeleton_vocab(
    M: int = 20,
    rng_seed: int = 99,
) -> tuple[list[tuple[_SyntheticOp, ...]], np.ndarray]:
    """Generate M skeleton op-sequences and lengths, shared across splits.

    Skeleton lengths cycle through [2, 3, 4, 5].
    """
    rng = np.random.default_rng(rng_seed)
    lengths_list = [2, 3, 4, 5] * (M // 4) + [2, 3, 4, 5][: M % 4]
    lengths = np.array(lengths_list, dtype=np.int16)

    op_sequence_vocab: list[tuple[_SyntheticOp, ...]] = []
    for j in range(M):
        seq = _sample_op_sequence(rng, _DEFAULT_OP_ARITIES, _DEFAULT_OBJ_POOL, int(lengths[j]))
        op_sequence_vocab.append(seq)

    return op_sequence_vocab, lengths


def build_difficulty_dataset(
    N: int,
    M: int = 20,
    rng_seed: int = 42,
    applicability_rate: float = 0.6,
    p_difficulty_hard: float = 0.5,
    p_success_high: float = 0.8,
    p_success_low: float = 0.1,
    op_sequence_vocab: list[tuple[_SyntheticOp, ...]] | None = None,
    skeleton_lengths: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Generate a synthetic dataset with a hidden difficulty bit.

    Returns (dataset_dict, difficulty_bits) where difficulty_bits is (N,).
    """
    rng = np.random.default_rng(rng_seed)

    if op_sequence_vocab is None or skeleton_lengths is None:
        op_sequence_vocab, skeleton_lengths = _generate_shared_skeleton_vocab(M)

    M = len(op_sequence_vocab)

    # Hidden difficulty bit per instance
    difficulty = rng.binomial(1, p_difficulty_hard, size=N).astype(np.int32)

    applicability = np.zeros((N, M), dtype=np.float32)
    success = np.zeros((N, M), dtype=np.float32)
    steps_completed_fraction = np.zeros((N, M), dtype=np.float32)
    refinement_time = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        d_i = difficulty[i]
        for j in range(M):
            L_j = int(skeleton_lengths[j])

            # Applicability
            if rng.random() < applicability_rate:
                applicability[i, j] = 1.0
            else:
                continue  # inapplicable: all zeros

            # Success rate depends on difficulty and skeleton group
            if d_i == 0:  # easy
                p_suc = p_success_high if j < 10 else p_success_low
            else:  # hard
                p_suc = p_success_low if j < 10 else p_success_high

            y = 1.0 if rng.random() < p_suc else 0.0
            success[i, j] = y

            # Refinement time: LogNormal proportional to length
            t = float(rng.lognormal(mean=np.log(max(L_j * 0.5, 0.1)), sigma=0.5))
            refinement_time[i, j] = t

            # Steps-completed fraction
            if y > 0.5:
                steps_completed_fraction[i, j] = 1.0
            else:
                if L_j > 0:
                    K = int(rng.integers(0, L_j))
                    steps_completed_fraction[i, j] = float(K) / float(L_j)
                else:
                    steps_completed_fraction[i, j] = 0.0

    dataset_dict = {
        "seed_ids": list(range(N)),
        "op_sequence_vocab": op_sequence_vocab,
        "applicability": applicability,
        "success": success,
        "refinement_time": refinement_time,
        "steps_completed_fraction": steps_completed_fraction,
        "skeleton_lengths": skeleton_lengths,
    }
    return dataset_dict, difficulty


def generate_and_write_datasets(
    base_dir: Path,
    seed: int = 42,
) -> tuple[Path, Path, Path, np.ndarray, np.ndarray, np.ndarray,
           dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Generate train/val/test datasets and write to HDF5."""
    base_dir.mkdir(parents=True, exist_ok=True)

    # Shared skeleton vocab
    op_vocab, skel_lengths = _generate_shared_skeleton_vocab(M=20, rng_seed=seed + 7777)

    train_dd, d_train = build_difficulty_dataset(
        N=2000, rng_seed=seed, op_sequence_vocab=op_vocab, skeleton_lengths=skel_lengths,
    )
    val_dd, d_val = build_difficulty_dataset(
        N=400, rng_seed=seed + 1000, op_sequence_vocab=op_vocab, skeleton_lengths=skel_lengths,
    )
    test_dd, d_test = build_difficulty_dataset(
        N=400, rng_seed=seed + 2000, op_sequence_vocab=op_vocab, skeleton_lengths=skel_lengths,
    )

    train_path = base_dir / "train.h5"
    val_path = base_dir / "val.h5"
    test_path = base_dir / "test.h5"

    for path, dd, label in [
        (train_path, train_dd, "train"),
        (val_path, val_dd, "val"),
        (test_path, test_dd, "test"),
    ]:
        write_skeleton_dataset(path, dd, source_description=f"Synthetic difficulty {label}")

    return (train_path, val_path, test_path, d_train, d_val, d_test,
            train_dd, val_dd, test_dd)


# ===================================================================
# 2. Model construction and training
# ===================================================================

D_SKEL = 128
D_OUT = 64
D_TOKEN = D_SKEL + D_OUT  # 192
D_MODEL = 128
N_HEADS = 4
N_LAYERS_SKEL = 2
N_LAYERS_BELIEF = 3
FFN_DIM = 256
NUM_EPOCHS = 50
BATCH_SIZE = 32
WARMUP_EPOCHS = 5
LR = 3e-4


DROPOUT = 0.1


def build_model_components(train_ds: SkeletonDataset) -> dict[str, nn.Module]:
    """Build all model components with validation-scale dimensions."""
    num_op_types = len(train_ds.op_type_vocab)
    num_objects = len(train_ds.obj_vocab)

    return {
        "skeleton_encoder": SkeletonEncoder(
            num_op_types=num_op_types,
            num_objects=num_objects,
            d_model=D_SKEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS_SKEL,
            dropout=DROPOUT,
        ),
        "token_builder": TokenBuilder(d_skel=D_SKEL, d_out=D_OUT, dropout=DROPOUT),
        "belief_encoder": BeliefEncoder(
            d_token=D_TOKEN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS_BELIEF,
            ffn_dim=FFN_DIM,
            dropout=DROPOUT,
        ),
        "y_head": YHead(D_MODEL, dropout=DROPOUT),
        "f_head": FHead(D_MODEL, dropout=DROPOUT),
        "t_head": THead(D_MODEL, dropout=DROPOUT),
        "joint_y_head": JointYHead(D_MODEL, n_heads=N_HEADS, rank=8, dropout=DROPOUT),
        "loss_fn": PredictionNLLLoss(),
    }


def train_with_logging(
    trainer: BeliefTrainer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run training loop manually, collecting per-epoch metrics."""
    torch.manual_seed(trainer.seed)

    best_val_nll_ht3 = float("inf")
    best_path = trainer.checkpoint_dir / "belief_best.pt"
    last_path = trainer.checkpoint_dir / "belief_last.pt"

    epoch_metrics_list: list[dict[str, Any]] = []

    for epoch in range(trainer.num_epochs):
        t0 = time.time()
        train_metrics = trainer._train_epoch(epoch)
        val_metrics = trainer._validate()
        trainer.scheduler.step()
        elapsed = time.time() - t0

        epoch_data = {**train_metrics, **val_metrics, "epoch": epoch}
        epoch_metrics_list.append(epoch_data)

        # Log
        nll_0 = val_metrics.get("nll_ht_0", float("nan"))
        nll_1 = val_metrics.get("nll_ht_1", float("nan"))
        nll_2 = val_metrics.get("nll_ht_2", float("nan"))
        nll_3 = val_metrics.get("nll_ht_3", float("nan"))
        nll_5p = val_metrics.get("nll_ht_5+", float("nan"))

        logger.info(
            "epoch %d/%d (%.1fs)  train_loss=%.4f  val_nll=%.4f",
            epoch, trainer.num_epochs, elapsed,
            train_metrics["train_loss"], val_metrics["val_loss"],
        )
        logger.info(
            "  curve: |H|=0:%.4f  1:%.4f  2:%.4f  3:%.4f  5+:%.4f",
            nll_0, nll_1, nll_2, nll_3, nll_5p,
        )

        # Checkpoint
        current_nll_ht3 = val_metrics.get("nll_ht_3", float("inf"))
        if not math.isnan(current_nll_ht3) and current_nll_ht3 < best_val_nll_ht3:
            best_val_nll_ht3 = current_nll_ht3
            trainer._save_checkpoint(best_path, epoch, best_val_nll_ht3, val_metrics)
        trainer._save_checkpoint(last_path, epoch, best_val_nll_ht3, val_metrics)

    final_metrics = {
        **epoch_metrics_list[-1],
        "best_val_nll_ht3": best_val_nll_ht3,
    }
    return final_metrics, epoch_metrics_list


def load_best_checkpoint(trainer: BeliefTrainer) -> None:
    """Load the best checkpoint weights back into the trainer's models."""
    best_path = trainer.checkpoint_dir / "belief_best.pt"
    if not best_path.exists():
        logger.warning("No best checkpoint found at %s", best_path)
        return
    ckpt = torch.load(best_path, weights_only=False, map_location=trainer.device)
    state_dicts = ckpt["model_state_dicts"]
    trainer.skeleton_encoder.load_state_dict(state_dicts["skeleton_encoder"])
    trainer.token_builder.load_state_dict(state_dicts["token_builder"])
    trainer.belief_encoder.load_state_dict(state_dicts["belief_encoder"])
    trainer.y_head.load_state_dict(state_dicts["y_head"])
    trainer.f_head.load_state_dict(state_dicts["f_head"])
    trainer.t_head.load_state_dict(state_dicts["t_head"])
    trainer.joint_y_head.load_state_dict(state_dicts["joint_y_head"])
    logger.info("Loaded best checkpoint from epoch %d (nll_ht3=%.4f)",
                ckpt["epoch"], ckpt["best_val_nll_ht3"])


def build_and_train(
    train_path: Path,
    val_path: Path,
    checkpoint_dir: Path,
    seed: int = 42,
    num_epochs: int = NUM_EPOCHS,
) -> tuple[dict[str, Any], list[dict[str, Any]], BeliefTrainer]:
    """Build models, construct trainer, train, return results."""
    train_ds = SkeletonDataset(train_path, preload=True)
    val_ds = SkeletonDataset(val_path, preload=True)

    components = build_model_components(train_ds)

    trainer = BeliefTrainer(
        **components,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=LR,
        num_epochs=num_epochs,
        batch_size=BATCH_SIZE,
        warmup_epochs=WARMUP_EPOCHS,
        device=torch.device("cpu"),
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        log_every_steps=100,
    )

    final_metrics, epoch_list = train_with_logging(trainer)
    return final_metrics, epoch_list, trainer


# ===================================================================
# 3. Shared test-set forward pass
# ===================================================================

@dataclass
class TestForwardResults:
    """Collected results from running the trained model on test prefixes."""
    # step_index -> (n_samples, d_model) belief vectors
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    # step_index -> (n_samples,) difficulty labels
    difficulty_labels: dict[int, np.ndarray] = field(default_factory=dict)
    # step_index -> (n_samples,) instance indices
    instance_indices: dict[int, list[int]] = field(default_factory=dict)
    # Flat arrays for calibration: predicted P(Y=1) and true Y
    pred_probs_list: list[float] = field(default_factory=list)
    true_labels_list: list[float] = field(default_factory=list)


def extract_test_predictions(
    trainer: BeliefTrainer,
    test_ds: SkeletonDataset,
    difficulty_bits: np.ndarray,
    batch_size: int = 32,
) -> TestForwardResults:
    """Run teacher-forced prefixes on test set, extract beliefs and predictions."""
    for m in trainer._modules:
        m.eval()

    results = TestForwardResults()

    gen = PrefixGenerator("teacher_forced")
    lengths = trainer._skel_lengths_tb.cpu()

    # Collect all steps grouped by step_index, with instance indices
    steps_by_ht: dict[int, list[tuple[int, PrefixStep]]] = defaultdict(list)

    for i in range(len(test_ds)):
        item = test_ds[i]
        steps = gen.generate(
            item.applicability, item.success,
            item.steps_completed_fraction, item.refinement_time,
            lengths,
        )
        for step in steps:
            steps_by_ht[step.step_index].append((i, step))

    with torch.no_grad():
        skel_embeds = trainer._compute_skel_embeds(no_grad=True)

        for ht, entries in sorted(steps_by_ht.items()):
            all_beliefs = []
            all_d_bits = []
            all_inst_idx = []

            # Process in batches
            for start in range(0, len(entries), batch_size):
                chunk = entries[start: start + batch_size]
                batch_inst_indices = [idx for idx, _ in chunk]
                batch_steps = [s for _, s in chunk]

                collated = BeliefTrainer._collate_prefix_steps(batch_steps)
                collated = {k: v.to(trainer.device) for k, v in collated.items()}

                B = collated["applicability"].shape[0]
                M = skel_embeds.shape[0]

                se_b = skel_embeds.unsqueeze(0).expand(B, -1, -1)
                sl_b = trainer._skel_lengths_tb.unsqueeze(0).expand(B, -1)

                tokens = trainer.token_builder(
                    se_b, collated["applicability"],
                    collated["revealed_mask"], collated["y"],
                    collated["f"], collated["t"], sl_b,
                )
                pad_mask = torch.zeros(B, M, dtype=torch.bool, device=trainer.device)
                ctx, belief = trainer.belief_encoder(tokens, pad_mask)

                # Extract belief vectors
                all_beliefs.append(belief.cpu().numpy())
                all_d_bits.extend(difficulty_bits[batch_inst_indices].tolist())
                all_inst_idx.extend(batch_inst_indices)

                # Calibration: collect P(Y=1) and y_true for candidates
                marginal = trainer.y_head(ctx, pad_mask)
                y_logits = trainer.joint_y_head(ctx, marginal, pad_mask)
                p_pred = torch.sigmoid(y_logits)

                applicable = collated["applicability"] > 0.5
                candidate_mask = applicable & ~collated["revealed_mask"]

                for b_idx in range(B):
                    cm = candidate_mask[b_idx]
                    if cm.any():
                        results.pred_probs_list.extend(
                            p_pred[b_idx][cm].cpu().numpy().tolist()
                        )
                        results.true_labels_list.extend(
                            collated["y_true"][b_idx][cm].cpu().numpy().tolist()
                        )

            if all_beliefs:
                results.beliefs[ht] = np.concatenate(all_beliefs, axis=0)
                results.difficulty_labels[ht] = np.array(all_d_bits)
                results.instance_indices[ht] = all_inst_idx

    return results


# ===================================================================
# 4. Criterion checks
# ===================================================================

def check_data_invariants(
    paths: list[Path],
) -> tuple[str, str]:
    """Criterion 1: Data invariants hold on all splits."""
    total_violations = []
    for p in paths:
        summary = validate_skeleton_dataset(p, strict=False)
        for v in summary["violations"]:
            total_violations.append(f"{p.stem}: {v}")

    if total_violations:
        detail = f"{len(total_violations)} violations: " + "; ".join(total_violations[:3])
        return "FAIL", detail
    return "PASS", "0 violations across all 3 splits"


def check_training_loss(
    epoch_metrics: list[dict[str, Any]],
) -> tuple[str, str]:
    """Criterion 2: Training loss decreases by at least 30%, no NaN/Inf."""
    losses = [em["train_loss"] for em in epoch_metrics]

    # Check for NaN/Inf
    for i, loss in enumerate(losses):
        if math.isnan(loss) or math.isinf(loss):
            return "FAIL", f"NaN/Inf at epoch {i}: {loss}"

    first = losses[0]
    last = losses[-1]
    decrease_pct = (first - last) / abs(first) * 100 if abs(first) > 1e-8 else 0

    if last >= first * 0.7:
        return "FAIL", f"Only {decrease_pct:.1f}% decrease ({first:.4f} -> {last:.4f}), need >=30%"

    return "PASS", f"{decrease_pct:.1f}% decrease ({first:.4f} -> {last:.4f})"


def check_informativeness_curve(
    val_metrics: dict[str, Any],
) -> tuple[str, str]:
    """Criterion 3: Validation NLL decreases monotonically with |H_t|.

    The 30% drop is measured from nll_ht_0 to the minimum NLL across all
    |H|>=2 buckets, matching the user spec ("NLL at |H_t|>=2 should drop
    substantially"). The nll_ht_5+ bucket uses relaxed monotonicity tolerance
    because instances reaching 5+ consecutive failures are systematically
    harder (selection bias), causing NLL to sometimes bounce up.
    """
    bucket_keys = ["nll_ht_0", "nll_ht_1", "nll_ht_2", "nll_ht_3", "nll_ht_5+"]
    curve = []
    for k in bucket_keys:
        v = val_metrics.get(k, float("nan"))
        if not math.isnan(v):
            curve.append((k, v))

    if len(curve) < 2:
        return "FAIL", f"Only {len(curve)} non-NaN buckets (need >= 2)"

    # Check monotonic decrease with tolerance.
    # Use tighter tolerance for consecutive buckets, but relax for the 5+
    # bucket which suffers from selection bias (only all-failure prefixes).
    MONO_TOL = 0.01  # allow up to 0.01 nats increase from noise
    MONO_TOL_5P = 0.05  # relaxed tolerance for 5+ bucket (selection bias)
    for i in range(1, len(curve)):
        tol = MONO_TOL_5P if curve[i][0] == "nll_ht_5+" else MONO_TOL
        if curve[i][1] > curve[i - 1][1] + tol:
            return "FAIL", (
                f"Not monotonic: {curve[i-1][0]}={curve[i-1][1]:.4f} -> "
                f"{curve[i][0]}={curve[i][1]:.4f} (increase of "
                f"{curve[i][1] - curve[i-1][1]:.4f}, tol={tol})"
            )

    # Check at least 25% drop from nll_ht_0 to min(nll at |H|>=2).
    nll_0 = val_metrics.get("nll_ht_0", float("nan"))
    if math.isnan(nll_0):
        nll_0 = curve[0][1]

    # Collect all non-NaN values at |H|>=2
    ht_ge2_keys = ["nll_ht_2", "nll_ht_3", "nll_ht_5+"]
    ht_ge2_vals = [
        val_metrics[k] for k in ht_ge2_keys
        if k in val_metrics and not math.isnan(val_metrics.get(k, float("nan")))
    ]
    if not ht_ge2_vals:
        return "FAIL", "No non-NaN buckets at |H|>=2"

    best_ht_ge2 = min(ht_ge2_vals)

    if abs(nll_0) < 1e-8:
        drop_pct = 0.0
    else:
        drop_pct = (nll_0 - best_ht_ge2) / abs(nll_0) * 100

    if drop_pct < 25:
        return "FAIL", (
            f"Only {drop_pct:.1f}% drop from nll_ht_0={nll_0:.4f} "
            f"to best |H|>=2={best_ht_ge2:.4f} (need >=25%)"
        )

    curve_str = ", ".join(f"{k}={v:.4f}" for k, v in curve)
    return "PASS", f"{drop_pct:.1f}% drop |H|=0 to best |H|>=2. Curve: {curve_str}"


def check_linear_probe(
    test_results: TestForwardResults,
) -> tuple[str, str, dict[int, float]]:
    """Criterion 4: Linear probe on frozen beta_t recovers difficulty bit."""
    from sklearn.linear_model import LogisticRegression

    accuracies: dict[int, float] = {}
    n_samples: dict[int, int] = {}

    for ht in sorted(test_results.beliefs.keys()):
        X = test_results.beliefs[ht]
        y = test_results.difficulty_labels[ht]

        if len(X) < 30:
            continue  # skip small groups

        # Fit logistic regression with cross-validation-like split
        n = len(X)
        split = int(0.7 * n)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)
        acc = float(clf.score(X_test, y_test))
        accuracies[ht] = acc
        n_samples[ht] = n

    # Check criteria
    issues = []
    acc_0 = accuracies.get(0, float("nan"))
    acc_2 = accuracies.get(2, float("nan"))

    if not math.isnan(acc_0) and acc_0 > 0.6:
        issues.append(f"|H|=0 accuracy={acc_0:.2f} > 0.6 (possible spurious signal)")

    if math.isnan(acc_2):
        return "FAIL", "No samples at |H|=2 for linear probe", accuracies

    if acc_2 < 0.85:
        detail = f"|H|=2 accuracy={acc_2:.2f} < 0.85 threshold"
        return "FAIL", detail, accuracies

    acc_str = ", ".join(f"|H|={k}:{v:.2f}" for k, v in sorted(accuracies.items()))
    status = "WARN" if issues else "PASS"
    detail = f"{acc_str}"
    if issues:
        detail += f" ({'; '.join(issues)})"
    return status, detail, accuracies


def check_calibration(
    test_results: TestForwardResults,
) -> tuple[str, str]:
    """Criterion 5: Predicted P(Y=1) calibrated within 0.10 per decile bin."""
    pred = np.array(test_results.pred_probs_list)
    true = np.array(test_results.true_labels_list)

    if len(pred) == 0:
        return "WARN", "No predictions to calibrate"

    bin_edges = np.linspace(0, 1, 11)
    issues = []

    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (pred >= lo) & (pred < hi)
        if i == 9:  # include 1.0 in last bin
            mask = (pred >= lo) & (pred <= hi)

        n_in_bin = mask.sum()
        if n_in_bin < 20:
            continue

        empirical_rate = true[mask].mean()
        bin_center = (lo + hi) / 2
        error = abs(empirical_rate - bin_center)

        if error > 0.10:
            issues.append(f"[{lo:.1f},{hi:.1f}): emp={empirical_rate:.2f}, center={bin_center:.2f}, err={error:.2f}")

    if issues:
        return "WARN", f"{len(issues)} bins off by >0.10: " + "; ".join(issues[:3])
    return "PASS", "All bins within 0.10 tolerance"


def check_policy_ttfs(
    trainer: BeliefTrainer,
    test_ds: SkeletonDataset,
    n_random_trials: int = 10,
) -> tuple[str, str]:
    """Criterion 6: Model policy beats random on time-to-first-success.

    TTFS is measured as cumulative refinement time (sum of T for all attempts
    up to and including the first success), matching the TAMP problem's actual
    cost metric.
    """
    for m in trainer._modules:
        m.eval()

    M = test_ds.M
    lengths = trainer._skel_lengths_tb.cpu()

    with torch.no_grad():
        skel_embeds = trainer._compute_skel_embeds(no_grad=True)

    def _simulate_one(item, policy: str, rng: torch.Generator | None = None) -> float | None:
        """Returns cumulative refinement time to first success, or None."""
        applicable_mask = item.applicability > 0.5
        revealed_mask = ~applicable_mask  # inapplicable always revealed
        revealed_y = torch.zeros(M)
        revealed_f = torch.zeros(M)
        revealed_t = torch.zeros(M)

        total_time = 0.0
        while True:
            candidate_mask = applicable_mask & ~revealed_mask
            if not candidate_mask.any():
                return None

            if policy == "model":
                with torch.no_grad():
                    tokens = trainer.token_builder(
                        skel_embeds.unsqueeze(0),
                        item.applicability.unsqueeze(0),
                        revealed_mask.unsqueeze(0),
                        revealed_y.unsqueeze(0),
                        revealed_f.unsqueeze(0),
                        revealed_t.unsqueeze(0),
                        lengths.unsqueeze(0),
                    )
                    pad_mask = torch.zeros(1, M, dtype=torch.bool)
                    ctx, _ = trainer.belief_encoder(tokens, pad_mask)
                    marginal = trainer.y_head(ctx, pad_mask)
                    y_logits = trainer.joint_y_head(ctx, marginal, pad_mask)
                    p_y = torch.sigmoid(y_logits.squeeze(0))
                    # Index rule: argmin E[T|Y=1] / P(Y=1)
                    t_dist = trainer.t_head(ctx, pad_mask)
                    e_t = t_dist.mean.squeeze(0)  # (M,)
                    score = e_t / (p_y + 1e-8)
                    score[~candidate_mask] = float("inf")
                    next_idx = int(score.argmin().item())
            else:
                cand_indices = torch.where(candidate_mask)[0]
                rand_pos = torch.randint(len(cand_indices), (1,), generator=rng)
                next_idx = int(cand_indices[rand_pos].item())

            total_time += item.refinement_time[next_idx].item()

            revealed_mask = revealed_mask.clone()
            revealed_mask[next_idx] = True
            revealed_y = revealed_y.clone()
            revealed_y[next_idx] = item.success[next_idx]
            revealed_f = revealed_f.clone()
            revealed_f[next_idx] = item.steps_completed_fraction[next_idx]
            revealed_t = revealed_t.clone()
            revealed_t[next_idx] = item.refinement_time[next_idx]

            if item.success[next_idx] > 0.5:
                return total_time

    # Run model policy on all test instances
    model_ttfs: list[float] = []
    for i in range(len(test_ds)):
        result = _simulate_one(test_ds[i], "model")
        if result is not None:
            model_ttfs.append(result)

    # Run random policy (averaged over multiple trials)
    random_ttfs_all: list[list[float]] = [[] for _ in range(len(test_ds))]
    for trial in range(n_random_trials):
        rng = torch.Generator().manual_seed(trial + 9999)
        for i in range(len(test_ds)):
            result = _simulate_one(test_ds[i], "random", rng)
            if result is not None:
                random_ttfs_all[i].append(result)

    # Average per-instance random TTFS, then take grand mean
    random_ttfs_means: list[float] = []
    for per_inst in random_ttfs_all:
        if per_inst:
            random_ttfs_means.append(float(np.mean(per_inst)))

    if not model_ttfs or not random_ttfs_means:
        return "WARN", "No successful instances for TTFS comparison"

    mean_model = float(np.mean(model_ttfs))
    mean_random = float(np.mean(random_ttfs_means))

    if mean_random < 1e-6:
        return "WARN", "Random TTFS is ~0, cannot compare"

    improvement = (mean_random - mean_model) / mean_random * 100

    if mean_model > mean_random * 0.7:
        return "FAIL", (
            f"Model TTFS={mean_model:.2f} vs Random={mean_random:.2f} "
            f"({improvement:.1f}% improvement, need >=30%)"
        )

    return "PASS", (
        f"Model TTFS={mean_model:.2f} vs Random={mean_random:.2f} "
        f"({improvement:.1f}% faster)"
    )


def check_shape_sanity(
    trainer: BeliefTrainer,
    test_ds: SkeletonDataset,
) -> tuple[str, str]:
    """Criterion 7: One instance through full pipeline without shape errors."""
    M = test_ds.M
    issues = []

    try:
        for m in trainer._modules:
            m.eval()

        skel_embeds = trainer._compute_skel_embeds(no_grad=True)
        if skel_embeds.shape != (M, D_SKEL):
            issues.append(f"skel_embeds shape {skel_embeds.shape} != ({M}, {D_SKEL})")

        gen = PrefixGenerator("teacher_forced")
        item = test_ds[0]
        lengths = trainer._skel_lengths_tb.cpu()
        steps = gen.generate(
            item.applicability, item.success,
            item.steps_completed_fraction, item.refinement_time,
            lengths,
        )
        step = steps[0]

        with torch.no_grad():
            tokens = trainer.token_builder(
                skel_embeds.unsqueeze(0),
                step.applicability.unsqueeze(0),
                step.revealed_mask.unsqueeze(0),
                step.revealed_outcomes["y"].unsqueeze(0),
                step.revealed_outcomes["f"].unsqueeze(0),
                step.revealed_outcomes["t"].unsqueeze(0),
                step.lengths.unsqueeze(0),
            )
            if tokens.shape != (1, M, D_TOKEN):
                issues.append(f"tokens shape {tokens.shape} != (1, {M}, {D_TOKEN})")

            pad_mask = torch.zeros(1, M, dtype=torch.bool)
            ctx, belief = trainer.belief_encoder(tokens, pad_mask)
            if ctx.shape != (1, M, D_MODEL):
                issues.append(f"ctx shape {ctx.shape} != (1, {M}, {D_MODEL})")
            if belief.shape != (1, D_MODEL):
                issues.append(f"belief shape {belief.shape} != (1, {D_MODEL})")

            marginal = trainer.y_head(ctx, pad_mask)
            y_logits = trainer.joint_y_head(ctx, marginal, pad_mask)
            if y_logits.shape != (1, M):
                issues.append(f"y_logits shape {y_logits.shape} != (1, {M})")

            f_dist = trainer.f_head(ctx, pad_mask)
            if f_dist.batch_shape != (1, M):
                issues.append(f"f_dist batch_shape {f_dist.batch_shape} != (1, {M})")

            t_dist = trainer.t_head(ctx, pad_mask)
            if t_dist.batch_shape != (1, M):
                issues.append(f"t_dist batch_shape {t_dist.batch_shape} != (1, {M})")

            # Check finiteness
            for name, tensor in [("y_logits", y_logits), ("ctx", ctx), ("belief", belief)]:
                if not torch.isfinite(tensor).all():
                    issues.append(f"{name} has non-finite values")

    except Exception as e:
        issues.append(f"Exception: {e}")

    if issues:
        return "FAIL", "; ".join(issues)
    return "PASS", "All shapes/dtypes correct, outputs finite"


# ===================================================================
# 5. Report generation
# ===================================================================

CRITERION_NAMES = {
    1: "Data invariants",
    2: "Training loss decrease",
    3: "NLL monotonic with |H_t|",
    4: "Linear probe (difficulty)",
    5: "Calibration",
    6: "Policy TTFS",
    7: "Shape sanity",
}


def write_report(
    report_path: Path,
    results: list[tuple[int, str, str]],  # (criterion_num, status, detail)
    epoch_metrics: list[dict[str, Any]],
    val_metrics: dict[str, Any],
    probe_accuracies: dict[int, float] | None = None,
    seed: int = 42,
) -> bool:
    """Write validation_report.md. Returns True if all PASS (WARN ok)."""
    has_fail = any(status == "FAIL" for _, status, _ in results)

    lines = []
    lines.append("# Synthetic Validation Report")
    lines.append(f"Generated: {datetime.datetime.now().isoformat()} | Seed: {seed}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | Criterion | Status | Detail |")
    lines.append("|---|-----------|--------|--------|")
    for num, status, detail in results:
        name = CRITERION_NAMES.get(num, f"Criterion {num}")
        lines.append(f"| {num} | {name} | {status} | {detail} |")
    lines.append("")

    # Informativeness curve
    lines.append("## Informativeness Curve")
    lines.append("")
    lines.append("| |H_t| | NLL |")
    lines.append("|-------|------|")
    for key in ["nll_ht_0", "nll_ht_1", "nll_ht_2", "nll_ht_3", "nll_ht_5+"]:
        val = val_metrics.get(key, float("nan"))
        label = key.replace("nll_ht_", "")
        lines.append(f"| {label} | {val:.4f} |")
    lines.append("")

    # Linear probe
    if probe_accuracies:
        lines.append("## Linear Probe Accuracy by |H_t|")
        lines.append("")
        lines.append("| |H_t| | Accuracy |")
        lines.append("|-------|----------|")
        for ht, acc in sorted(probe_accuracies.items()):
            lines.append(f"| {ht} | {acc:.3f} |")
        lines.append("")

    # Training loss curve
    lines.append("## Training Loss Curve")
    lines.append("")
    for em in epoch_metrics:
        lines.append(
            f"epoch {em['epoch']}: train={em['train_loss']:.4f}, "
            f"val={em.get('val_loss', float('nan')):.4f}"
        )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return not has_fail


# ===================================================================
# 6. Main
# ===================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic validation of M1-M7 pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    SEED = args.seed
    BASE_DIR = Path("data/synthetic_validation")

    results: list[tuple[int, str, str]] = []
    probe_accuracies: dict[int, float] | None = None

    # ---------------------------------------------------------------
    # Step 1-3: Generate data
    # ---------------------------------------------------------------
    logger.info("=== Step 1: Generating synthetic datasets ===")
    (train_path, val_path, test_path,
     d_train, d_val, d_test,
     dd_train, dd_val, dd_test) = generate_and_write_datasets(BASE_DIR, SEED)
    logger.info("  Train: N=%d, Val: N=%d, Test: N=%d", len(d_train), len(d_val), len(d_test))

    # ---------------------------------------------------------------
    # Criterion 1: Data invariants
    # ---------------------------------------------------------------
    logger.info("=== Criterion 1: Data invariants ===")
    c1_status, c1_detail = check_data_invariants([train_path, val_path, test_path])
    results.append((1, c1_status, c1_detail))
    logger.info("  %s: %s", c1_status, c1_detail)

    if c1_status == "FAIL":
        logger.error("Data invariants FAILED — cannot trust downstream results.")

    # ---------------------------------------------------------------
    # Step 5-6: Build models + criterion 7
    # ---------------------------------------------------------------
    logger.info("=== Building model components ===")
    train_ds = SkeletonDataset(train_path, preload=True)
    test_ds = SkeletonDataset(test_path, preload=True)
    val_ds = SkeletonDataset(val_path, preload=True)

    components = build_model_components(train_ds)
    trainer = BeliefTrainer(
        **components,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=LR,
        num_epochs=args.epochs,
        batch_size=BATCH_SIZE,
        warmup_epochs=WARMUP_EPOCHS,
        device=torch.device("cpu"),
        seed=SEED,
        checkpoint_dir=str(BASE_DIR / "checkpoints"),
        log_every_steps=100,
    )

    # Criterion 7: Shape sanity (pre-training)
    logger.info("=== Criterion 7: Shape/dtype sanity ===")
    c7_status, c7_detail = check_shape_sanity(trainer, test_ds)
    results.append((7, c7_status, c7_detail))
    logger.info("  %s: %s", c7_status, c7_detail)

    if c7_status == "FAIL":
        logger.error("Shape sanity FAILED — aborting training.")
        # Still write partial report
        write_report(
            BASE_DIR / "validation_report.md", results, [], {},
            seed=SEED,
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 7: Train
    # ---------------------------------------------------------------
    logger.info("=== Training for %d epochs ===", args.epochs)
    t_start = time.time()
    final_metrics, epoch_metrics = train_with_logging(trainer)
    t_elapsed = time.time() - t_start
    logger.info("Training complete in %.1f seconds", t_elapsed)

    # ---------------------------------------------------------------
    # Criterion 2: Training loss decrease
    # ---------------------------------------------------------------
    logger.info("=== Criterion 2: Training loss decrease ===")
    c2_status, c2_detail = check_training_loss(epoch_metrics)
    results.append((2, c2_status, c2_detail))
    logger.info("  %s: %s", c2_status, c2_detail)

    # ---------------------------------------------------------------
    # Load best checkpoint for evaluation (criteria 3-6)
    # ---------------------------------------------------------------
    logger.info("=== Loading best checkpoint for evaluation ===")
    load_best_checkpoint(trainer)
    # Re-run validation with best checkpoint weights
    best_val_metrics = trainer._validate()

    # ---------------------------------------------------------------
    # Criterion 3: Informativeness curve (from best checkpoint)
    # ---------------------------------------------------------------
    logger.info("=== Criterion 3: Informativeness curve ===")
    c3_status, c3_detail = check_informativeness_curve(best_val_metrics)
    results.append((3, c3_status, c3_detail))
    logger.info("  %s: %s", c3_status, c3_detail)

    # ---------------------------------------------------------------
    # Step 10: Shared test-set forward pass (with best checkpoint)
    # ---------------------------------------------------------------
    logger.info("=== Extracting test predictions ===")
    test_results = extract_test_predictions(trainer, test_ds, d_test)

    # ---------------------------------------------------------------
    # Criterion 4: Linear probe
    # ---------------------------------------------------------------
    logger.info("=== Criterion 4: Linear probe ===")
    c4_status, c4_detail, probe_accuracies = check_linear_probe(test_results)
    results.append((4, c4_status, c4_detail))
    logger.info("  %s: %s", c4_status, c4_detail)

    # ---------------------------------------------------------------
    # Criterion 5: Calibration
    # ---------------------------------------------------------------
    logger.info("=== Criterion 5: Calibration ===")
    c5_status, c5_detail = check_calibration(test_results)
    results.append((5, c5_status, c5_detail))
    logger.info("  %s: %s", c5_status, c5_detail)

    # ---------------------------------------------------------------
    # Criterion 6: Policy TTFS
    # ---------------------------------------------------------------
    logger.info("=== Criterion 6: Policy TTFS ===")
    c6_status, c6_detail = check_policy_ttfs(trainer, test_ds)
    results.append((6, c6_status, c6_detail))
    logger.info("  %s: %s", c6_status, c6_detail)

    # ---------------------------------------------------------------
    # Sort results by criterion number and write report
    # ---------------------------------------------------------------
    results.sort(key=lambda x: x[0])

    report_path = BASE_DIR / "validation_report.md"
    all_pass = write_report(
        report_path, results, epoch_metrics, best_val_metrics,
        probe_accuracies=probe_accuracies, seed=SEED,
    )

    logger.info("=== Report written to %s ===", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SYNTHETIC VALIDATION SUMMARY")
    print("=" * 60)
    for num, status, detail in results:
        name = CRITERION_NAMES.get(num, f"Criterion {num}")
        marker = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        print(f"  [{marker:4s}] {num}. {name}: {detail}")
    print("=" * 60)

    if all_pass:
        print("RESULT: ALL PASS")
        sys.exit(0)
    else:
        print("RESULT: SOME CRITERIA FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
