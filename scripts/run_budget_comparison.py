#!/usr/bin/env python3
"""Budget-sweep comparison for belief-policy skeleton selection baselines.

Evaluates these policies under a strict per-instance time budget:
  - OracleBaseline
  - IndexPolicy (belief encoder; requires checkpoint)
  - SuccessFirstFixedOrder (fit on train)
  - ShortestFirstFixedOrder

Outputs:
  - success_rate_vs_budget.png
  - time_success_only_vs_budget.png
  - budget_comparison_summary.json
  - budget_comparison_metrics.npz

Usage:
    uv run python scripts/run_budget_comparison.py \
        --test-data artifacts_hdf5/encoder_o2/test.h5 \
        --train-data artifacts_hdf5/encoder_o2/train.h5 \
        --checkpoint checkpoints/belief_encoder/o2/belief_best.pt \
        --output-dir results/budget_comparison/o2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

# Ensure project src is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from alphatamp.data.skeleton_dataset import SkeletonDataset, SkeletonItem
from alphatamp.evaluation.policy import (
    IndexPolicy,
    OracleBaseline,
    SelectionPolicy,
    ShortestFirstFixedOrder,
    SuccessFirstFixedOrder,
)
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.prediction_heads import JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder

logger = logging.getLogger(__name__)

# Match scripts/run_comparison.py defaults for checkpoint compatibility.
D_SKEL = 128
D_OUT = 64
D_TOKEN = D_SKEL + D_OUT
D_MODEL = 128
N_HEADS = 4
N_LAYERS_SKEL = 2
N_LAYERS_BELIEF = 4
FFN_DIM = 256
DROPOUT = 0.0


@dataclass(frozen=True)
class BudgetEvalMetrics:
    """Aggregated metrics at a single budget for one policy."""

    success_rate: float
    mean_time_success_only: float
    mean_time_total: float
    solved_count: int
    failed_count: int
    elapsed_times: list[float]
    success_flags: list[bool]


def build_model_components(dataset: SkeletonDataset) -> dict[str, nn.Module]:
    """Build belief model components matching training architecture."""
    num_op_types = len(dataset.op_type_vocab)
    num_objects = len(dataset.obj_vocab)

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
        "t_head": THead(D_MODEL, dropout=DROPOUT),
        "joint_y_head": JointYHead(D_MODEL, n_heads=N_HEADS, rank=8, dropout=DROPOUT),
    }


def load_checkpoint(
    components: dict[str, nn.Module],
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Load trained belief model state dicts."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dicts = ckpt["model_state_dicts"]

    for name in [
        "skeleton_encoder",
        "token_builder",
        "belief_encoder",
        "y_head",
        "t_head",
        "joint_y_head",
    ]:
        components[name].load_state_dict(state_dicts[name])

    epoch = ckpt.get("epoch", "?")
    best_nll = ckpt.get("best_val_nll_ht3", float("nan"))
    logger.info("Loaded checkpoint from epoch %s (best_val_nll_ht3=%.4f)", epoch, best_nll)


def _budget_rollout_single(
    policy: SelectionPolicy,
    item: SkeletonItem,
    dataset: SkeletonDataset,
    budget_seconds: float,
    epsilon: float,
) -> tuple[bool, float]:
    """Run one budgeted rollout with reveal semantics matching OfflineEvaluator.

    Inapplicable skeletons are revealed at start. If next attempt would exceed
    remaining budget, rollout stops immediately and counts as failure unless a
    prior success was already found.
    """
    applicable_mask = item.applicability > 0.5
    revealed_mask = ~applicable_mask
    revealed_y = torch.zeros(dataset.M)
    revealed_f = torch.zeros(dataset.M)
    revealed_t = torch.zeros(dataset.M)

    policy.reset(item, dataset)

    remaining = float(budget_seconds)
    elapsed = 0.0

    while True:
        candidate_mask = applicable_mask & ~revealed_mask
        if not candidate_mask.any():
            return False, elapsed

        next_idx = policy.select(
            candidate_mask,
            revealed_mask,
            revealed_y,
            revealed_f,
            revealed_t,
        )

        attempt_time = float(item.refinement_time[next_idx].item())
        if attempt_time > (remaining + epsilon):
            return False, elapsed

        elapsed += attempt_time
        remaining -= attempt_time

        revealed_mask = revealed_mask.clone()
        revealed_mask[next_idx] = True
        revealed_y = revealed_y.clone()
        revealed_y[next_idx] = item.success[next_idx]
        revealed_f = revealed_f.clone()
        revealed_f[next_idx] = item.steps_completed_fraction[next_idx]
        revealed_t = revealed_t.clone()
        revealed_t[next_idx] = item.refinement_time[next_idx]

        if item.success[next_idx] > 0.5:
            return True, elapsed


def evaluate_policy_at_budget(
    dataset: SkeletonDataset,
    policy: SelectionPolicy,
    budget_seconds: float,
    epsilon: float,
) -> BudgetEvalMetrics:
    """Evaluate one policy at one budget across all instances."""
    elapsed_times: list[float] = []
    success_flags: list[bool] = []

    for idx in range(len(dataset)):
        solved, elapsed = _budget_rollout_single(
            policy,
            dataset[idx],
            dataset,
            budget_seconds,
            epsilon,
        )
        success_flags.append(bool(solved))
        elapsed_times.append(float(elapsed))

    solved_times = [t for t, s in zip(elapsed_times, success_flags) if s]
    solved_count = int(sum(1 for s in success_flags if s))
    total = len(success_flags)

    return BudgetEvalMetrics(
        success_rate=(float(solved_count) / float(total)) if total > 0 else 0.0,
        mean_time_success_only=(
            float(np.mean(solved_times)) if solved_times else float("nan")
        ),
        mean_time_total=(float(np.mean(elapsed_times)) if elapsed_times else float("nan")),
        solved_count=solved_count,
        failed_count=total - solved_count,
        elapsed_times=elapsed_times,
        success_flags=success_flags,
    )


def _plot_success_rate_vs_budget(
    budgets: np.ndarray,
    curves: dict[str, np.ndarray],
    budget_marker: float,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, values in curves.items():
        plt.plot(budgets, values, linewidth=2, label=label)
    plt.axvline(
        budget_marker,
        linestyle="--",
        linewidth=1.5,
        label=f"Budget={budget_marker:g}s",
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time budget (seconds)")
    plt.ylabel("Success rate")
    plt.title("Success rate vs budget")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_success_only_time_vs_budget(
    budgets: np.ndarray,
    curves: dict[str, np.ndarray],
    budget_marker: float,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, values in curves.items():
        plt.plot(budgets, values, linewidth=2, label=label)
    plt.axvline(
        budget_marker,
        linestyle="--",
        linewidth=1.5,
        label=f"Budget={budget_marker:g}s",
    )
    plt.xlabel("Time budget (seconds)")
    plt.ylabel("Mean time to success (seconds)")
    plt.title("Refinement time on success vs budget")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def run_budget_comparison(
    test_ds: SkeletonDataset,
    train_ds: SkeletonDataset,
    components: dict[str, nn.Module],
    device: torch.device,
    budget_seconds: float,
    sweep_min_seconds: float,
    sweep_max_seconds: float,
    sweep_num_points: int,
    epsilon: float,
    output_dir: Path,
    dpi: int,
) -> None:
    """Evaluate all policies over a budget sweep and write artifacts."""
    policies: list[tuple[str, str, SelectionPolicy]] = []

    policies.append(("oracle", "Oracle", OracleBaseline()))
    policies.append(
        (
            "index_policy",
            "Ours (IndexPolicy)",
            IndexPolicy(
                **components,
                dataset=test_ds,
                device=device,
            ),
        )
    )

    success_first = SuccessFirstFixedOrder()
    success_first.fit(train_ds)
    success_first.save_ordering(output_dir / "success_first_ordering.json")
    policies.append(("success_first", "SuccessFirst", success_first))

    policies.append(
        (
            "shortest_first",
            "ShortestFirst",
            ShortestFirstFixedOrder(test_ds.skeleton_lengths),
        )
    )

    budgets = np.linspace(
        sweep_min_seconds,
        sweep_max_seconds,
        sweep_num_points,
        dtype=np.float64,
    )

    success_curves: dict[str, list[float]] = {slug: [] for slug, _, _ in policies}
    success_time_curves: dict[str, list[float]] = {slug: [] for slug, _, _ in policies}

    fixed_budget_metrics: dict[str, BudgetEvalMetrics] = {}
    for slug, _, policy in policies:
        fixed_budget_metrics[slug] = evaluate_policy_at_budget(
            test_ds,
            policy,
            budget_seconds,
            epsilon,
        )

    for sweep_budget in budgets.tolist():
        for slug, _, policy in policies:
            metrics = evaluate_policy_at_budget(
                test_ds,
                policy,
                float(sweep_budget),
                epsilon,
            )
            success_curves[slug].append(float(metrics.success_rate))
            success_time_curves[slug].append(float(metrics.mean_time_success_only))

    success_curves_np = {
        slug: np.asarray(vals, dtype=np.float32)
        for slug, vals in success_curves.items()
    }
    success_time_curves_np = {
        slug: np.asarray(vals, dtype=np.float32)
        for slug, vals in success_time_curves.items()
    }

    label_map = {slug: label for slug, label, _ in policies}
    plot_success_curves = {
        label_map[slug]: arr for slug, arr in success_curves_np.items()
    }
    plot_success_time_curves = {
        label_map[slug]: arr for slug, arr in success_time_curves_np.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    success_plot_path = output_dir / "success_rate_vs_budget.png"
    success_time_plot_path = output_dir / "time_success_only_vs_budget.png"

    _plot_success_rate_vs_budget(
        budgets,
        plot_success_curves,
        budget_seconds,
        success_plot_path,
        dpi,
    )
    _plot_success_only_time_vs_budget(
        budgets,
        plot_success_time_curves,
        budget_seconds,
        success_time_plot_path,
        dpi,
    )

    summary = {
        "paths": {
            "output_dir": str(output_dir),
        },
        "config": {
            "budget_seconds": float(budget_seconds),
            "sweep_min_seconds": float(sweep_min_seconds),
            "sweep_max_seconds": float(sweep_max_seconds),
            "sweep_num_points": int(sweep_num_points),
            "epsilon": float(epsilon),
            "dpi": int(dpi),
            "device": str(device),
        },
        "policies": {
            slug: {
                "label": label_map[slug],
                "success_rate": float(metric.success_rate),
                "mean_time_success_only": float(metric.mean_time_success_only),
                "mean_time_total": float(metric.mean_time_total),
                "solved_count": int(metric.solved_count),
                "failed_count": int(metric.failed_count),
            }
            for slug, metric in fixed_budget_metrics.items()
        },
    }

    summary_path = output_dir / "budget_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    npz_payload: dict[str, np.ndarray] = {
        "budgets": budgets.astype(np.float32),
    }
    for slug in success_curves_np:
        npz_payload[f"{slug}_success_curve"] = success_curves_np[slug]
        npz_payload[f"{slug}_time_success_only_curve"] = success_time_curves_np[slug]

        fixed = fixed_budget_metrics[slug]
        npz_payload[f"{slug}_elapsed_times"] = np.asarray(
            fixed.elapsed_times,
            dtype=np.float32,
        )
        npz_payload[f"{slug}_success_flags"] = np.asarray(
            fixed.success_flags,
            dtype=np.int8,
        )
        npz_payload[f"{slug}_success_rate"] = np.asarray(
            fixed.success_rate,
            dtype=np.float32,
        )
        npz_payload[f"{slug}_mean_time_success_only"] = np.asarray(
            fixed.mean_time_success_only,
            dtype=np.float32,
        )
        npz_payload[f"{slug}_mean_time_total"] = np.asarray(
            fixed.mean_time_total,
            dtype=np.float32,
        )

    metrics_path = output_dir / "budget_comparison_metrics.npz"
    np.savez(metrics_path, **npz_payload)

    print(f"Saved summary: {summary_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {success_plot_path}")
    print(f"Saved plot: {success_time_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budget-sweep comparison for belief-policy baselines.",
    )
    parser.add_argument("--test-data", type=Path, required=True, help="Path to test HDF5")
    parser.add_argument("--train-data", type=Path, required=True, help="Path to train HDF5")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to belief encoder checkpoint (belief_best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/budget_comparison"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--budget-seconds", type=float, default=20.0)
    parser.add_argument("--sweep-min-seconds", type=float, default=1.0)
    parser.add_argument("--sweep-max-seconds", type=float, default=40.0)
    parser.add_argument("--sweep-num-points", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.budget_seconds <= 0:
        raise ValueError("--budget-seconds must be > 0")
    if args.sweep_min_seconds <= 0 or args.sweep_max_seconds <= 0:
        raise ValueError("--sweep-min-seconds and --sweep-max-seconds must be > 0")
    if args.sweep_max_seconds < args.sweep_min_seconds:
        raise ValueError("--sweep-max-seconds must be >= --sweep-min-seconds")
    if args.sweep_num_points < 2:
        raise ValueError("--sweep-num-points must be >= 2")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be >= 0")

    for path in [args.test_data, args.train_data, args.checkpoint]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device(args.device)

    logger.info("Loading test dataset: %s", args.test_data)
    test_ds = SkeletonDataset(args.test_data, preload=True)

    logger.info("Loading train dataset: %s", args.train_data)
    train_ds = SkeletonDataset(args.train_data, preload=True)

    logger.info("Building model components")
    components = build_model_components(test_ds)
    for mod in components.values():
        mod.to(device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    load_checkpoint(components, args.checkpoint, device)

    run_budget_comparison(
        test_ds=test_ds,
        train_ds=train_ds,
        components=components,
        device=device,
        budget_seconds=args.budget_seconds,
        sweep_min_seconds=args.sweep_min_seconds,
        sweep_max_seconds=args.sweep_max_seconds,
        sweep_num_points=args.sweep_num_points,
        epsilon=args.epsilon,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
