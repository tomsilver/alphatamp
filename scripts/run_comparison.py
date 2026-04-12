#!/usr/bin/env python3
"""Compare skeleton selection policies on held-out test data.

Evaluates:
  - OracleBaseline (ground-truth lower bound)
  - IndexPolicy (learned method)
  - SuccessFirstFixedOrder (fitted on training set)
  - ShortestFirstFixedOrder (by plan length)

Reports a Markdown table with mean TTFS, 95% bootstrap CI, success@k,
and fraction of instances where success was ever found.

Usage:
    uv run python scripts/run_comparison.py \
        --test-data data/synthetic/test.h5 \
        --train-data data/synthetic/train.h5 \
        --checkpoint checkpoints/belief_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Ensure project src is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from alphatamp.data.skeleton_dataset import SkeletonDataset
from alphatamp.evaluation.evaluator import EvalMetrics, OfflineEvaluator
from alphatamp.evaluation.policy import (
    IndexPolicy,
    OracleBaseline,
    ShortestFirstFixedOrder,
    SuccessFirstFixedOrder,
)
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model hyperparameters (match validate_synthetic.py)
# ---------------------------------------------------------------------------

D_SKEL = 128
D_OUT = 64
D_TOKEN = D_SKEL + D_OUT  # 192
D_MODEL = 128
N_HEADS = 4
N_LAYERS_SKEL = 2
N_LAYERS_BELIEF = 4
FFN_DIM = 256
DROPOUT = 0.0  # eval mode — dropout inactive regardless


# ---------------------------------------------------------------------------
# Model construction and checkpoint loading
# ---------------------------------------------------------------------------


def build_model_components(dataset: SkeletonDataset) -> dict[str, nn.Module]:
    """Build model components matching the training architecture."""
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
    """Load model state dicts from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dicts = ckpt["model_state_dicts"]

    for name in [
        "skeleton_encoder", "token_builder", "belief_encoder",
        "y_head", "t_head", "joint_y_head",
    ]:
        components[name].load_state_dict(state_dicts[name])

    epoch = ckpt.get("epoch", "?")
    best_nll = ckpt.get("best_val_nll_ht3", float("nan"))
    logger.info("Loaded checkpoint from epoch %s (best_val_nll_ht3=%.4f)", epoch, best_nll)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap resampling."""
    if not values:
        return (float("inf"), float("inf"), float("inf"))
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([
        arr[rng.integers(len(arr), size=len(arr))].mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Feasibility ceiling
# ---------------------------------------------------------------------------


def compute_feasibility_fraction(dataset: SkeletonDataset) -> float:
    """Fraction of instances with at least one applicable Y=1 skeleton."""
    n_feasible = 0
    for i in range(len(dataset)):
        item = dataset[i]
        applicable_mask = item.applicability > 0.5
        if (item.success * applicable_mask.float()).sum() > 0.5:
            n_feasible += 1
    return n_feasible / len(dataset) if len(dataset) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def run_comparison(
    test_ds: SkeletonDataset,
    train_ds: SkeletonDataset,
    components: dict[str, nn.Module] | None,
    device: torch.device,
    n_seeds: int = 5,
    output_dir: Path | None = None,
) -> str:
    """Run the full comparison and return the Markdown table as a string."""

    # -- Construct policies ------------------------------------------------
    policies: dict[str, object] = {}

    policies["Oracle"] = OracleBaseline()

    if components is not None:
        policies["Ours (IndexPolicy)"] = IndexPolicy(
            **components,
            dataset=test_ds,
            device=device,
        )

    sf = SuccessFirstFixedOrder()
    sf.fit(train_ds)
    policies["SuccessFirst"] = sf

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.save_ordering(output_dir / "success_first_ordering.json")
        logger.info("Saved SuccessFirst ordering to %s", output_dir / "success_first_ordering.json")

    policies["ShortestFirst"] = ShortestFirstFixedOrder(test_ds.skeleton_lengths)

    # -- Evaluate across seeds --------------------------------------------
    # All policies here are deterministic; running N seeds verifies this.
    all_metrics: dict[str, list[EvalMetrics]] = {name: [] for name in policies}

    for seed in range(n_seeds):
        evaluator = OfflineEvaluator(test_ds)
        for name, policy in policies.items():
            metrics = evaluator.evaluate(policy)
            all_metrics[name].append(metrics)

    # -- Sanity checks ----------------------------------------------------
    issues: list[str] = []

    # Check 1: Oracle TTFS ≤ every other method
    oracle_ttfs = all_metrics["Oracle"][0].mean_ttfs
    for name, runs in all_metrics.items():
        if name == "Oracle":
            continue
        other_ttfs = runs[0].mean_ttfs
        if oracle_ttfs > other_ttfs + 1e-6:
            msg = f"SANITY FAIL: Oracle mean TTFS ({oracle_ttfs:.4f}) > {name} ({other_ttfs:.4f})"
            issues.append(msg)
            logger.error(msg)

    # Check 2: Oracle success@1 == feasibility ceiling
    feasibility = compute_feasibility_fraction(test_ds)
    oracle_s1 = all_metrics["Oracle"][0].success_at_k[1]
    if abs(oracle_s1 - feasibility) > 1e-6:
        msg = (
            f"SANITY FAIL: Oracle success@1 ({oracle_s1:.4f}) != "
            f"feasibility ceiling ({feasibility:.4f})"
        )
        issues.append(msg)
        logger.error(msg)

    # Check 3: Deterministic baselines identical across seeds
    for name in ["Oracle", "SuccessFirst", "ShortestFirst"]:
        if name not in all_metrics:
            continue
        runs = all_metrics[name]
        first_ttfs = [
            r.ttfs for r in runs[0].per_instance
        ]
        for seed_idx in range(1, len(runs)):
            other_ttfs_list = [
                r.ttfs for r in runs[seed_idx].per_instance
            ]
            if first_ttfs != other_ttfs_list:
                msg = f"SANITY FAIL: {name} produced different results across seeds 0 and {seed_idx}"
                issues.append(msg)
                logger.error(msg)

    # Also check IndexPolicy determinism
    if "Ours (IndexPolicy)" in all_metrics:
        runs = all_metrics["Ours (IndexPolicy)"]
        first_ttfs = [r.ttfs for r in runs[0].per_instance]
        for seed_idx in range(1, len(runs)):
            other_ttfs_list = [r.ttfs for r in runs[seed_idx].per_instance]
            if first_ttfs != other_ttfs_list:
                msg = f"SANITY FAIL: IndexPolicy produced different results across seeds 0 and {seed_idx}"
                issues.append(msg)
                logger.error(msg)

    # Check 4: Our method should beat both fixed-order baselines
    if "Ours (IndexPolicy)" in all_metrics:
        ours_ttfs = all_metrics["Ours (IndexPolicy)"][0].mean_ttfs
        for baseline_name in ["SuccessFirst", "ShortestFirst"]:
            baseline_ttfs = all_metrics[baseline_name][0].mean_ttfs
            if ours_ttfs > baseline_ttfs + 1e-6:
                msg = (
                    f"WARNING: IndexPolicy mean TTFS ({ours_ttfs:.4f}) > "
                    f"{baseline_name} ({baseline_ttfs:.4f}) — "
                    "belief machinery not beating this trivial baseline"
                )
                issues.append(msg)
                logger.warning(msg)

    # -- Build results table ----------------------------------------------
    # Use first seed's results (verified identical across seeds for
    # deterministic policies).
    row_order = ["Oracle", "Ours (IndexPolicy)", "SuccessFirst", "ShortestFirst"]

    lines: list[str] = []
    lines.append(
        "| Method              | Mean TTFS | 95% CI           "
        "| S@1  | S@2  | S@3  | S@5  | Frac Success |"
    )
    lines.append(
        "|---------------------|-----------|------------------"
        "|------|------|------|------|--------------|"
    )

    for name in row_order:
        if name not in all_metrics:
            continue
        m = all_metrics[name][0]

        # Bootstrap CI on TTFS (successful instances only)
        ttfs_values = [
            r.ttfs for r in m.per_instance if r.success and r.ttfs is not None
        ]
        mean, ci_lo, ci_hi = bootstrap_ci(ttfs_values)

        frac_success = m.n_succeeded / m.n_instances if m.n_instances > 0 else 0.0

        if mean < float("inf"):
            ttfs_str = f"{mean:.2f}"
            ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        else:
            ttfs_str = "inf"
            ci_str = "N/A"

        lines.append(
            f"| {name:<19s} | {ttfs_str:>9s} | {ci_str:>16s} "
            f"| {m.success_at_k.get(1, 0.0):.2f} "
            f"| {m.success_at_k.get(2, 0.0):.2f} "
            f"| {m.success_at_k.get(3, 0.0):.2f} "
            f"| {m.success_at_k.get(5, 0.0):.2f} "
            f"| {frac_success:.2f}         |"
        )

    table = "\n".join(lines)

    # -- Sanity check summary ---------------------------------------------
    sanity_lines = ["\n## Sanity Checks\n"]
    if not issues:
        sanity_lines.append("All sanity checks PASSED.")
    else:
        for issue in issues:
            sanity_lines.append(f"- {issue}")
    sanity_text = "\n".join(sanity_lines)

    full_output = f"## Comparison Results\n\n{table}\n{sanity_text}\n"

    # -- Save to disk ------------------------------------------------------
    if output_dir is not None:
        result_path = output_dir / "comparison_results.md"
        result_path.write_text(full_output)
        logger.info("Results saved to %s", result_path)

    return full_output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare skeleton selection policies on held-out test data.",
    )
    parser.add_argument("--test-data", type=Path, required=True, help="Path to test HDF5")
    parser.add_argument("--train-data", type=Path, required=True, help="Path to train HDF5")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to belief_best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device(args.device)

    logger.info("Loading test dataset: %s", args.test_data)
    test_ds = SkeletonDataset(args.test_data, preload=True)

    logger.info("Loading train dataset: %s", args.train_data)
    train_ds = SkeletonDataset(args.train_data, preload=True)

    components = None
    if args.checkpoint is not None:
        logger.info("Building model components...")
        components = build_model_components(test_ds)
        for mod in components.values():
            mod.to(device)
        load_checkpoint(components, args.checkpoint, device)

    output = run_comparison(
        test_ds=test_ds,
        train_ds=train_ds,
        components=components,
        device=device,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )

    print(output)


if __name__ == "__main__":
    main()
