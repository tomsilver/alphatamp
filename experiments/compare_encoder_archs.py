"""Compare bottleneck sweep results across architectures for one environment.

Reads completed offline eval outputs from arch_*/offline_eval/ subdirectories
and produces:
- A stdout table: success rate at configured budget for each arch + baselines
- A CSV:  {env_dir}/arch_comparison.csv
- A PNG overlay: {env_dir}/arch_comparison.png  (all encoder curves + baselines)

Does NOT load pickled data, so no environment bootstrap is required.

Usage:
    python experiments/compare_encoder_archs.py --env-dir artifacts/encoder_o2
    python experiments/compare_encoder_archs.py --env-dir artifacts/encoder_o4 \
        --budget 30
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_arch_results(env_dir: Path) -> list[dict]:
    """Return list of result dicts for each arch that has a completed eval."""
    results = []
    for summary_path in sorted(
        env_dir.glob("arch_*/offline_eval/offline_encoder_eval_summary.json")
    ):
        arch_name = summary_path.parent.parent.name  # e.g. arch_half_M
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        npz_path = summary_path.parent / "offline_encoder_eval_metrics.npz"
        npz = np.load(npz_path) if npz_path.exists() else None
        results.append({"arch_name": arch_name, "summary": summary, "npz": npz})
    return results


def _sr(d: dict, key: str) -> float:
    return float(d.get(key, {}).get("success_rate", float("nan")))


def _print_table(results: list[dict]) -> None:
    col_w = 26
    header = (
        f"{'arch':<{col_w}}"
        f"{'encoder':>10}"
        f"{'baseline':>10}"
        f"{'gen-order':>11}"
        f"{'BOX':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s = r["summary"]
        enc = _sr(s, "encoder")
        base = _sr(s, "baseline")
        gen = _sr(s, "baseline_generator_order")
        box = _sr(s, "box_offline")
        name = r["arch_name"].removeprefix("arch_")
        print(
            f"{name:<{col_w}}"
            f"{enc:>10.4f}"
            f"{base:>10.4f}"
            f"{gen:>11.4f}"
            f"{box:>8.4f}"
        )


def _save_csv(results: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["arch", "encoder_sr", "baseline_sr", "generator_order_sr", "box_sr"]
        )
        for r in results:
            s = r["summary"]
            writer.writerow(
                [
                    r["arch_name"].removeprefix("arch_"),
                    _sr(s, "encoder"),
                    _sr(s, "baseline"),
                    _sr(s, "baseline_generator_order"),
                    _sr(s, "box_offline"),
                ]
            )


def _save_plot(results: list[dict], budget: float, env_dir: Path, dpi: int) -> Path:
    plt.figure(figsize=(9, 5))
    baselines_plotted = False

    for r in results:
        npz = r["npz"]
        if npz is None:
            continue
        budgets = npz["budgets"]
        encoder_curve = npz["encoder_success_curve"]
        label = r["arch_name"].removeprefix("arch_")
        plt.plot(budgets, encoder_curve, linewidth=2, label=f"encoder ({label})")

        if not baselines_plotted:
            plt.plot(
                budgets,
                npz["baseline_success_curve"],
                linewidth=2,
                linestyle="--",
                color="gray",
                label="baseline (fixed-order)",
            )
            gen_curve = npz["baseline_generator_success_curve"]
            if len(gen_curve) > 0:
                plt.plot(
                    budgets,
                    gen_curve,
                    linewidth=2,
                    linestyle=":",
                    color="dimgray",
                    label="baseline (generator-order)",
                )
            box_curve = npz["box_offline_success_curve"]
            if len(box_curve) > 0:
                plt.plot(
                    budgets,
                    box_curve,
                    linewidth=2,
                    linestyle="-.",
                    label="BOX (offline)",
                )
            baselines_plotted = True

    plt.axvline(
        budget,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=f"Budget={budget:g}s",
    )
    plt.xlabel("Time budget (seconds)")
    plt.ylabel("Success rate")
    plt.title(f"Encoder bottleneck sweep — {env_dir.name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = env_dir / "arch_comparison.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def main() -> None:
    """Load architecture eval summaries and emit comparison table/artifacts."""
    parser = argparse.ArgumentParser(
        description="Compare bottleneck sweep results for one environment."
    )
    parser.add_argument(
        "--env-dir",
        required=True,
        help="Path to env artifact directory, e.g. artifacts/encoder_o2",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget marker in seconds (default: read from first summary found)",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    env_dir = Path(args.env_dir)
    if not env_dir.exists():
        raise FileNotFoundError(f"env_dir not found: {env_dir}")

    results = _load_arch_results(env_dir)
    if not results:
        print(
            f"No completed eval results found under {env_dir}/arch_*/offline_eval/. "
            "Run launch_eval_sweep.sh first."
        )
        return

    budget = args.budget
    if budget is None:
        budget = float(results[0]["summary"].get("budget_seconds", 20.0))

    print(f"\nEnvironment: {env_dir.name}  |  Budget: {budget:g}s\n")
    _print_table(results)

    csv_path = env_dir / "arch_comparison.csv"
    _save_csv(results, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    plot_path = _save_plot(results, budget, env_dir, args.dpi)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
