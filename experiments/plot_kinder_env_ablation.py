"""Plot kinder env ablation results from SLURM output files.

Reads kinder_env_test_ablation_<jobid>_<taskid>.out files, groups by task ID
(ablation condition), averages eval success rates across seeds, and plots
with stderr error bars.

Usage::

    python experiments/plot_kinder_env_ablation.py \
        experiments/slurm_outputs/kinder_env_test_ablation_6216633_*.out \
        experiments/slurm_outputs/kinder_env_test_ablation_6256835_*.out
    python experiments/plot_kinder_env_ablation.py \
        experiments/slurm_outputs/kinder_env_test_ablation_*.out \
        --save experiments/analysis/kinder_ablation.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ablation labels by task ID (matches kinder_env_test_ablation.slurm)
_TASK_LABELS = {
    "0": "COMPLETE",
    "1": "PRACTICE",
    "2": "EXPLORER",
    "3": "NAIVE",
}

# Regex to extract task ID from filename like
# kinder_env_test_ablation_6216633_0.out
_TASK_ID_RE = re.compile(r"kinder_env_test_ablation_(?:\d+_)+(\d+)\.out$")

# Regex for data rows.  Captures: step, overall%, eval_rate%, and optionally clear%
# Example row (with Clear%):
#   "  2000             0.00%     0.00%       0      10            0.00%    0/20     0.00%"
# Example row (without Clear%):
#   "   500             0.00%     0.00%       0       3            0.00%    0/10"
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"          # step
    r"\S*\s+"                # skip Roll% (or empty [Train] label)
    r"([\d.]+)%\s+"          # Roll%
    r"([\d.]+)%\s+"          # Overall%
    r"\d+\s+"                # Succ
    r"\d+\s+"                # Exhaus
    r"\S*\s+"                # skip [Eval] label column (empty)
    r"([\d.]+)%\s+"          # eval Rate%
    r"\d+/\d+"               # Succ/N
    r"(?:\s+([\d.]+)%)?"     # optional Clear%
)

# Regex for final summary: "Eval  successes: 120 / 400 episodes"
_EVAL_SUMMARY_RE = re.compile(r"Eval\s+successes:\s+(\d+)\s*/\s*(\d+)\s+episodes")


def parse_out_file(path: Path) -> dict:
    """Parse a kinder_env_test_ablation .out file and return metrics."""
    steps: list[int] = []
    overall_rates: list[float] = []
    eval_rates: list[float] = []
    clear_rates: list[float] = []
    eval_total_successes: int | None = None
    eval_total_episodes: int | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                overall_rates.append(float(m.group(3)) / 100.0)
                eval_rates.append(float(m.group(4)) / 100.0)
                if m.group(5) is not None:
                    clear_rates.append(float(m.group(5)) / 100.0)
            else:
                m2 = _EVAL_SUMMARY_RE.search(line)
                if m2:
                    eval_total_successes = int(m2.group(1))
                    eval_total_episodes = int(m2.group(2))

    return {
        "steps": steps,
        "overall_rates": overall_rates,
        "eval_rates": eval_rates,
        "clear_rates": clear_rates,
        "eval_total_successes": eval_total_successes,
        "eval_total_episodes": eval_total_episodes,
    }


def get_task_id(path: Path) -> str | None:
    """Extract the task ID (ablation condition) from a filename."""
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def aggregate_runs(runs: list[dict]) -> dict:
    """Average metrics across seeds, truncating to the shortest run.

    Returns means and stderrs for overall_rates, eval_rates, and clear_rates.
    """
    min_len = min(len(r["steps"]) for r in runs)
    steps = runs[0]["steps"][:min_len]

    overall = np.array([r["overall_rates"][:min_len] for r in runs], dtype=float)
    eval_r = np.array([r["eval_rates"][:min_len] for r in runs], dtype=float)

    # Clear rates may not be present in all runs
    runs_with_clear = [r for r in runs if r["clear_rates"]]
    has_clear = len(runs_with_clear) > 0
    if has_clear:
        clear_min_len = min(len(r["clear_rates"]) for r in runs_with_clear)
        clear_min_len = min(clear_min_len, min_len)
        clear = np.array(
            [r["clear_rates"][:clear_min_len] for r in runs_with_clear], dtype=float
        )
        n_clear = len(runs_with_clear)
    else:
        clear_min_len = 0
        clear = np.array([])
        n_clear = 0

    n = len(runs)
    result: dict = {
        "steps": steps,
        "overall_mean": overall.mean(axis=0) * 100,
        "overall_stderr": (
            overall.std(axis=0, ddof=1) / np.sqrt(n) * 100
            if n > 1
            else np.zeros(min_len)
        ),
        "eval_mean": eval_r.mean(axis=0) * 100,
        "eval_stderr": (
            eval_r.std(axis=0, ddof=1) / np.sqrt(n) * 100
            if n > 1
            else np.zeros(min_len)
        ),
        "n": n,
        "has_clear": has_clear,
    }
    if has_clear:
        result["clear_steps"] = steps[:clear_min_len]
        result["clear_mean"] = clear.mean(axis=0) * 100
        result["clear_stderr"] = (
            clear.std(axis=0, ddof=1) / np.sqrt(n_clear) * 100
            if n_clear > 1
            else np.zeros(clear_min_len)
        )
        result["n_clear"] = n_clear
    return result


def main() -> None:
    """Main plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot kinder env ablation results (eval success rates)"
    )
    parser.add_argument(
        "out_files",
        nargs="+",
        metavar="FILE",
        help="kinder_env_test_ablation_*.out files to plot",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save figure to PATH instead of displaying it",
    )
    parser.add_argument(
        "--oracle",
        default=None,
        type=float,
        metavar="PCT",
        help="Oracle success rate (%%) to show as horizontal reference line",
    )
    args = parser.parse_args()

    paths = sorted(Path(f) for f in args.out_files)
    if not paths:
        print("No files provided.", file=sys.stderr)
        sys.exit(1)

    # Group parsed runs by task ID
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        task_id = get_task_id(path)
        if task_id is None:
            print(
                f"Warning: cannot determine task ID from {path.name}, skipping",
                file=sys.stderr,
            )
            continue
        data = parse_out_file(path)
        if not data["steps"]:
            print(f"Warning: no data rows found in {path}", file=sys.stderr)
            continue
        groups[task_id].append(data)

    if not groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Figure 1: training overall + eval success rates (side by side)
    fig1, (ax1_overall, ax1_eval) = plt.subplots(1, 2, figsize=(14, 5))
    # Figure 2: clear rates
    fig2, ax2_clear = plt.subplots(1, 1, figsize=(7, 5))
    # Figure 3: per-ablation overlays (COMPLETE, EXPLORER)
    fig3, (ax2_complete, ax2_explorer) = plt.subplots(1, 2, figsize=(14, 5))

    any_clear = False
    # Store aggregated data per task for the overlay plot
    agg_by_task: dict[str, tuple[str, dict, str]] = {}

    for task_id in sorted(groups):
        runs = groups[task_id]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")
        agg = aggregate_runs(runs)
        steps = agg["steps"]
        n = agg["n"]
        full_label = f"{label} (n={n})"

        # Training overall success rate
        (line,) = ax1_overall.plot(
            steps, agg["overall_mean"], marker="o", label=full_label
        )
        color = line.get_color()
        ax1_overall.fill_between(
            steps,
            agg["overall_mean"] - agg["overall_stderr"],
            agg["overall_mean"] + agg["overall_stderr"],
            alpha=0.2,
            color=color,
        )

        # Eval success rate
        ax1_eval.plot(
            steps, agg["eval_mean"], marker="o", label=full_label, color=color
        )
        ax1_eval.fill_between(
            steps,
            agg["eval_mean"] - agg["eval_stderr"],
            agg["eval_mean"] + agg["eval_stderr"],
            alpha=0.2,
            color=color,
        )

        agg_by_task[task_id] = (label, agg, color)

        # Clear rate (if available)
        if agg["has_clear"]:
            any_clear = True
            n_c = agg["n_clear"]
            clear_label = f"{label} (n={n_c})"
            ax2_clear.plot(
                agg["clear_steps"],
                agg["clear_mean"],
                marker="s",
                label=clear_label,
                color=color,
            )
            ax2_clear.fill_between(
                agg["clear_steps"],
                agg["clear_mean"] - agg["clear_stderr"],
                agg["clear_mean"] + agg["clear_stderr"],
                alpha=0.2,
                color=color,
            )

    # --- Overlay subplots: one each for COMPLETE and EXPLORER ---
    _OVERLAY_AXES = {"0": ax2_complete, "2": ax2_explorer}
    for task_id, ax_ov in _OVERLAY_AXES.items():
        if task_id not in agg_by_task:
            continue
        label, agg, color = agg_by_task[task_id]
        if not agg["has_clear"]:
            continue
        clear_len = len(agg["clear_steps"])
        # Eval success rate (solid)
        ax_ov.plot(
            agg["steps"][:clear_len],
            agg["eval_mean"][:clear_len],
            marker="o",
            color=color,
            linestyle="-",
            label="Eval success",
        )
        ax_ov.fill_between(
            agg["steps"][:clear_len],
            agg["eval_mean"][:clear_len] - agg["eval_stderr"][:clear_len],
            agg["eval_mean"][:clear_len] + agg["eval_stderr"][:clear_len],
            alpha=0.15,
            color=color,
        )
        # Clear rate (dashed)
        ax_ov.plot(
            agg["clear_steps"],
            agg["clear_mean"],
            marker="s",
            color=color,
            linestyle="--",
            label="Clear rate",
        )
        ax_ov.fill_between(
            agg["clear_steps"],
            agg["clear_mean"] - agg["clear_stderr"],
            agg["clear_mean"] + agg["clear_stderr"],
            alpha=0.15,
            color=color,
        )

    # --- Figure 1: training overall + eval success rate ---
    for ax, title_prefix in [
        (ax1_overall, "Average overall training success rate"),
        (ax1_eval, "Average overall evaluation success rate"),
    ]:
        if args.oracle is not None:
            ax.axhline(
                y=args.oracle, color="grey", linestyle="--", linewidth=1.5,
                label=f"ORACLE ({args.oracle:.0f}%)",
            )
        ax.set_xlabel("Env step")
        ax.set_ylabel(f"{title_prefix} (%)")
        ax.set_title(f"{title_prefix} (mean ± stderr)")
        ax.set_ylim(0, 105)
        ax.legend(prop={"weight": "bold"})
        ax.grid(True, alpha=0.3)
    ax1_overall.text(
        0.5, -0.15, "(a)", transform=ax1_overall.transAxes, ha="center", fontsize=14
    )
    ax1_eval.text(
        0.5, -0.15, "(b)", transform=ax1_eval.transAxes, ha="center", fontsize=14
    )
    fig1.tight_layout()

    # --- Figure 2: clear rate ---
    ax2_clear.set_xlabel("Env step")
    ax2_clear.set_ylabel("Eval clear rate (%)")
    ax2_clear.set_title("Eval clear rate (mean ± stderr)")
    ax2_clear.set_ylim(0, 105)
    ax2_clear.legend(prop={"weight": "bold"})
    ax2_clear.grid(True, alpha=0.3)
    fig2.tight_layout()

    # --- Figure 3: per-ablation overlays ---
    for ax_ov, panel_label, title_label in [
        (ax2_complete, "(a)", "COMPLETE"),
        (ax2_explorer, "(b)", "EXPLORER"),
    ]:
        ax_ov.set_xlabel("Env step")
        ax_ov.set_ylabel("Rate (%)")
        ax_ov.set_title(
            f"{title_label}: eval success (solid) vs\nclear rate (dashed)"
        )
        ax_ov.set_ylim(0, 105)
        ax_ov.legend(prop={"weight": "bold"})
        ax_ov.grid(True, alpha=0.3)
        ax_ov.text(
            0.5, -0.15, panel_label, transform=ax_ov.transAxes, ha="center",
            fontsize=14,
        )
    fig3.tight_layout()

    if args.save:
        save_path = Path(args.save)
        stem, suffix = save_path.stem, save_path.suffix
        path1 = save_path.with_name(f"{stem}_success{suffix}")
        path2 = save_path.with_name(f"{stem}_clear{suffix}")
        path3 = save_path.with_name(f"{stem}_overlay{suffix}")
        fig1.savefig(path1, dpi=150)
        print(f"Figure saved to: {path1}")
        if any_clear:
            fig2.savefig(path2, dpi=150)
            print(f"Figure saved to: {path2}")
            fig3.savefig(path3, dpi=150)
            print(f"Figure saved to: {path3}")
        else:
            print("No clear rate data found; skipping clear/overlay plots.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
