"""Plot bandit ablation results from SLURM output files.

Reads bandit_ablation_<jobid>_<taskid>.out files, groups by task ID (ablation
condition), averages across seeds, and plots with stderr error bars.

Usage::

    python experiments/plot_bandit_ablation.py
        experiments/slurm_outputs/bandit_ablation_*.out
    python experiments/plot_bandit_ablation.py
        experiments/slurm_outputs/bandit_ablation_*.out
        --save experiments/analysis/ablation.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ablation labels by task ID (matches bandit_ablation.slurm)
_TASK_LABELS = {
    "0": "COMPLETE",
    "1": "PRACTICE",
    "2": "EXPLORER",
    "3": "NAIVE",
}

# Regex to extract task ID from filename like bandit_ablation_5772715_2.out
# or bandit_ablation_20_5822174_3.out (extra number segment before job ID)
_TASK_ID_RE = re.compile(r"bandit_ablation_(?:\d+_)+(\d+)\.out$")

# Regex for data rows with 4 percentage columns:
#   "  100  72.73%  72.73%  10.00%  63.64%  8  221  20"
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+(\d+)\s+(\d+)"
)

# Regex for total episodes: "  Total successes: 1249 / 1343 episodes"
_EPISODES_RE = re.compile(r"Total successes:\s+\d+\s*/\s*(\d+)\s+episodes")


def parse_out_file(path: Path) -> dict:
    """Parse a bandit_ablation .out file and return metrics."""
    steps: list[int] = []
    success_rates: list[float] = []
    eval_success_rates: list[float] = []
    widen_rates: list[float] = []
    exhaustions: list[int] = []
    total_episodes: int | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                success_rates.append(float(m.group(3)) / 100.0)
                eval_success_rates.append(float(m.group(4)) / 100.0)
                widen_rates.append(float(m.group(5)) / 100.0)
                exhaustions.append(int(m.group(8)))
            else:
                m2 = _EPISODES_RE.search(line)
                if m2:
                    total_episodes = int(m2.group(1))

    return {
        "steps": steps,
        "success_rates": success_rates,
        "eval_success_rates": eval_success_rates,
        "widen_rates": widen_rates,
        "exhaustions": exhaustions,
        "total_episodes": total_episodes,
    }


def get_task_id(path: Path) -> str | None:
    """Uses REGEX to determine the approach type."""
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def aggregate_runs(runs: list[dict]) -> dict:
    """Average metrics across seeds, truncating to the shortest run.

    Returns means and stderrs for success_rates, widen_rates, and total_episodes.
    """
    min_len = min(len(r["steps"]) for r in runs)
    steps = runs[0]["steps"][:min_len]

    rates = np.array([r["success_rates"][:min_len] for r in runs], dtype=float)
    eval_rates = np.array(
        [r["eval_success_rates"][:min_len] for r in runs], dtype=float
    )
    widens = np.array([r["widen_rates"][:min_len] for r in runs], dtype=float)

    episode_counts = [
        r["total_episodes"] for r in runs if r["total_episodes"] is not None
    ]
    ep = np.array(episode_counts, dtype=float)

    # Final exhaustion count (last value in each run)
    final_exhaustions = [r["exhaustions"][-1] for r in runs if r["exhaustions"]]
    ex = np.array(final_exhaustions, dtype=float)

    n = len(runs)
    n_ep = len(ep)
    n_ex = len(ex)
    return {
        "steps": steps,
        "rates_mean": rates.mean(axis=0) * 100,
        "rates_stderr": rates.std(axis=0, ddof=1) / np.sqrt(n) * 100,
        "eval_rates_mean": eval_rates.mean(axis=0) * 100,
        "eval_rates_stderr": eval_rates.std(axis=0, ddof=1) / np.sqrt(n) * 100,
        "widens_mean": widens.mean(axis=0) * 100,
        "widens_stderr": widens.std(axis=0, ddof=1) / np.sqrt(n) * 100,
        "episodes_mean": ep.mean() if n_ep > 0 else float("nan"),
        "episodes_stderr": (ep.std(ddof=1) / np.sqrt(n_ep) if n_ep > 1 else 0.0),
        "exhaustions_mean": ex.mean() if n_ex > 0 else float("nan"),
        "exhaustions_stderr": (ex.std(ddof=1) / np.sqrt(n_ex) if n_ex > 1 else 0.0),
        "n": n,
    }


def main() -> None:
    """Main plotting script."""
    parser = argparse.ArgumentParser(description="Plot bandit ablation results")
    parser.add_argument(
        "out_files",
        nargs="+",
        metavar="FILE",
        help="bandit_ablation_*.out files to plot",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save figure to PATH instead of displaying it",
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

    fig1, (ax1_overall, ax1_eval) = plt.subplots(1, 2, figsize=(14, 5))
    fig2, (ax2_episodes, ax2_exhaustions) = plt.subplots(1, 2, figsize=(12, 5))
    fig3, ax3_widen = plt.subplots(1, 1, figsize=(7, 5))

    bar_labels: list[str] = []
    bar_means: list[float] = []
    bar_stderrs: list[float] = []
    bar_colors: list = []
    ex_means: list[float] = []
    ex_stderrs: list[float] = []

    for task_id in sorted(groups):
        runs = groups[task_id]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")
        agg = aggregate_runs(runs)
        steps = agg["steps"]
        n = agg["n"]
        full_label = f"{label} (n=5)"

        (line,) = ax1_overall.plot(
            steps, agg["rates_mean"], marker="o", label=full_label
        )
        color = line.get_color()
        ax1_overall.fill_between(
            steps,
            agg["rates_mean"] - agg["rates_stderr"],
            agg["rates_mean"] + agg["rates_stderr"],
            alpha=0.2,
            color=color,
        )

        ax1_eval.plot(
            steps, agg["eval_rates_mean"], marker="o", label=full_label, color=color
        )
        ax1_eval.fill_between(
            steps,
            agg["eval_rates_mean"] - agg["eval_rates_stderr"],
            agg["eval_rates_mean"] + agg["eval_rates_stderr"],
            alpha=0.2,
            color=color,
        )

        ax3_widen.plot(
            steps, agg["widens_mean"], marker="s", label=full_label, color=color
        )
        ax3_widen.fill_between(
            steps,
            agg["widens_mean"] - agg["widens_stderr"],
            agg["widens_mean"] + agg["widens_stderr"],
            alpha=0.2,
            color=color,
        )

        bar_labels.append(label)
        bar_means.append(agg["episodes_mean"])
        bar_stderrs.append(agg["episodes_stderr"])
        bar_colors.append(color)
        ex_means.append(agg["exhaustions_mean"])
        ex_stderrs.append(agg["exhaustions_stderr"])

    # --- Figure 1: Average overall success rate & eval success rate ---
    for ax, title_prefix in [
        (ax1_overall, "Average overall training success rate"),
        (ax1_eval, "Average overall evaluation success rate"),
    ]:
        ax.axhline(
            y=97, color="gray", linestyle="--", linewidth=1, label="ORACLE (97%)"
        )
        ax.set_xlabel("Step")
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

    # --- Figure 2: Total episodes & resample exhaustions ---
    x = np.arange(len(bar_labels))

    ax2_episodes.bar(
        x, bar_means, yerr=bar_stderrs, color=bar_colors, capsize=5, alpha=0.8
    )
    ax2_episodes.set_xticks(x)
    ax2_episodes.set_xticklabels(
        bar_labels, rotation=15, ha="right", fontweight="bold"
    )
    ax2_episodes.set_ylabel("Total episodes completed")
    ax2_episodes.set_title("Total episodes completed (mean ± stderr)")
    ax2_episodes.grid(True, alpha=0.3, axis="y")

    ax2_exhaustions.bar(
        x, ex_means, yerr=ex_stderrs, color=bar_colors, capsize=5, alpha=0.8
    )
    ax2_exhaustions.set_xticks(x)
    ax2_exhaustions.set_xticklabels(
        bar_labels, rotation=15, ha="right", fontweight="bold"
    )
    ax2_exhaustions.set_ylabel("Total resample exhaustions")
    ax2_exhaustions.set_title("Resample exhaustions over 4000 steps (mean ± stderr)")
    ax2_exhaustions.grid(True, alpha=0.3, axis="y")

    ax2_episodes.text(
        0.5, -0.15, "(a)", transform=ax2_episodes.transAxes, ha="center", fontsize=14
    )
    ax2_exhaustions.text(
        0.5, -0.15, "(b)", transform=ax2_exhaustions.transAxes, ha="center", fontsize=14
    )
    fig2.tight_layout()

    # --- Figure 3: Widen plan fraction ---
    ax3_widen.set_xlabel("Step")
    ax3_widen.set_ylabel("Episodes with Widen plan (%)")
    ax3_widen.set_title(
        "Fraction of episodes where exploit planner "
        "chose [Widen, …] plan\n(mean ± stderr)"
    )
    ax3_widen.set_ylim(0, 105)
    ax3_widen.legend(prop={"weight": "bold"})
    ax3_widen.grid(True, alpha=0.3)
    fig3.tight_layout()

    if args.save:
        save_path = Path(args.save)
        stem, suffix = save_path.stem, save_path.suffix
        path1 = save_path.with_name(f"{stem}_success{suffix}")
        path2 = save_path.with_name(f"{stem}_episodes_exhaustions{suffix}")
        path3 = save_path.with_name(f"{stem}_widen{suffix}")
        fig1.savefig(path1, dpi=150)
        fig2.savefig(path2, dpi=150)
        fig3.savefig(path3, dpi=150)
        print(f"Figures saved to: {path1}, {path2}, {path3}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
