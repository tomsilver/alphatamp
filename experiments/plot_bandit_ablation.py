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
    "0": "Full model",
    "1": "No abstract plan scorer",
    "2": "No parameter scorer",
    "3": "No scorers",
}

# Regex to extract task ID from filename like bandit_ablation_5772715_2.out
# or bandit_ablation_20_5822174_3.out (extra number segment before job ID)
_TASK_ID_RE = re.compile(r"bandit_ablation_(?:\d+_)+(\d+)\.out$")

# Regex for data rows: "  100  25.00%  20.00%  15.00%  42  100  5"
_ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%")

# Regex for total episodes: "  Total successes: 1249 / 1343 episodes"
_EPISODES_RE = re.compile(r"Total successes:\s+\d+\s*/\s*(\d+)\s+episodes")


def parse_out_file(path: Path) -> dict:
    """Parse a bandit_ablation .out file and return metrics."""
    steps: list[int] = []
    success_rates: list[float] = []
    widen_rates: list[float] = []
    total_episodes: int | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                success_rates.append(float(m.group(3)) / 100.0)
                widen_rates.append(float(m.group(4)) / 100.0)
            else:
                m2 = _EPISODES_RE.search(line)
                if m2:
                    total_episodes = int(m2.group(1))

    return {
        "steps": steps,
        "success_rates": success_rates,
        "widen_rates": widen_rates,
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
    widens = np.array([r["widen_rates"][:min_len] for r in runs], dtype=float)

    episode_counts = [
        r["total_episodes"] for r in runs if r["total_episodes"] is not None
    ]
    ep = np.array(episode_counts, dtype=float)

    n = len(runs)
    n_ep = len(ep)
    return {
        "steps": steps,
        "rates_mean": rates.mean(axis=0) * 100,
        "rates_stderr": rates.std(axis=0, ddof=1) / np.sqrt(n) * 100,
        "widens_mean": widens.mean(axis=0) * 100,
        "widens_stderr": widens.std(axis=0, ddof=1) / np.sqrt(n) * 100,
        "episodes_mean": ep.mean() if n_ep > 0 else float("nan"),
        "episodes_stderr": (ep.std(ddof=1) / np.sqrt(n_ep) if n_ep > 1 else 0.0),
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

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    bar_labels: list[str] = []
    bar_means: list[float] = []
    bar_stderrs: list[float] = []
    bar_colors: list = []

    for task_id in sorted(groups):
        runs = groups[task_id]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")
        agg = aggregate_runs(runs)
        steps = agg["steps"]
        n = agg["n"]
        full_label = f"{label} (n=5)"

        (line,) = ax1.plot(steps, agg["rates_mean"], marker="o", label=full_label)
        ax1.fill_between(
            steps,
            agg["rates_mean"] - agg["rates_stderr"],
            agg["rates_mean"] + agg["rates_stderr"],
            alpha=0.2,
            color=line.get_color(),
        )

        ax = axes2[0]
        (line,) = ax.plot(steps, agg["widens_mean"], marker="s", label=full_label)
        ax.fill_between(
            steps,
            agg["widens_mean"] - agg["widens_stderr"],
            agg["widens_mean"] + agg["widens_stderr"],
            alpha=0.2,
            color=line.get_color(),
        )

        bar_labels.append(label)
        bar_means.append(agg["episodes_mean"])
        bar_stderrs.append(agg["episodes_stderr"])
        bar_colors.append(line.get_color())

    # --- Figure 1: Average overall success rate ---
    ax1.axhline(y=97, color="gray", linestyle="--", linewidth=1, label="Oracle (97%)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Average overall success rate (%)")
    ax1.set_title("Average overall success rate (mean ± stderr)")
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # --- Figure 2: Widen plan fraction ---
    ax = axes2[0]
    ax.set_xlabel("Step")
    ax.set_ylabel("Episodes with Widen plan (%)")
    ax.set_title(
        "Fraction of episodes where exploit planner "
        "chose [Widen, …] plan\n(mean ± stderr)"
    )
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Figure 2: Total episodes completed ---
    ax = axes2[1]
    x = np.arange(len(bar_labels))
    ax.bar(x, bar_means, yerr=bar_stderrs, color=bar_colors, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=15, ha="right")
    ax.set_ylabel("Total episodes completed")
    ax.set_title("Total episodes completed (mean ± stderr)")
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()

    if args.save:
        save_path = Path(args.save)
        stem, suffix = save_path.stem, save_path.suffix
        path1 = save_path.with_name(f"{stem}_success{suffix}")
        path2 = save_path.with_name(f"{stem}_widen_episodes{suffix}")
        fig1.savefig(path1, dpi=150)
        fig2.savefig(path2, dpi=150)
        print(f"Figures saved to: {path1}, {path2}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
