"""Plot bandit ablation results from SLURM output files.

Reads bandit_ablation_<jobid>_<taskid>.out files, groups by task ID (ablation
condition), averages across seeds, and plots with stderr error bars.

Usage::

    python experiments/plot_bandit_ablation.py experiments/slurm_outputs/bandit_ablation_*.out
    python experiments/plot_bandit_ablation.py experiments/slurm_outputs/bandit_ablation_*.out --save experiments/analysis/ablation.png
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
_TASK_ID_RE = re.compile(r"bandit_ablation_\d+_(\d+)\.out$")

# Regex for data rows: "  100  25.00%  0.00%  1  105  24"
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+\d+\s+(\d+)"
)


def parse_out_file(path: Path) -> dict:
    """Parse a bandit_ablation .out file and return metrics."""
    steps: list[int] = []
    total_successes: list[int] = []
    widen_rates: list[float] = []

    with open(path) as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                widen_rates.append(float(m.group(3)) / 100.0)
                total_successes.append(int(m.group(4)))

    return {
        "steps": steps,
        "total_successes": total_successes,
        "widen_rates": widen_rates,
    }


def get_task_id(path: Path) -> str | None:
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def aggregate_runs(runs: list[dict]) -> dict:
    """Average metrics across seeds, truncating to the shortest run.

    Returns means and stderrs for total_successes and widen_rates.
    """
    min_len = min(len(r["steps"]) for r in runs)
    steps = runs[0]["steps"][:min_len]

    successes = np.array([r["total_successes"][:min_len] for r in runs], dtype=float)
    widens = np.array([r["widen_rates"][:min_len] for r in runs], dtype=float)

    n = len(runs)
    return {
        "steps": steps,
        "successes_mean": successes.mean(axis=0),
        "successes_stderr": successes.std(axis=0, ddof=1) / np.sqrt(n),
        "widens_mean": widens.mean(axis=0) * 100,
        "widens_stderr": widens.std(axis=0, ddof=1) / np.sqrt(n) * 100,
        "n": n,
    }


def main() -> None:
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
            print(f"Warning: cannot determine task ID from {path.name}, skipping", file=sys.stderr)
            continue
        data = parse_out_file(path)
        if not data["steps"]:
            print(f"Warning: no data rows found in {path}", file=sys.stderr)
            continue
        groups[task_id].append(data)

    if not groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for task_id in sorted(groups):
        runs = groups[task_id]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")
        agg = aggregate_runs(runs)
        steps = agg["steps"]
        n = agg["n"]
        full_label = f"{label} (n={n})"

        ax = axes[0]
        line, = ax.plot(steps, agg["successes_mean"], marker="o", label=full_label)
        ax.fill_between(
            steps,
            agg["successes_mean"] - agg["successes_stderr"],
            agg["successes_mean"] + agg["successes_stderr"],
            alpha=0.2,
            color=line.get_color(),
        )

        ax = axes[1]
        line, = ax.plot(steps, agg["widens_mean"], marker="s", label=full_label)
        ax.fill_between(
            steps,
            agg["widens_mean"] - agg["widens_stderr"],
            agg["widens_mean"] + agg["widens_stderr"],
            alpha=0.2,
            color=line.get_color(),
        )

    # --- Total successes ---
    ax = axes[0]
    ax.set_xlabel("Step")
    ax.set_ylabel("Total successes")
    ax.set_title("Cumulative successes (mean ± stderr)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Widen plan fraction ---
    ax = axes[1]
    ax.set_xlabel("Step")
    ax.set_ylabel("Episodes with Widen plan (%)")
    ax.set_title("Fraction of episodes where exploit planner chose [Widen, …] plan\n(mean ± stderr)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
