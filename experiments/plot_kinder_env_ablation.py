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

# Regex for data rows in the new output format:
#   "   500             0.00%     0.00%       0       3            0.00%    0/10"
# Captures: step, eval_rate
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"  # step
    r"(?:\S+\s+){4}"  # skip Roll%, Overall%, Succ, Exhaus
    r"\S*\s+"  # skip [Eval] label column (empty)
    r"([\d.]+)%"  # eval Rate%
)

# Regex for final summary: "Eval  successes: 120 / 400 episodes"
_EVAL_SUMMARY_RE = re.compile(r"Eval\s+successes:\s+(\d+)\s*/\s*(\d+)\s+episodes")


def parse_out_file(path: Path) -> dict:
    """Parse a kinder_env_test_ablation .out file and return metrics."""
    steps: list[int] = []
    eval_rates: list[float] = []
    eval_total_successes: int | None = None
    eval_total_episodes: int | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                eval_rates.append(float(m.group(2)) / 100.0)
            else:
                m2 = _EVAL_SUMMARY_RE.search(line)
                if m2:
                    eval_total_successes = int(m2.group(1))
                    eval_total_episodes = int(m2.group(2))

    return {
        "steps": steps,
        "eval_rates": eval_rates,
        "eval_total_successes": eval_total_successes,
        "eval_total_episodes": eval_total_episodes,
    }


def get_task_id(path: Path) -> str | None:
    """Extract the task ID (ablation condition) from a filename."""
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def aggregate_runs(runs: list[dict]) -> dict:
    """Average eval metrics across seeds, truncating to the shortest run.

    Returns means and stderrs for eval_rates.
    """
    min_len = min(len(r["steps"]) for r in runs)
    steps = runs[0]["steps"][:min_len]

    rates = np.array([r["eval_rates"][:min_len] for r in runs], dtype=float)

    n = len(runs)
    return {
        "steps": steps,
        "rates_mean": rates.mean(axis=0) * 100,
        "rates_stderr": (
            rates.std(axis=0, ddof=1) / np.sqrt(n) * 100 if n > 1 else np.zeros(min_len)
        ),
        "n": n,
    }


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

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for task_id in sorted(groups):
        runs = groups[task_id]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")
        agg = aggregate_runs(runs)
        steps = agg["steps"]
        n = agg["n"]
        full_label = f"{label} (n={n})"

        (line,) = ax.plot(steps, agg["rates_mean"], marker="o", label=full_label)
        ax.fill_between(
            steps,
            agg["rates_mean"] - agg["rates_stderr"],
            agg["rates_mean"] + agg["rates_stderr"],
            alpha=0.2,
            color=line.get_color(),
        )

    ax.axhline(y=94, color="grey", linestyle="--", linewidth=1.5, label="ORACLE (94%)")

    ax.set_xlabel("Env step")
    ax.set_ylabel("Avg eval success rate (%)")
    ax.set_title("Eval success rate over env steps (mean ± stderr)")
    ax.set_ylim(0, 105)
    ax.legend(prop={"weight": "bold"})
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
