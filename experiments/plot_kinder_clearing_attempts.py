"""Plot attempt counts per clearing-plan step for COMPLETE vs EXPLORER.

Parses kinder_env_test_ablation .err files and extracts the final attempt
count for each step of the 4-step clearing plan:

    PkTb(ob0)|(init) → PlTb(ob0)|PkTb(ob0) → PkTb(tgt)|... → PlTg(tgt)|...

Produces a grouped bar chart with COMPLETE vs EXPLORER, individual run dots
overlaid, to show the data starvation cascade in EXPLORER.

Usage::

    python experiments/plot_kinder_clearing_attempts.py \
        experiments/slurm_outputs/kinder_env_test_ablation_672981{7,8,9}_*.err \
        experiments/slurm_outputs/kinder_env_test_ablation_672982{0,3}_*.err \
        --save experiments/analysis/kinder_clearing_attempts.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ablation labels by task ID
_TASK_LABELS = {
    "0": "COMPLETE",
    "2": "EXPLORER",
}

# The 4 steps of the clearing plan, as (action_regex, history_regex) pairs.
# We match on the raw log strings (before abbreviation).
_CLEARING_STEPS = [
    (
        "PickFromTable(robot, obstruction0)",
        "(initial)",
        "PkTb(ob0)\n|(init)",
    ),
    (
        "PlaceOnTable(robot, obstruction0)",
        "PickFromTable(robot, obstruction0)",
        "PlTb(ob0)\n|PkTb(ob0)",
    ),
    (
        "PickFromTable(robot, target_block)",
        "PickFromTable(robot, obstruction0) -> PlaceOnTable(robot, obstruction0)",
        "PkTb(tgt)\n|PkTb(ob0) PlTb(ob0)",
    ),
    (
        "PlaceOnTarget(robot, target_block)",
        "PickFromTable(robot, obstruction0) -> PlaceOnTable(robot, obstruction0) -> PickFromTable(robot, target_block)",
        "PlTg(tgt)\n|PkTb(ob0) PlTb(ob0)\nPkTb(tgt)",
    ),
]

# Regex to extract task ID from filename
_TASK_ID_RE = re.compile(r"kinder_env_test_ablation_(?:\d+_)+(\d+)\.err$")

# Regex for [Scorer:Action] history lines
_SCORER_HIST_RE = re.compile(
    r"\[Scorer:(\w+\([^)]*\))\]\s+"
    r"history=\[([^\]]*)\]\s+"
    r"failures=(\d+)\s+attempts=(\d+)\s+rate=([\d.nan]+)"
)


def get_task_id(path: Path) -> str | None:
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def parse_final_attempts(path: Path) -> dict[int, int]:
    """Parse .err file and return {step_index: max_attempt_count} for clearing steps.

    Returns the maximum attempt count seen for each clearing step across all
    logging rounds.  Multiple entries can share the same (action, history) but
    differ in initial state; we take the max across all of them.
    """
    max_attempts: dict[int, int] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _SCORER_HIST_RE.search(line)
            if not m:
                continue
            action = m.group(1)
            history = m.group(2).strip()
            attempts = int(m.group(4))

            for i, (step_action, step_history, _) in enumerate(_CLEARING_STEPS):
                if action == step_action and history == step_history:
                    if attempts > max_attempts.get(i, 0):
                        max_attempts[i] = attempts
                    break

    return max_attempts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clearing-plan attempt counts: COMPLETE vs EXPLORER"
    )
    parser.add_argument(
        "err_files", nargs="+", metavar="FILE",
        help="kinder_env_test_ablation_*.err files",
    )
    parser.add_argument("--save", default=None, metavar="PATH", help="Save figure")
    args = parser.parse_args()

    paths = sorted(Path(f) for f in args.err_files)

    # Group by task ID, only keep COMPLETE (0) and EXPLORER (2)
    groups: dict[str, list[dict[int, int]]] = defaultdict(list)
    for path in paths:
        task_id = get_task_id(path)
        if task_id not in ("0", "2"):
            continue
        data = parse_final_attempts(path)
        groups[task_id].append(data)

    if not groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    n_steps = len(_CLEARING_STEPS)
    step_labels = [label for _, _, label in _CLEARING_STEPS]

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.35
    x = np.arange(n_steps)
    colors = {"0": "#1F77B4", "2": "#2CA02C"}

    for offset, task_id in enumerate(["0", "2"]):
        if task_id not in groups:
            continue
        runs = groups[task_id]
        label = _TASK_LABELS[task_id]
        color = colors[task_id]

        # Collect attempt counts per step across runs (0 if step never observed)
        per_step: list[list[int]] = [[] for _ in range(n_steps)]
        for run in runs:
            for i in range(n_steps):
                per_step[i].append(run.get(i, 0))

        means = [np.mean(vals) for vals in per_step]
        positions = x - bar_width / 2 + offset * bar_width

        ax.bar(
            positions, means, bar_width,
            label=label, color=color, alpha=0.8, edgecolor="white",
        )

        # Overlay individual run dots — spread duplicates horizontally
        for i in range(n_steps):
            vals = sorted(per_step[i])
            # Count occurrences of each value
            counts: dict[int, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            # Place each dot, centering groups of duplicates on the bar
            placed: dict[int, int] = {}
            x_final, y_final = [], []
            for v in vals:
                idx = placed.get(v, 0)
                n_dup = counts[v]
                spread = (idx - (n_dup - 1) / 2) * 0.04
                x_final.append(positions[i] + spread)
                y_final.append(v)
                placed[v] = idx + 1
            ax.scatter(
                x_final, y_final,
                color="black", s=30, zorder=5, alpha=0.7,
                edgecolors="white", linewidths=0.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Final Attempt Count (window)")
    ax.set_title("Clearing Plan: Attempts per Step (COMPLETE vs EXPLORER)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Draw a reference line at window size = 50
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="_window=50")
    ax.annotate("window=50", xy=(n_steps - 0.5, 51), fontsize=8, color="gray")

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
