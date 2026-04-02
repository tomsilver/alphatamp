"""Plot Reach success rate alongside Widen plan fraction over env steps.

Shows both curves on the same axes to reveal the temporal relationship:
Reach skill improvement precedes Widen plan adoption.

Parses bandit_ablation .err files for Reach success rate data and .out files
for Widen plan fraction.

Usage::

    python experiments/plot_reach_vs_widen.py \
        experiments/slurm_outputs/bandit_ablation_*.err
    python experiments/plot_reach_vs_widen.py \
        experiments/slurm_outputs/bandit_ablation_*.err \
        --save experiments/analysis/reach_vs_widen.png
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
    "0": "Full model",
    "1": "No abstract plan scorer",
    "2": "No parameter scorer",
    "3": "No scorers",
}

# Regex to extract task ID from filename
_TASK_ID_RE = re.compile(r"bandit_ablation_(?:\d+_)+(\d+)\.(?:err|out)$")

# Regex for BANDIT STEP line
_STEP_RE = re.compile(r"BANDIT STEP:\s*(\d+)")

# Regex for [Scorer-ctx] Reach(robot) with history context
_REACH_CTX_RE = re.compile(
    r"\[Scorer-ctx\] Reach\(robot\) history=(\[.*?\])\s+"
    r"predicted=([\d.]+) actual=([\d.]+) "
    r"\(failures=(\d+) attempts=(\d+)\)"
)

# Regex for data rows in .out files
_ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%")


def parse_err_file(path: Path) -> dict:
    """Parse a .err file to extract Reach cumulative failures/attempts.

    Extracts two series:
    - "no_widen": Reach attempts where history=[] (Reach-only plans, hard
      threshold) — isolates the parameter learner's true skill.
    - "with_widen": Reach attempts where history contains Widen (easy
      threshold).

    Only records a new data point when the cumulative attempts count
    actually changes (filters out repeated log lines from the same state).
    """
    no_widen_steps: list[int] = []
    no_widen_failures: list[int] = []
    no_widen_attempts: list[int] = []
    with_widen_steps: list[int] = []
    with_widen_failures: list[int] = []
    with_widen_attempts: list[int] = []
    prev_no_widen_attempts = -1
    prev_with_widen_attempts = -1
    current_step = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                continue

            m = _REACH_CTX_RE.search(line)
            if m:
                history = m.group(1)
                failures = int(m.group(4))
                attempts = int(m.group(5))
                if history == "[]":
                    if attempts != prev_no_widen_attempts:
                        no_widen_steps.append(current_step)
                        no_widen_failures.append(failures)
                        no_widen_attempts.append(attempts)
                        prev_no_widen_attempts = attempts
                else:
                    if attempts != prev_with_widen_attempts:
                        with_widen_steps.append(current_step)
                        with_widen_failures.append(failures)
                        with_widen_attempts.append(attempts)
                        prev_with_widen_attempts = attempts

    return {
        "no_widen": {
            "steps": no_widen_steps,
            "failures": no_widen_failures,
            "attempts": no_widen_attempts,
        },
        "with_widen": {
            "steps": with_widen_steps,
            "failures": with_widen_failures,
            "attempts": with_widen_attempts,
        },
    }


def parse_out_file(path: Path) -> dict:
    """Parse a .out file for widen rates and overall success rates."""
    steps: list[int] = []
    success_rates: list[float] = []
    widen_rates: list[float] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if m:
                steps.append(int(m.group(1)))
                success_rates.append(float(m.group(3)) / 100.0)
                widen_rates.append(float(m.group(4)) / 100.0)

    return {"steps": steps, "success_rates": success_rates, "widen_rates": widen_rates}


def get_task_id(path: Path) -> str | None:
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def bin_reach_rates(
    steps: list[int],
    failures: list[int],
    attempts: list[int],
    bin_size: int = 100,
    window: int = 20,
) -> tuple[list[int], list[float]]:
    """Bin rolling Reach success rates into fixed-width step bins.

    Computes a rolling success rate over the last *window* Reach attempts
    using consecutive differences of the cumulative failure/attempt counters.
    Each (step, failures, attempts) entry should already be deduplicated
    (only entries where attempts changed).
    """
    if not steps:
        return [], []

    # Reconstruct individual attempt outcomes from cumulative deltas.
    # Each entry: (global_step, success: bool)
    events: list[tuple[int, bool]] = []
    prev_f, prev_a = 0, 0
    for s, f, a in zip(steps, failures, attempts):
        df = f - prev_f
        da = a - prev_a
        # da new attempts, df of which failed
        for _ in range(df):
            events.append((s, False))
        for _ in range(da - df):
            events.append((s, True))
        prev_f, prev_a = f, a

    if not events:
        return [], []

    # Compute rolling success rate after each event
    rolling_rates: list[tuple[int, float]] = []
    for i, (s, _) in enumerate(events):
        win_start = max(0, i + 1 - window)
        win = events[win_start : i + 1]
        rate = sum(1 for _, ok in win if ok) / len(win)
        rolling_rates.append((s, rate))

    # Bin into fixed step intervals, taking the last rate in each bin
    binned_steps: list[int] = []
    binned_rates: list[float] = []
    current_bin = bin_size
    last_rate = rolling_rates[0][1]
    for s, rate in rolling_rates:
        last_rate = rate
        while s >= current_bin:
            binned_steps.append(current_bin)
            binned_rates.append(last_rate)
            current_bin += bin_size

    return binned_steps, binned_rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Reach success rate vs Widen plan fraction"
    )
    parser.add_argument(
        "err_files", nargs="+", metavar="FILE", help="bandit_ablation_*.err files"
    )
    parser.add_argument("--save", default=None, metavar="PATH", help="Save figure")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task IDs to include (default: all)",
    )
    args = parser.parse_args()

    filter_ids = set(args.task_ids.split(",")) if args.task_ids else None

    err_paths = sorted(Path(f) for f in args.err_files)

    # Group by task ID
    err_groups: dict[str, list[dict]] = defaultdict(list)
    out_groups: dict[str, list[dict]] = defaultdict(list)

    for err_path in err_paths:
        task_id = get_task_id(err_path)
        if task_id is None:
            print(
                f"Warning: cannot determine task ID from {err_path.name}",
                file=sys.stderr,
            )
            continue
        if filter_ids and task_id not in filter_ids:
            continue

        err_data = parse_err_file(err_path)
        if not err_data["no_widen"]["steps"]:
            print(
                f"Warning: no Reach (no-widen) scorer data in {err_path}",
                file=sys.stderr,
            )
            continue
        err_groups[task_id].append(err_data)

        # Try to find matching .out file
        out_path = err_path.with_suffix(".out")
        if out_path.exists():
            out_data = parse_out_file(out_path)
            if out_data["steps"]:
                out_groups[task_id].append(out_data)

    if not err_groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # One subplot per task ID
    task_ids = sorted(err_groups)
    n_tasks = len(task_ids)
    n_cols = 2
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, task_id in enumerate(task_ids):
        ax = axes[idx]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")
        runs = err_groups[task_id]
        n = len(runs)
        bin_size = 100

        # Helper to bin and aggregate a series across seeds
        def _aggregate_reach(key: str, color: str, line_label: str, ls: str = "-"):
            binned = []
            for run in runs:
                series = run[key]
                bs, br = bin_reach_rates(
                    series["steps"], series["failures"],
                    series["attempts"], bin_size,
                )
                binned.append({"steps": bs, "rates": br})
            binned = [b for b in binned if b["steps"]]
            if not binned:
                return
            ml = min(len(b["steps"]) for b in binned)
            if ml == 0:
                return
            steps = binned[0]["steps"][:ml]
            arr = np.array([b["rates"][:ml] for b in binned])
            mean = arr.mean(axis=0) * 100
            stderr = (
                arr.std(axis=0, ddof=1) / np.sqrt(len(binned)) * 100
                if len(binned) > 1
                else np.zeros_like(mean)
            )
            (line,) = ax.plot(
                steps, mean, color=color, linewidth=2,
                linestyle=ls, label=line_label,
            )
            if len(binned) > 1:
                ax.fill_between(
                    steps, mean - stderr, mean + stderr,
                    alpha=0.15, color=color,
                )

        # Plot Reach success rate WITHOUT Widen (hard threshold — true skill)
        _aggregate_reach(
            "no_widen", "tab:blue",
            "Reach success (no Widen, hard threshold)",
        )

        # Plot Reach success rate WITH Widen (easy threshold)
        _aggregate_reach(
            "with_widen", "tab:cyan",
            "Reach success (after Widen, easy threshold)",
            ls=":",
        )

        # Plot Widen adoption on same axes
        if task_id in out_groups:
            out_runs = out_groups[task_id]
            n_out = len(out_runs)
            min_out_len = min(len(r["steps"]) for r in out_runs)
            widen_steps = out_runs[0]["steps"][:min_out_len]
            widens = np.array(
                [r["widen_rates"][:min_out_len] for r in out_runs]
            )
            w_mean = widens.mean(axis=0) * 100
            w_stderr = (
                widens.std(axis=0, ddof=1) / np.sqrt(n_out) * 100
                if n_out > 1
                else np.zeros_like(w_mean)
            )

            (line_widen,) = ax.plot(
                widen_steps,
                w_mean,
                color="tab:orange",
                linewidth=2,
                linestyle="--",
                label="Widen plan fraction",
            )
            if n_out > 1:
                ax.fill_between(
                    widen_steps,
                    w_mean - w_stderr,
                    w_mean + w_stderr,
                    alpha=0.15,
                    color="tab:orange",
                )

        ax.set_xlabel("Env Step")
        ax.set_ylabel("%")
        ax.set_title(f"{label} (n={n})")
        ax.set_ylim(-5, 105)
        ax.legend(loc="center right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_tasks, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Widen Plan Adoption Influences Reach Skill Competency",
        fontsize=14,
    )
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
