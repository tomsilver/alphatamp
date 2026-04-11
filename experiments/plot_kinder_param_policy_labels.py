"""Plot per-action positive/negative ParamPolicy label counts over env steps.

Parses kinder_env_test_ablation .err files for [ParamPolicy] log lines that
report cumulative pos/neg counts per abstract action.  For each ablation,
plots cumulative pos/neg counts per action (PickFromTable, PickFromTarget,
PlaceOnTable, PlaceOnTarget, ...), averaged across trials.

Usage::

    python experiments/plot_kinder_param_policy_labels.py \
        experiments/slurm_outputs/kinder_env_test_ablation_6686213_*.err
    python experiments/plot_kinder_param_policy_labels.py \
        experiments/slurm_outputs/kinder_env_test_ablation_668621{3,4,5,6,7}_*.err \
        --save experiments/analysis/kinder_param_policy_labels.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_TASK_LABELS = {
    "0": "COMPLETE",
    "1": "PRACTICE",
    "2": "EXPLORER",
    "3": "NAIVE",
}

_TASK_ID_RE = re.compile(r"kinder_env_test_ablation_(?:\d+_)+(\d+)\.err$")
_STEP_RE = re.compile(r"CLUTTERED RETRIEVAL STEP:\s*(\d+)")
_PARAM_POLICY_RE = re.compile(
    r"\[ParamPolicy\]\s+(\w+\([^)]*\))\s+"
    r"n=(\d+)\s+\(pos=(\d+)\s+neg=(\d+)\)"
)


def _short_action_name(full_name: str) -> str:
    m = re.match(r"(\w+)\(robot,\s*(.+)\)", full_name)
    if m:
        return f"{m.group(1)}({m.group(2)})"
    return full_name


def parse_err_file(path: Path) -> dict:
    """Return {'actions': {name: {'steps': [...], 'pos': [...], 'neg': [...]}},
                'max_step': int}."""
    actions: dict[str, dict] = defaultdict(lambda: {"steps": [], "pos": [], "neg": []})
    current_step = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                continue
            m = _PARAM_POLICY_RE.search(line)
            if m:
                name = _short_action_name(m.group(1))
                pos = int(m.group(3))
                neg = int(m.group(4))
                rec = actions[name]
                rec["steps"].append(current_step)
                rec["pos"].append(pos)
                rec["neg"].append(neg)

    return {"actions": dict(actions), "max_step": current_step}


def bin_cumulative(
    steps: list[int], values: list[int], bin_size: int, max_step: int
) -> tuple[list[int], list[float]]:
    """Bin cumulative values into fixed step intervals (carry-forward)."""
    if not steps:
        return [], []
    binned_steps: list[int] = []
    binned_vals: list[float] = []
    current_bin = bin_size
    i = 0
    last_val = 0
    while current_bin <= max_step:
        while i < len(steps) and steps[i] <= current_bin:
            last_val = values[i]
            i += 1
        binned_steps.append(current_bin)
        binned_vals.append(float(last_val))
        current_bin += bin_size
    return binned_steps, binned_vals


def get_task_id(path: Path) -> str | None:
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def _average_action(
    runs: list[dict], action: str, key: str, bin_size: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    binned: list[tuple[list[int], list[float]]] = []
    for run in runs:
        if action not in run["actions"]:
            continue
        rec = run["actions"][action]
        bs, bv = bin_cumulative(rec["steps"], rec[key], bin_size, run["max_step"])
        if bs:
            binned.append((bs, bv))
    if not binned:
        return [], np.array([]), np.array([])
    min_len = min(len(bs) for bs, _ in binned)
    steps = binned[0][0][:min_len]
    arr = np.array([bv[:min_len] for _, bv in binned])
    mean = arr.mean(axis=0)
    n = len(binned)
    stderr = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return steps, mean, stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-action ParamPolicy pos/neg label counts for kinder env ablations"
    )
    parser.add_argument("err_files", nargs="+", metavar="FILE")
    parser.add_argument("--save", default=None, metavar="PATH")
    parser.add_argument("--bin-size", default=2000, type=int)
    args = parser.parse_args()

    paths = sorted(Path(f) for f in args.err_files)
    if not paths:
        print("No files provided.", file=sys.stderr)
        sys.exit(1)

    groups: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        task_id = get_task_id(path)
        if task_id is None:
            print(f"Warning: cannot determine task ID from {path.name}", file=sys.stderr)
            continue
        data = parse_err_file(path)
        if not data["actions"]:
            print(f"Warning: no ParamPolicy data in {path}", file=sys.stderr)
            continue
        groups[task_id].append(data)

    if not groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Collect all action names across all runs/ablations.
    all_actions: set[str] = set()
    for runs in groups.values():
        for run in runs:
            all_actions.update(run["actions"].keys())
    actions_sorted = sorted(all_actions)

    task_ids = sorted(groups.keys(), key=lambda t: int(t))

    if not actions_sorted:
        print("No actions found.", file=sys.stderr)
        sys.exit(1)

    figs: list[tuple[plt.Figure, str]] = []
    for task_id in task_ids:
        runs = groups[task_id]
        label = _TASK_LABELS.get(task_id, f"Task {task_id}")

        # Only plot actions that have data for this ablation and reach
        # at least 200 total (pos+neg) entries in at least one run.
        def _max_total(action: str) -> int:
            best = 0
            for run in runs:
                rec = run["actions"].get(action)
                if not rec or not rec["pos"]:
                    continue
                total = rec["pos"][-1] + rec["neg"][-1]
                if total > best:
                    best = total
            return best

        present_actions = [
            a for a in actions_sorted if _max_total(a) >= 200
        ]
        if not present_actions:
            continue

        n_cols = min(len(present_actions), 2)
        n_rows = (len(present_actions) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, action in enumerate(present_actions):
            ax = axes[idx // n_cols, idx % n_cols]
            for key, color, name in [
                ("pos", "tab:green", "Positive"),
                ("neg", "tab:red", "Negative"),
            ]:
                steps, mean, stderr = _average_action(runs, action, key, args.bin_size)
                if not steps:
                    continue
                ax.plot(steps, mean, linewidth=2, color=color, label=name)
                if np.any(stderr > 0):
                    ax.fill_between(
                        steps, mean - stderr, mean + stderr, alpha=0.2, color=color
                    )
            ax.set_title(action, fontsize=11, fontweight="bold")
            ax.set_xlabel("Env Step")
            ax.set_ylabel("Cumulative Count")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide any unused axes
        for idx in range(len(present_actions), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle(
            f"{label}: ParamPolicy Pos/Neg Labels per Action (n={len(runs)})",
            fontsize=14,
        )
        fig.tight_layout()
        figs.append((fig, label))

    if args.save:
        save_path = Path(args.save)
        stem, suffix = save_path.stem, save_path.suffix
        for fig, label in figs:
            path = save_path.with_name(f"{stem}_{label.lower()}{suffix}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
