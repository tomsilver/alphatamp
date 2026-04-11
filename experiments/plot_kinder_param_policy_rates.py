"""Plot per-action parameter policy success rates over env steps.

Parses kinder_env_test_ablation .err files for [ParamPolicy] log lines that
report cumulative pos/neg counts for each abstract action.  Computes rolling
success rates from the deltas between consecutive entries and plots one subplot
per ablation, with each abstract action as a separate line.

Usage::

    python experiments/plot_kinder_param_policy_rates.py \
        experiments/slurm_outputs/kinder_env_test_ablation_6686213_*.err
    python experiments/plot_kinder_param_policy_rates.py \
        experiments/slurm_outputs/kinder_env_test_ablation_668621{3,4,5,6,7}_*.err \
        --save experiments/analysis/kinder_param_policy_rates.png
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
    "1": "PRACTICE",
    "2": "EXPLORER",
    "3": "NAIVE",
}

# Regex to extract task ID from filename
_TASK_ID_RE = re.compile(r"kinder_env_test_ablation_(?:\d+_)+(\d+)\.err$")

# Regex for step marker
_STEP_RE = re.compile(r"CLUTTERED RETRIEVAL STEP:\s*(\d+)")

# Regex for [ParamPolicy] lines:
#   [ParamPolicy] PlaceOnTarget(robot, target_block) n=10 (pos=0 neg=10) iters=217 loss: 0.9535 -> 0.0073
_PARAM_POLICY_RE = re.compile(
    r"\[ParamPolicy\]\s+(\w+\([^)]*\))\s+"
    r"n=(\d+)\s+\(pos=(\d+)\s+neg=(\d+)\)"
)


def _short_action_name(full_name: str) -> str:
    """Shorten e.g. 'PlaceOnTarget(robot, target_block)' to 'PlaceOnTarget(target_block)'."""
    m = re.match(r"(\w+)\(robot,\s*(.+)\)", full_name)
    if m:
        return f"{m.group(1)}({m.group(2)})"
    return full_name


def parse_err_file(path: Path) -> dict:
    """Parse .err file for per-action cumulative pos/neg over steps.

    Returns {"actions": {action_name: {"steps": [...], "pos": [...], "neg": [...]}},
             "max_step": int}.
    """
    actions: dict[str, dict] = defaultdict(
        lambda: {"steps": [], "pos": [], "neg": [], "_prev_n": -1}
    )
    current_step = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                continue

            m = _PARAM_POLICY_RE.search(line)
            if m:
                action = _short_action_name(m.group(1))
                n = int(m.group(2))
                pos = int(m.group(3))
                neg = int(m.group(4))
                rec = actions[action]
                if n != rec["_prev_n"] and n > 0:
                    rec["steps"].append(current_step)
                    rec["pos"].append(pos)
                    rec["neg"].append(neg)
                    rec["_prev_n"] = n

    for rec in actions.values():
        del rec["_prev_n"]

    return {"actions": dict(actions), "max_step": current_step}


def bin_success_rates(
    steps: list[int],
    pos: list[int],
    neg: list[int],
    bin_size: int = 2000,
    window: int = 20,
    max_step: int | None = None,
) -> tuple[list[int], list[float]]:
    """Compute rolling success rate from cumulative pos/neg, binned by step."""
    if not steps:
        return [], []

    # Reconstruct individual attempt outcomes from cumulative deltas
    events: list[tuple[int, bool]] = []
    prev_pos, prev_neg = 0, 0
    for s, p, n in zip(steps, pos, neg):
        dp = p - prev_pos
        dn = n - prev_neg
        for _ in range(dp):
            events.append((s, True))
        for _ in range(dn):
            events.append((s, False))
        prev_pos, prev_neg = p, n

    if not events:
        return [], []

    # Rolling success rate
    rolling: list[tuple[int, float]] = []
    for i, (s, _) in enumerate(events):
        win_start = max(0, i + 1 - window)
        win = events[win_start : i + 1]
        rate = sum(1 for _, ok in win if ok) / len(win)
        rolling.append((s, rate))

    # Bin into fixed step intervals
    binned_steps: list[int] = []
    binned_rates: list[float] = []
    current_bin = bin_size
    last_rate = rolling[0][1]
    for s, rate in rolling:
        last_rate = rate
        while s >= current_bin:
            binned_steps.append(current_bin)
            binned_rates.append(last_rate)
            current_bin += bin_size

    # Carry forward the last rate to max_step
    if max_step is not None:
        while current_bin <= max_step:
            binned_steps.append(current_bin)
            binned_rates.append(last_rate)
            current_bin += bin_size

    return binned_steps, binned_rates


def get_task_id(path: Path) -> str | None:
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def _plot_actions(ax, runs, actions, bin_size):
    for action in sorted(actions):
        binned_runs: list[dict] = []
        for run in runs:
            run_actions = run["actions"]
            if action not in run_actions:
                continue
            rec = run_actions[action]
            bs, br = bin_success_rates(
                rec["steps"], rec["pos"], rec["neg"],
                bin_size=bin_size,
                max_step=run["max_step"],
            )
            if bs:
                binned_runs.append({"steps": bs, "rates": br})

        if not binned_runs:
            continue

        min_len = min(len(b["steps"]) for b in binned_runs)
        if min_len == 0:
            continue

        steps = binned_runs[0]["steps"][:min_len]
        arr = np.array([b["rates"][:min_len] for b in binned_runs])
        mean = arr.mean(axis=0) * 100
        n_runs = len(binned_runs)
        stderr = (
            arr.std(axis=0, ddof=1) / np.sqrt(n_runs) * 100
            if n_runs > 1
            else np.zeros_like(mean)
        )

        (line,) = ax.plot(steps, mean, linewidth=2, label=action)
        if n_runs > 1:
            ax.fill_between(
                steps, mean - stderr, mean + stderr,
                alpha=0.15, color=line.get_color(),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-action parameter policy success rates for kinder env ablations"
    )
    parser.add_argument(
        "err_files", nargs="+", metavar="FILE",
        help="kinder_env_test_ablation_*.err files",
    )
    parser.add_argument("--save", default=None, metavar="PATH", help="Save figure")
    parser.add_argument(
        "--bin-size", default=2000, type=int,
        help="Step bin size for averaging (default: 2000)",
    )
    args = parser.parse_args()

    paths = sorted(Path(f) for f in args.err_files)
    if not paths:
        print("No files provided.", file=sys.stderr)
        sys.exit(1)

    # Group by task ID
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        task_id = get_task_id(path)
        if task_id is None:
            print(
                f"Warning: cannot determine task ID from {path.name}, skipping",
                file=sys.stderr,
            )
            continue
        data = parse_err_file(path)
        if not data["actions"]:
            print(f"Warning: no ParamPolicy data in {path}", file=sys.stderr)
            continue
        groups[task_id].append(data)

    if not groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Split into two figures:
    #   fig1: COMPLETE + EXPLORER (task IDs 0, 2)
    #   fig2: PRACTICE + NAIVE (task IDs 1, 3)
    _FIG_GROUPS = [
        ("COMPLETE & EXPLORER", ["0", "2"]),
        ("PRACTICE & NAIVE", ["1", "3"]),
    ]

    figs: list[tuple[plt.Figure, str]] = []
    for fig_title, fig_task_ids in _FIG_GROUPS:
        present = [t for t in fig_task_ids if t in groups]
        if not present:
            continue
        n_rows = len(present)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(14, 5 * n_rows), squeeze=False,
        )

        for row, task_id in enumerate(present):
            ax_pick = axes[row, 0]
            ax_place = axes[row, 1]
            label = _TASK_LABELS.get(task_id, f"Task {task_id}")
            runs = groups[task_id]
            n = len(runs)

            # Collect all action names across runs
            all_actions: set[str] = set()
            for run in runs:
                all_actions.update(run["actions"].keys())

            pick_actions = {a for a in all_actions if a.startswith("Pick")}
            place_actions = {a for a in all_actions if a.startswith("Place")}

            _plot_actions(ax_pick, runs, pick_actions, args.bin_size)
            _plot_actions(ax_place, runs, place_actions, args.bin_size)

            for ax in (ax_pick, ax_place):
                ax.set_xlabel("Env Step")
                ax.set_ylabel("Success Rate (%)")
                ax.set_ylim(-5, 105)
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.3)

            ax_pick.set_title(
                f"{label} — Pick actions (n={n})", fontweight="bold"
            )
            ax_place.set_title(
                f"{label} — Place actions (n={n})", fontweight="bold"
            )

        fig.suptitle(
            f"{fig_title}: Parameter Policy Success Rates (rolling window)",
            fontsize=14,
        )
        fig.tight_layout()
        figs.append((fig, fig_title))

    if args.save:
        save_path = Path(args.save)
        stem, suffix = save_path.stem, save_path.suffix
        suffixes = {"COMPLETE & EXPLORER": "_explore", "PRACTICE & NAIVE": "_baseline"}
        for fig, fig_title in figs:
            tag = suffixes.get(fig_title, "")
            path = save_path.with_name(f"{stem}{tag}{suffix}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
