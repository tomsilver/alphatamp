"""Plot per-abstract-action success rates over env steps for kinder env ablations.

Parses kinder_env_test_ablation .err files for [Scorer:Action] log lines that
report per-history-key failures/attempts for each abstract action.  Each unique
(action, history) pair is tracked as a separate series.  Computes rolling success
rates and plots one subplot per ablation.

Usage::

    python experiments/plot_kinder_action_rates.py \
        experiments/slurm_outputs/kinder_env_test_ablation_6636485_*.err
    python experiments/plot_kinder_action_rates.py \
        experiments/slurm_outputs/kinder_env_test_ablation_663648{5,6,7,8,9}_*.err \
        --save experiments/analysis/kinder_action_rates.png
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

# Regex for [Scorer:Action] history lines:
#   [Scorer:PlaceOnTarget(robot, target_block)] history=[(initial)] failures=5 attempts=10 rate=0.5000
_SCORER_HIST_RE = re.compile(
    r"\[Scorer:(\w+\([^)]*\))\]\s+"
    r"history=\[([^\]]*)\]\s+"
    r"failures=(\d+)\s+attempts=(\d+)\s+rate=([\d.nan]+)"
)


_OP_ABBREV = {
    "PickFromTable": "PkTb",
    "PickFromTarget": "PkTg",
    "PlaceOnTable": "PlTb",
    "PlaceOnTarget": "PlTg",
    "PickObstruction": "PkOb",
    "PlaceObstruction": "PlOb",
}

_OBJ_ABBREV = {
    "target_block": "tgt",
    "obstruction0": "ob0",
    "obstruction1": "ob1",
    "obstruction2": "ob2",
}


def _short_action_name(full_name: str) -> str:
    """Shorten e.g. 'PlaceOnTarget(robot, target_block)' to 'PlTg(tgt)'."""
    m = re.match(r"(\w+)\(robot,\s*(.+)\)", full_name)
    if m:
        op = _OP_ABBREV.get(m.group(1), m.group(1))
        obj = _OBJ_ABBREV.get(m.group(2).strip(), m.group(2).strip())
        return f"{op}({obj})"
    return full_name


def _short_history(history_str: str) -> str:
    """Shorten a history string for compact legend labels.

    E.g. 'PickFromTable(robot, obstruction0) -> PlaceOnTarget(robot, obstruction0)'
    becomes 'PkTb(ob0) PlTg(ob0)'.
    """
    if history_str.strip() == "(initial)":
        return "(init)"
    parts = [p.strip() for p in history_str.split("->")]
    short_parts = []
    for p in parts:
        m = re.match(r"(\w+)\(robot,\s*(.+)\)", p)
        if m:
            op = _OP_ABBREV.get(m.group(1), m.group(1))
            obj = _OBJ_ABBREV.get(m.group(2).strip(), m.group(2).strip())
            short_parts.append(f"{op}({obj})")
        else:
            short_parts.append(p)
    return " ".join(short_parts)


def parse_err_file(path: Path) -> dict[str, dict]:
    """Parse .err file for per-(action, history) failures/attempts over steps.

    Returns {series_key: {"steps": [...], "failures": [...], "attempts": [...],
                          "action": str, "history": str}}.
    series_key = "action | history" for unique identification.
    Only records when attempts changes for a given series.
    """
    series: dict[str, dict] = defaultdict(
        lambda: {
            "steps": [], "success_rates": [],
            "action": "", "history": "",
        }
    )
    current_step = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                continue

            m = _SCORER_HIST_RE.search(line)
            if m:
                action = _short_action_name(m.group(1))
                history = m.group(2).strip()
                failures = int(m.group(3))
                attempts = int(m.group(4))
                rate_str = m.group(5)
                if rate_str == "nan" or attempts == 0:
                    continue
                # Logged rate is failure rate; convert to success rate
                success_rate = 1.0 - failures / attempts
                key = f"{action} | {_short_history(history)}"
                rec = series[key]
                rec["action"] = action
                rec["history"] = history
                rec["steps"].append(current_step)
                rec["success_rates"].append(success_rate)

    return {"actions": dict(series), "max_step": current_step}


def get_task_id(path: Path) -> str | None:
    m = _TASK_ID_RE.search(path.name)
    return m.group(1) if m else None


def bin_success_rates(
    steps: list[int],
    success_rates: list[float],
    bin_size: int = 2000,
    max_step: int | None = None,
) -> tuple[list[int], list[float]]:
    """Bin pre-computed success rates into fixed step intervals.

    Within each bin, the success rate is the average of all observations that
    fall into that bin.  If *max_step* is provided, the last known rate is
    carried forward to fill bins up to that step.
    """
    if not steps:
        return [], []

    # Group observations into bins and average
    binned_steps: list[int] = []
    binned_rates: list[float] = []
    current_bin = bin_size
    bin_vals: list[float] = []
    last_rate = success_rates[0]

    for s, rate in zip(steps, success_rates):
        while s >= current_bin:
            if bin_vals:
                last_rate = sum(bin_vals) / len(bin_vals)
                bin_vals = []
            binned_steps.append(current_bin)
            binned_rates.append(last_rate)
            current_bin += bin_size
        bin_vals.append(rate)

    # Flush remaining values
    if bin_vals:
        last_rate = sum(bin_vals) / len(bin_vals)

    # Carry forward the last rate to max_step
    if max_step is not None:
        while current_bin <= max_step:
            binned_steps.append(current_bin)
            binned_rates.append(last_rate)
            current_bin += bin_size

    return binned_steps, binned_rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-abstract-action success rates for kinder env ablations"
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
            print(f"Warning: no scorer data in {path}", file=sys.stderr)
            continue
        groups[task_id].append(data)

    if not groups:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    task_ids = sorted(groups)
    n_tasks = len(task_ids)

    # Only plot the 4-step clearing plan series.
    _ALLOWED_SERIES = {
        "PkTb(ob0) | (init)",
        "PlTb(ob0) | PkTb(ob0)",
        "PkTb(tgt) | PkTb(ob0) PlTb(ob0)",
        "PlTg(tgt) | PkTb(ob0) PlTb(ob0) PkTb(tgt)",
    }

    # Helper: plot a set of actions onto an axis.
    # Each run is drawn as its own bin-averaged line (no cross-run averaging)
    # so per-run noise remains visible.
    def _plot_actions(ax, runs, actions, bin_size):
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        action_colors: dict[str, str] = {}
        for i, action in enumerate(sorted(actions)):
            action_colors[action] = color_cycle[i % len(color_cycle)]

        # Find the global max step across all runs for consistent bin edges
        global_max_step = max(run["max_step"] for run in runs) if runs else 0

        for action in sorted(actions):
            color = action_colors[action]

            # Collect binned rates from each run, aligned to common bin edges
            all_binned: list[list[float]] = []
            common_steps: list[int] | None = None
            for run in runs:
                run_actions = run["actions"]
                if action not in run_actions:
                    continue
                rec = run_actions[action]
                bs, br = bin_success_rates(
                    rec["steps"], rec["success_rates"],
                    bin_size=bin_size,
                    max_step=global_max_step,
                )
                if not bs:
                    continue
                all_binned.append([r * 100 for r in br])
                if common_steps is None:
                    common_steps = bs

            if not all_binned or common_steps is None:
                continue

            # Pad shorter runs with NaN so all arrays have the same length
            max_len = len(common_steps)
            for i in range(len(all_binned)):
                if len(all_binned[i]) < max_len:
                    all_binned[i].extend(
                        [float("nan")] * (max_len - len(all_binned[i]))
                    )

            arr = np.array(all_binned)  # (n_runs, n_bins)
            mean = np.nanmean(arr, axis=0)
            stderr = np.nanstd(arr, axis=0) / np.sqrt(
                np.sum(~np.isnan(arr), axis=0)
            )

            ax.plot(
                common_steps, mean,
                linewidth=1.5, color=color, label=action,
            )
            ax.fill_between(
                common_steps, mean - stderr, mean + stderr,
                alpha=0.2, color=color,
            )

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

            # Collect all series keys across runs, filtered to allowed set
            all_keys: set[str] = set()
            for run in runs:
                all_keys.update(k for k in run["actions"] if k in _ALLOWED_SERIES)

            pick_keys = {k for k in all_keys if k.startswith("Pk")}
            place_keys = {k for k in all_keys if k.startswith("Pl")}

            _plot_actions(ax_pick, runs, pick_keys, args.bin_size)
            _plot_actions(ax_place, runs, place_keys, args.bin_size)

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
            f"{fig_title}: Per-Action Success Rates (rolling window)",
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
