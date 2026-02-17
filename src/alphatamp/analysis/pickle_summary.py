"""Summarize the collected dataset folder.

Usage:
    python src/alphatamp/analysis/pickle_summary.py datasets/
"""

import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_pickle(path: Path):
    """Load in pickle at specified path."""
    with path.open("rb") as f:
        return pickle.load(f)


def calculate_stats(entries: list[int]) -> tuple:
    """Given list of numbers, return its mean, median, mode."""
    return (
        statistics.mean(entries),
        statistics.median(entries),
        statistics.mode(entries),
    )


def summarize_dataset(dataset_dir: Path) -> None:
    """Return summary statistics of dataset at path."""
    seed_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not seed_dirs:
        print(f"No seed directories found in {dataset_dir}")
        return

    total_parameters = 0
    total_abstract_plans = 0
    total_abstract_actions = 0
    plan_successes = 0
    plan_failures = 0
    param_counts_per_action: dict[str, int] = defaultdict(int)
    param_success_per_action: dict[str, int] = defaultdict(int)
    action_counts_per_action: dict[str, int] = defaultdict(int)
    action_stats_per_action: dict[str, dict[tuple, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    seeds_loaded = 0

    for seed_dir in seed_dirs:
        param_path = seed_dir / "parameter_dataset.pkl"
        plan_path = seed_dir / "abstract_plan_dataset.pkl"
        action_path = seed_dir / "abstract_action_dataset.pkl"

        if not param_path.exists() or not plan_path.exists():
            print(f"  Skipping {seed_dir.name}: missing pickle files")
            continue

        seeds_loaded += 1

        # Parameter dataset: dict[str, list[(obs, param, label)]]
        param_data = load_pickle(param_path)
        for action_key, entries in param_data.items():
            total_parameters += len(entries)
            param_counts_per_action[action_key] += len(entries)
            param_success_per_action[action_key] += sum(
                1 for _, _, label in entries if label == 1
            )

        # Abstract plan dataset: list[(sequence, sequence_length, label)]
        plan_data = load_pickle(plan_path)
        total_abstract_plans += len(plan_data)
        for _, _, label in plan_data:
            if label == 1:
                plan_successes += 1
            else:
                plan_failures += 1

        # Abstract action dataset: dict[str, list[(sequence, seq_len, resample_count)]]
        if action_path.exists():
            action_data = load_pickle(action_path)
            print(f"Seed {seed_dir}")
            print(action_data)
            for action_key, entries in action_data.items():
                total_abstract_actions += len(entries)
                action_counts_per_action[action_key] += len(entries)

                for abstract_plan, _, resample_count in entries:
                    plan_key = tuple(float(x) for x in abstract_plan.flatten())
                    action_stats_per_action[action_key][plan_key].append(resample_count)

    # Print summary.
    print("=" * 60)
    print(f"Dataset Summary: {dataset_dir}")
    print("=" * 60)
    print(f"Seeds loaded:           {seeds_loaded} / {len(seed_dirs)}")
    print()

    print("--- Abstract Plans ---")
    print(f"Total plans:            {total_abstract_plans}")
    if total_abstract_plans:
        print(f"  Successes:            {plan_successes}")
        print(f"  Failures:             {plan_failures}")
        print(f"  Success rate:         {plan_successes / total_abstract_plans:.1%}")
    print()

    print("--- Parameters ---")
    print(f"Total parameter samples: {total_parameters}")
    if param_counts_per_action:
        print("  Per action:")
        for action, count in sorted(param_counts_per_action.items()):
            successes = param_success_per_action[action]
            message = f"    {action}: {count} samples "
            message += f"({successes} success, {count - successes} fail)"
            print(message)
    print()

    print("--- Abstract Actions ---")
    print(f"Total action entries:   {total_abstract_actions}")
    if action_counts_per_action:
        print("  Per action:")
        for action, count in sorted(action_counts_per_action.items()):
            print(f"    {action}: {count}")
            for plan in action_stats_per_action[action].keys():
                resample_counts = action_stats_per_action[action][plan]
                print(f"      Abstract plan: {plan} - {len(resample_counts)}")
                stats = calculate_stats(resample_counts)
                print(f"         Stats: {stats}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        dataset_path = Path("datasets")
    else:
        dataset_path = Path(sys.argv[1])

    if not dataset_path.is_dir():
        print(f"Error: {dataset_path} is not a directory")
        sys.exit(1)

    summarize_dataset(dataset_path)
