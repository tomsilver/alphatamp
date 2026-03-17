"""Display refinement quality metrics from results.csv.

Usage:
    python experiments/display_metrics.py                  # reads results.csv
    python experiments/display_metrics.py path/to/file.csv
"""

import ast
import sys
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # attempts_per_step is stored as a Python list repr; parse it back if present.
    if "attempts_per_step" in df.columns:
        df["attempts_per_step"] = df["attempts_per_step"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"  Results: {len(df)} run(s)")
    print(f"{'='*60}\n")

    # ── Per-run table ────────────────────────────────────────────
    display_cols = [
        c for c in [
            "seed", "approach_name", "success", "cost", "duration",
            "avg_attempts_per_step", "total_sampling_attempts",
            "steps_above_5_attempts",
        ] if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))

    # ── Aggregate stats for refinement metrics ───────────────────
    ref_cols = [c for c in ["avg_attempts_per_step", "total_sampling_attempts",
                             "steps_above_5_attempts"] if c in df.columns]
    if ref_cols:
        print(f"\n{'─'*60}")
        print("Aggregate refinement metrics (mean ± std across runs):")
        for col in ref_cols:
            vals = df[col].dropna()
            if len(vals):
                print(f"  {col:<30}  {vals.mean():.3f} ± {vals.std():.3f}")

    # ── Per-step attempt breakdown (if available) ────────────────
    if "attempts_per_step" in df.columns:
        print(f"\n{'─'*60}")
        print("Per-step attempt counts (each row = one run):")
        for i, row in df.iterrows():
            label = f"  run {i}"
            if "seed" in df.columns:
                label += f" (seed={row['seed']})"
            print(f"{label}: {row['attempts_per_step']}")

    print()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results.csv")
    if not path.exists():
        print(f"No results file found at {path}")
        sys.exit(1)
    df = _load(path)
    _print_summary(df)


if __name__ == "__main__":
    main()
