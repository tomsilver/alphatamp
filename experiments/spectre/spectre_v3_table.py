"""Emit the per-stratum comparison table with real seed error bars.

Replaces hand-transcription. ``as_built_v2.2`` Section 3.7 was copied by hand out of the
analysis notebook, so it silently desynchronised from the cache on every re-run; this is
the single artifact every v3 gate quotes.

Two things it does that the notebook's summary cell does not:

- **Seed spread, not problem spread.** ``load_fp_records`` averages a problem's FP across
  seeds before returning it, so a std taken downstream is the across-*problem* spread of a
  seed-mean. Every v3 gate is accepted on "no stratum regresses beyond seed noise", which
  needs the between-*seed* spread of the per-stratum mean. This reads the per-seed loader
  and reports both, because they answer different questions and confusing them would make
  the gates either trivially passable or unpassable.
- **Marks unseeded rows.** Deterministic baselines (astar, PIGINet) have no seed axis;
  reporting a ``+- 0.00`` next to them would imply a stability they were never tested for.

Usage::

    python experiments/spectre/spectre_v3_table.py --env-variant dd2d_v4
    python experiments/spectre/spectre_v3_table.py --env-variant dd2d_v3 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from alphatamp.approaches.spectre.dd2d_compare import (
    METHOD_ORDER,
    load_fp_records_per_seed,
)

REPO = Path(__file__).resolve().parents[2]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    """Sample standard deviation; ``nan`` below two observations.

    ``nan`` rather than ``0.0`` on a single seed on purpose -- a zero would read as
    "this method is perfectly stable" when the truth is "nobody measured".
    """
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "--"
    return f"{mean:.2f}" if math.isnan(std) else f"{mean:.2f} ± {std:.2f}"


def build_table(records: list[dict]) -> tuple[list[str], list[list[str]], list[dict]]:
    """Return ``(header, rows, tidy)`` for the per-stratum FP table."""
    strata = sorted({r["stratum"] for r in records})
    by_method_seed: dict[tuple[str, object, object], list[float]] = defaultdict(list)
    for r in records:
        by_method_seed[(r["method"], r["seed"], r["stratum"])].append(r["fp"])
        by_method_seed[(r["method"], r["seed"], "ALL")].append(r["fp"])

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in records)]
    methods += sorted({r["method"] for r in records} - set(methods))

    header = ["method", "seeds", "ALL"] + [f"s{s}" for s in strata]
    rows: list[list[str]] = []
    tidy: list[dict] = []
    for method in methods:
        seeds = sorted(
            {r["seed"] for r in records if r["method"] == method},
            key=lambda s: (s is None, s),
        )
        row = [method, "-" if seeds == [None] else str(len(seeds))]
        for stratum in ["ALL"] + list(strata):
            # per seed: mean over that seed's problems; then spread across seeds
            per_seed = [
                _mean(by_method_seed[(method, s, stratum)])
                for s in seeds
                if by_method_seed[(method, s, stratum)]
            ]
            mean, std = _mean(per_seed), _std(per_seed)
            row.append(_fmt(mean, std))
            tidy.append(
                {
                    "method": method,
                    "stratum": stratum,
                    "n_seeds": len(per_seed),
                    "mean_fp": mean,
                    "std_fp_across_seeds": std,
                }
            )
        rows.append(row)
    return header, rows, tidy


def render_markdown(header: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(header))
    ]
    out = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out += [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-variant", default="dd2d_v4")
    ap.add_argument("--data-root", default=str(REPO / "data" / "spectre"))
    ap.add_argument("--csv", default=None, help="also write the tidy table here")
    a = ap.parse_args(argv)

    cache = Path(a.data_root) / "derived" / a.env_variant / "compare_cache"
    records = load_fp_records_per_seed(cache)
    header, rows, tidy = build_table(records)

    n_problems = len({r["problem_id"] for r in records})
    print(f"# {a.env_variant} — mean rollout FP (lower is better), n={n_problems}")
    print(
        "# ± is the spread ACROSS SEEDS of the per-stratum mean, not across problems."
    )
    print("# 'seeds = -' marks a deterministic single run with no seed axis.")
    print()
    print(render_markdown(header, rows))

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tidy[0]))
            w.writeheader()
            w.writerows(tidy)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
