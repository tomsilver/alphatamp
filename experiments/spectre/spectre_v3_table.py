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
from pathlib import Path

from alphatamp.approaches.spectre.compare import (
    build_table,
    load_fp_records_per_seed,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[2]


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
