#!/usr/bin/env bash
# Collect the restock3d_v3 SYNTHETIC dataset, ONE stratum per process.
#
# The geometry pool draw is the only heavy step (analytic labels = no motion planning), but the
# n=9 K_max=200 A* enumeration balloons to ~5 GB/worker, so a single mixed job at high concurrency
# overran the 59 GB box (12 workers x ~5 GB OOM'd a child -> BrokenProcessPool). Running one block
# count per process gives a uniform, predictable per-worker RAM peak, so each stratum is sized to
# its own safe concurrency and fully reclaims memory before the next. Mirrors restock3d_v2_run_all.
#
# Resumable: each invocation pre-scans on-disk episodes, so a re-run continues where it stopped and
# a completed stratum submits nothing. `set -u` (not -e) so a SHORTFALL on one stratum does not
# abort the rest.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

# Per-stratum workers sized to the geometry-search RSS peak: ~0.5 GB (n=6) up to ~8-9 GB for the
# n=9 K_max=200 FULL enumeration (measured: 6 workers overran the watchdog). One block count per
# process => uniform RAM; conservative so the peak stays well under 59 GB (heavy strata ~36 GB).
declare -A W=( [0]=16 [1]=12 [2]=6 [3]=4 )

for s in 0 1 2 3; do
  echo "### restock3d_v3 stratum $s workers ${W[$s]} $(date -Is)"
  python experiments/spectre/restock3d_v3_collect.py --strata "$s" --workers "${W[$s]}"
done
echo "### restock3d_v3 collection done $(date -Is)"
