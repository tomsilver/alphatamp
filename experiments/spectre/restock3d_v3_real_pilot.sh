#!/usr/bin/env bash
# PILOT for the restock3d_v3_real (hybrid-prune) collection: real-refine 5 problems/stratum, SAVED
# into the train split (indices 0-4), so the full run's pre-scan merges them (no re-collection).
#
# Purpose (a resource + agreement GATE before the multi-hour full run):
#   * measure per-worker RSS (the heartbeat's wRSSmax) => calibrate _PER_WORKER_GB_REAL for the
#     full run's auto-sizing (plan anchor: ~5 GB/worker on n=9);
#   * measure wall-clock/problem (full-run ETA);
#   * measure analytic<->real agreement (analytic-feasible->real-fail = false positives we want;
#     audit-sampled analytic-infeasible->real-success = false negatives = is trusting the bulk safe);
#   * confirm GPU ~= 0 (collection is CPU+RAM only).
#
# CONSERVATIVE fixed workers (RSS is unknown pre-pilot, so under-subscribe; the watchdog backstops):
# n=6/7/8/9 -> 8/6/4/4. Same K_max/r_cap/samples as the full run so episodes are config-compatible.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

declare -A KMAX=( [0]=35 [1]=40 [2]=135 [3]=185 )
declare -A RCAP=( [0]=60 [1]=65 [2]=65  [3]=75  )
# Bumped 2026-08-25 after the pilot measured a LOW per-worker RSS (n=8 = 2.3 GB, n=6/7 = 1.2 GB) --
# far under the ~5 GB estimate -- with ~49 GB free, so the heavy strata were badly under-subscribed.
declare -A W=(    [0]=8  [1]=8  [2]=10  [3]=10  )

STRATA=("$@")
if [ "${#STRATA[@]}" -eq 0 ]; then STRATA=(0 1 2 3); fi

for s in "${STRATA[@]}"; do
  echo "### PILOT restock3d_v3_real stratum $s (n=$((6 + s))) K_max=${KMAX[$s]} r_cap=${RCAP[$s]}s workers=${W[$s]} $(date -Is)"
  python experiments/spectre/restock3d_v3_collect.py \
    --env-variant restock3d_v3_real \
    --refiner-mode hybrid_prune \
    --strata "$s" \
    --k-max "${KMAX[$s]}" \
    --refinement-timeout "${RCAP[$s]}" \
    --samples-per-step 6 \
    --train 5 --val 0 --test 0 \
    --workers "${W[$s]}"
done
echo "### PILOT restock3d_v3_real done $(date -Is)"
