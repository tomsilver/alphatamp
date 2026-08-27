#!/usr/bin/env bash
# Collect the restock3d_v3_REAL dataset, ONE stratum per process, sequential + auto-chained.
#
# SPLIT-SPECIFIC LABELING (decided 2026-08-25 after the pilot found the analytic classifier is a
# poor proxy for real MP on v3 -- ~58% false-positive on analytic-feasible, ~13% false-negative on
# the trusted analytic-infeasible bulk; samples=6 confirmed adequate, so the disagreement is real):
#   * TRAIN  -> hybrid-prune (25% audit): listwise training is robust to ~13% FN in the trusted bulk,
#              and pruning keeps train affordable. Merges the pilot's 5/stratum via pre-scan.
#   * VAL/TEST -> FULLY real (refiner_mode=real, every candidate real-refined, no analytic trust):
#              a mislabeled-feasible candidate biases the time-to-first-success eval, so eval labels
#              must be clean. ~2x the per-problem cost, but only 60 val + 120 test problems.
#
# One process = one block count => uniform, predictable per-worker RAM; workers AUTO-SIZE from live
# free RAM (0.80*CPU / 0.80*RAM, RAM-aware; _PER_WORKER_GB_REAL calibrated from the pilot). Resumable
# (pre-scan). `set -u` (not -e) so a SHORTFALL on one stratum/split does not abort the rest.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

declare -A KMAX=(  [0]=35  [1]=40  [2]=135 [3]=185 )
declare -A RCAP=(  [0]=60  [1]=65  [2]=65  [3]=75  )
declare -A TRAIN=( [0]=100 [1]=100 [2]=50  [3]=50  )
declare -A VAL=(   [0]=20  [1]=20  [2]=10  [3]=10  )
declare -A TEST=(  [0]=40  [1]=40  [2]=20  [3]=20  )

STRATA=("$@")
if [ "${#STRATA[@]}" -eq 0 ]; then STRATA=(0 1 2 3); fi

for s in "${STRATA[@]}"; do
  echo "### restock3d_v3_real stratum $s (n=$((6 + s))) K_max=${KMAX[$s]} r_cap=${RCAP[$s]}s $(date -Is)"

  echo "###   [train] hybrid_prune"
  python experiments/spectre/restock3d_v3_collect.py \
    --env-variant restock3d_v3_real --refiner-mode hybrid_prune --splits train --strata "$s" \
    --k-max "${KMAX[$s]}" --refinement-timeout "${RCAP[$s]}" --samples-per-step 6 \
    --train "${TRAIN[$s]}"

  echo "###   [val+test] fully-real"
  python experiments/spectre/restock3d_v3_collect.py \
    --env-variant restock3d_v3_real --refiner-mode real --splits val test --strata "$s" \
    --k-max "${KMAX[$s]}" --refinement-timeout "${RCAP[$s]}" --samples-per-step 6 \
    --val "${VAL[$s]}" --test "${TEST[$s]}"
done
echo "### restock3d_v3_real collection done $(date -Is)"
