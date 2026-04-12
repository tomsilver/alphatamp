#!/usr/bin/env bash
set -euo pipefail

# Submit budget-sweep belief comparison jobs for one or more environments.
#
# Evaluates Oracle, IndexPolicy, SuccessFirst, and ShortestFirst over a
# budget grid and saves:
#   - success_rate_vs_budget.png
#   - time_success_only_vs_budget.png
#   - budget_comparison_summary.json
#   - budget_comparison_metrics.npz
#
# Usage:
#   bash experiments/launch_belief_budget_comparison_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_belief_budget_comparison_matrix.sh "sb1 sb2 sb3"

cd "$(dirname "$0")/.."

ENVS="${1:-o2 o3 o4}"
SUBMITTED=0
SKIPPED=0

for env in $ENVS; do
  hdf5_dir="artifacts_hdf5/encoder_${env}"
  ckpt_dir="checkpoints/belief_encoder/${env}"

  if [[ ! -f "${hdf5_dir}/test.h5" ]]; then
    echo "SKIP ${env}: missing ${hdf5_dir}/test.h5" >&2
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  if [[ ! -f "${hdf5_dir}/train.h5" ]]; then
    echo "SKIP ${env}: missing ${hdf5_dir}/train.h5" >&2
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  if [[ ! -f "${ckpt_dir}/belief_best.pt" ]]; then
    echo "SKIP ${env}: missing ${ckpt_dir}/belief_best.pt" >&2
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "Submitting budget comparison for env=${env}"
  sbatch experiments/belief_budget_comparison.slurm "$env"
  SUBMITTED=$((SUBMITTED + 1))
done

echo "Submitted ${SUBMITTED} budget comparison jobs, skipped ${SKIPPED} (${ENVS})."
