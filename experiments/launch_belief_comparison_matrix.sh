#!/usr/bin/env bash
set -euo pipefail

# Submit belief encoder comparison jobs for one or more environments.
#
# Compares IndexPolicy (belief-encoder-guided) against Oracle, SuccessFirst,
# and ShortestFirst baselines using OfflineEvaluator.
#
# Usage:
#   bash experiments/launch_belief_comparison_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty config names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_belief_comparison_matrix.sh "sb1 sb2 sb3"
#
# Prerequisites: Stage 6 (belief encoder training) must have completed.
#
# Outputs per difficulty:
#   results/comparison/<difficulty>/comparison_results.md
#   results/comparison/<difficulty>/success_first_ordering.json

cd "$(dirname "$0")/.."

ENVS="${1:-o2 o3 o4}"
SUBMITTED=0
SKIPPED=0

for env in $ENVS; do
  hdf5_dir="artifacts_hdf5/encoder_${env}"
  ckpt_dir="checkpoints/belief_encoder/${env}"

  # Skip-and-warn on missing artifacts (comparison is a terminal stage;
  # partial results are useful).
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

  echo "Submitting comparison for env=${env}"
  sbatch experiments/belief_comparison.slurm "$env"
  SUBMITTED=$((SUBMITTED + 1))
done

echo "Submitted ${SUBMITTED} comparison jobs, skipped ${SKIPPED} (${ENVS})."
