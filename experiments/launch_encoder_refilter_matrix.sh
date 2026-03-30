#!/usr/bin/env bash
set -euo pipefail

# Submit offline re-filter jobs for o2/o3/o4.
#
# This ONLY re-runs the vocabulary filtering step from existing
# encoder_filter_dataset.pkl artifacts and overwrites:
#   - artifacts/encoder_oX/encoder_vocab_filtered_all_filtered.pkl
#   - artifacts/encoder_oX/encoder_filter_dataset_filtered.pkl
#
# Usage:
#   bash experiments/launch_encoder_refilter_matrix.sh [threshold]
#
# Example:
#   bash experiments/launch_encoder_refilter_matrix.sh 0.0

cd "$(dirname "$0")/.."

THRESHOLD="${1:-0.0}"

for difficulty in o2 o3 o4; do
  filter_artifact="artifacts/encoder_${difficulty}/encoder_filter_dataset.pkl"
  if [[ ! -f "$filter_artifact" ]]; then
    echo "Missing required filter artifact: $filter_artifact" >&2
    echo "Run all_filtered stage first before re-filtering." >&2
    exit 2
  fi
done

for difficulty in o2 o3 o4; do
  echo "Submitting re-filter job for difficulty=$difficulty threshold=$THRESHOLD"
  sbatch experiments/refilter_encoder_vocab.slurm "$difficulty" "$THRESHOLD"
done

echo "Submitted all re-filter jobs (o2/o3/o4)."
