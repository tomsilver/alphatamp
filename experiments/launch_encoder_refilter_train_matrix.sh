#!/usr/bin/env bash
set -euo pipefail

# Submit offline re-filter jobs for o2/o3/o4 using the full 500-seed training
# dataset instead of the small 50-seed filter dataset.
#
# Outputs per difficulty:
#   artifacts/encoder_oX/encoder_vocab_filtered_train_filtered.pkl
#   artifacts/encoder_oX/encoder_filter_dataset_filtered.pkl  (overwritten)
#
# To use the resulting vocab for the next training run, set in the config:
#   run.vocab_file: artifacts/encoder_oX/encoder_vocab_filtered_train_filtered.pkl
#
# Usage:
#   bash experiments/launch_encoder_refilter_train_matrix.sh [threshold] [min_appl_count]
#
# Example:
#   bash experiments/launch_encoder_refilter_train_matrix.sh 0.05 20

cd "$(dirname "$0")/.."

THRESHOLD="${1:-0.05}"
MIN_APPL_COUNT="${2:-20}"

for difficulty in o2 o3 o4; do
  train_artifact="artifacts/encoder_${difficulty}/encoder_train_dataset.pkl"
  if [[ ! -f "$train_artifact" ]]; then
    echo "Missing required train artifact: $train_artifact" >&2
    echo "Run mode=dataset first to build the training dataset." >&2
    exit 2
  fi
done

for difficulty in o2 o3 o4; do
  echo "Submitting train-refilter job for difficulty=$difficulty threshold=$THRESHOLD min_appl_count=$MIN_APPL_COUNT"
  sbatch experiments/refilter_encoder_vocab.slurm \
    "$difficulty" "$THRESHOLD" "$MIN_APPL_COUNT" \
    "artifacts/encoder_${difficulty}/encoder_train_dataset.pkl" \
    "train_filtered"
done

echo "Submitted all train-refilter jobs (o2/o3/o4)."
echo "Output vocab files will be: artifacts/encoder_oX/encoder_vocab_filtered_train_filtered.pkl"
