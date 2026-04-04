#!/usr/bin/env bash
set -euo pipefail

# Submit offline re-filter jobs using the full 500-seed training dataset
# instead of the small 50-seed filter dataset.
#
# Outputs per difficulty:
#   artifacts_{ob|sb}/encoder_<difficulty>/encoder_vocab_filtered_train_filtered.pkl
#   artifacts_{ob|sb}/encoder_<difficulty>/encoder_filter_dataset_filtered.pkl  (overwritten)
#
# To use the resulting vocab for the next training run, set in the config:
#   run.vocab_file: artifacts_{ob|sb}/encoder_<difficulty>/encoder_vocab_filtered_train_filtered.pkl
#
# Usage:
#   bash experiments/launch_encoder_refilter_train_matrix.sh [threshold] [min_appl_count] [ENVS]
#
#   threshold      Success-rate lower bound (default 0.05)
#   min_appl_count Minimum applicability count (default 20)
#   ENVS           Space-separated env list (default "o2 o3 o4")
#
# Example:
#   bash experiments/launch_encoder_refilter_train_matrix.sh 0.05 20 "sb1 sb2 sb3"

cd "$(dirname "$0")/.."

# Map difficulty name to its artifact root directory.
artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

THRESHOLD="${1:-0.05}"
MIN_APPL_COUNT="${2:-20}"
ENVS="${3:-o2 o3 o4}"

for difficulty in $ENVS; do
  train_artifact="$(artifact_dir "$difficulty")/encoder_train_dataset.pkl"
  if [[ ! -f "$train_artifact" ]]; then
    echo "Missing required train artifact: $train_artifact" >&2
    echo "Run mode=dataset first to build the training dataset." >&2
    exit 2
  fi
done

for difficulty in $ENVS; do
  echo "Submitting train-refilter job for difficulty=$difficulty threshold=$THRESHOLD min_appl_count=$MIN_APPL_COUNT"
  sbatch experiments/refilter_encoder_vocab.slurm \
    "$difficulty" "$THRESHOLD" "$MIN_APPL_COUNT" \
    "$(artifact_dir "$difficulty")/encoder_train_dataset.pkl" \
    "train_filtered"
done

echo "Submitted all train-refilter jobs (${ENVS})."
echo "Output vocab files will be: artifacts_{ob|sb}/encoder_<difficulty>/encoder_vocab_filtered_train_filtered.pkl"
