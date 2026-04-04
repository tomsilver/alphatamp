#!/bin/bash
#SBATCH --job-name=encoder_refilter
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/slurm_outputs/encoder_refilter_%j.out
#SBATCH --error=experiments/slurm_outputs/encoder_refilter_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/refilter_encoder_vocab.slurm \
#   <difficulty:o2|o3|o4> [threshold] [min_appl_count] [filter_artifact] [split_name]
#
# Positional args:
#   $1  difficulty     : o2 | o3 | o4
#   $2  threshold      : success-rate lower bound (default 0.0)
#   $3  min_appl_count : min applicability count   (default 0)
#   $4  filter_artifact: path to the dataset pkl to filter from
#                        (default artifacts_{ob|sb}/encoder_<difficulty>/encoder_filter_dataset.pkl)
#   $5  split_name     : suffix for the output vocab filename
#                        (default all_filtered → encoder_vocab_filtered_all_filtered.pkl)
#
# Examples:
#   # Re-filter using the small 50-seed filter dataset (original behaviour):
#   sbatch experiments/refilter_encoder_vocab.slurm o4 0.05 15
#
#   # Re-filter using the full 500-seed training dataset:
#   sbatch experiments/refilter_encoder_vocab.slurm o4 0.05 20 \
#     artifacts_ob/encoder_o4/encoder_train_dataset.pkl train_filtered

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch experiments/refilter_encoder_vocab.slurm <difficulty> [threshold] [min_appl_count] [filter_artifact] [split_name]"
  exit 2
fi

DIFFICULTY="$1"
THRESHOLD="${2:-0.0}"
MIN_APPL_COUNT="${3:-0}"

case "$DIFFICULTY" in
  o2|o3|o4|sb1|sb2|sb3|sb5|sb10)
    ;;
  *)
    echo "Unknown difficulty: $DIFFICULTY (expected o2|o3|o4|sb1|sb2|sb3|sb5|sb10)" >&2
    exit 2
    ;;
esac

# Map difficulty name to its artifact root directory.
artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

# Allow caller to override the artifact path and output split name.
FILTER_ARTIFACT="${4:-$(artifact_dir "$DIFFICULTY")/encoder_filter_dataset.pkl}"
SPLIT_NAME="${5:-all_filtered}"
OUTPUT_DIR="$(artifact_dir "$DIFFICULTY")"

if [[ ! -f "$FILTER_ARTIFACT" ]]; then
  echo "Missing filter artifact: $FILTER_ARTIFACT" >&2
  exit 2
fi

echo "difficulty=$DIFFICULTY  threshold=$THRESHOLD  min_appl_count=$MIN_APPL_COUNT"
echo "artifact=$FILTER_ARTIFACT  split_name=$SPLIT_NAME"

# When filtering the training dataset, propagate the vocab filter to val/test
# and save the filtered train dataset under a distinct name.
EXTRA_ARGS=()
if [[ "$FILTER_ARTIFACT" == *"encoder_train_dataset.pkl" ]]; then
  DATASET_DIR="$(dirname "$FILTER_ARTIFACT")"
  VAL_PATH="${DATASET_DIR}/encoder_validation_dataset.pkl"
  TEST_PATH="${DATASET_DIR}/encoder_test_dataset.pkl"
  EXTRA_ARGS+=(--filtered-dataset-name encoder_train_filtered_dataset.pkl)
  [[ -f "$VAL_PATH" ]]  && EXTRA_ARGS+=(--val-path  "$VAL_PATH")
  [[ -f "$TEST_PATH" ]] && EXTRA_ARGS+=(--test-path "$TEST_PATH")
  echo "Train-mode: propagating filter to val/test splits"
fi

python -u experiments/refilter_encoder_vocab.py \
  --filter-artifact "$FILTER_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --threshold "$THRESHOLD" \
  --min-appl-count "$MIN_APPL_COUNT" \
  --split-name "$SPLIT_NAME" \
  "${EXTRA_ARGS[@]}" \
  --overwrite
