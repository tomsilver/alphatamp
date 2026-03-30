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
#   <difficulty:o2|o3|o4> [threshold]
#
# Example:
# sbatch experiments/refilter_encoder_vocab.slurm o2 0.0

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch experiments/refilter_encoder_vocab.slurm <difficulty:o2|o3|o4> [threshold]"
  exit 2
fi

DIFFICULTY="$1"
THRESHOLD="${2:-0.0}"

case "$DIFFICULTY" in
  o2|o3|o4)
    ;;
  *)
    echo "Unknown difficulty: $DIFFICULTY (expected o2|o3|o4)" >&2
    exit 2
    ;;
esac

FILTER_ARTIFACT="artifacts/encoder_${DIFFICULTY}/encoder_filter_dataset.pkl"
OUTPUT_DIR="artifacts/encoder_${DIFFICULTY}"

if [[ ! -f "$FILTER_ARTIFACT" ]]; then
  echo "Missing filter artifact: $FILTER_ARTIFACT" >&2
  echo "Run all_filtered stage first to generate it." >&2
  exit 2
fi

python -u experiments/refilter_encoder_vocab.py \
  --filter-artifact "$FILTER_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --threshold "$THRESHOLD" \
  --split-name all_filtered \
  --overwrite
