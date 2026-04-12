#!/bin/bash
#SBATCH --job-name=belief_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/slurm_outputs/belief_comparison_%j.out
#SBATCH --error=experiments/slurm_outputs/belief_comparison_%j.err

set -euo pipefail

# Inputs:  artifacts_hdf5/encoder_<difficulty>/{test,train}.h5
#          checkpoints/belief_encoder/<difficulty>/belief_best.pt
# Outputs: results/comparison/<difficulty>/comparison_results.md
#          results/comparison/<difficulty>/success_first_ordering.json
# Runtime: ~5 minutes per difficulty

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/belief_comparison.slurm <difficulty> [extra args...]
#
# Positional args:
#   $1  difficulty : o2|o3|o4|sb1|sb2|sb3|sb5|sb10
#   $2+ extra args : passed through to run_comparison.py
#
# Examples:
#   sbatch experiments/belief_comparison.slurm o3
#   sbatch experiments/belief_comparison.slurm o3 --device cpu --n-seeds 10

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch experiments/belief_comparison.slurm <difficulty> [extra args...]" >&2
  exit 2
fi

DIFFICULTY="$1"; shift

case "$DIFFICULTY" in
  o2|o3|o4|sb1|sb2|sb3|sb5|sb10)
    ;;
  *)
    echo "Unknown difficulty: $DIFFICULTY (expected o2|o3|o4|sb1|sb2|sb3|sb5|sb10)" >&2
    exit 2
    ;;
esac

HDF5_DIR="artifacts_hdf5/encoder_${DIFFICULTY}"
CHECKPOINT_DIR="checkpoints/belief_encoder/${DIFFICULTY}"

# Verify inputs exist.
for f in "${HDF5_DIR}/test.h5" "${HDF5_DIR}/train.h5" "${CHECKPOINT_DIR}/belief_best.pt"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[belief_comparison] difficulty=$DIFFICULTY"
echo "[belief_comparison] host=$(hostname) job=${SLURM_JOB_ID:-none}"

$PYTHON_BIN -u scripts/run_comparison.py \
  --test-data "${HDF5_DIR}/test.h5" \
  --train-data "${HDF5_DIR}/train.h5" \
  --checkpoint "${CHECKPOINT_DIR}/belief_best.pt" \
  --output-dir "results/comparison/${DIFFICULTY}" \
  --device cuda \
  "$@"
