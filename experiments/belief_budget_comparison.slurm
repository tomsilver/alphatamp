#!/bin/bash
#SBATCH --job-name=belief_budget_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=experiments/slurm_outputs/belief_budget_comparison_%j.out
#SBATCH --error=experiments/slurm_outputs/belief_budget_comparison_%j.err

set -euo pipefail

# Inputs:  artifacts_hdf5/encoder_<difficulty>/{test,train}.h5
#          checkpoints/belief_encoder/<difficulty>/belief_best.pt
# Outputs: results/budget_comparison/<difficulty>/{*.png,*.json,*.npz}

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/belief_budget_comparison.slurm <difficulty> [extra args...]
#
# Positional args:
#   $1  difficulty : o2|o3|o4|sb1|sb2|sb3|sb5|sb10
#   $2+ extra args : passed through to run_budget_comparison.py

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch experiments/belief_budget_comparison.slurm <difficulty> [extra args...]" >&2
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

for f in "${HDF5_DIR}/test.h5" "${HDF5_DIR}/train.h5" "${CHECKPOINT_DIR}/belief_best.pt"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f" >&2
    exit 2
  fi
done

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "[belief_budget_comparison] difficulty=$DIFFICULTY"
echo "[belief_budget_comparison] host=$(hostname) job=${SLURM_JOB_ID:-none}"

$PYTHON_BIN -u scripts/run_budget_comparison.py \
  --test-data "${HDF5_DIR}/test.h5" \
  --train-data "${HDF5_DIR}/train.h5" \
  --checkpoint "${CHECKPOINT_DIR}/belief_best.pt" \
  --output-dir "results/budget_comparison/${DIFFICULTY}" \
  --device cuda \
  "$@"
