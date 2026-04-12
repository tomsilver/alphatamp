#!/bin/bash
#SBATCH --job-name=belief_encoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/slurm_outputs/belief_encoder_%j.out
#SBATCH --error=experiments/slurm_outputs/belief_encoder_%j.err

set -euo pipefail

# Inputs:  artifacts_hdf5/encoder_<difficulty>/{train,validation}.h5
# Outputs: checkpoints/belief_encoder/<difficulty>/belief_{best,last}.pt
# Runtime: ~30 min to 2 hours depending on dataset size and num_epochs

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/train_belief_encoder.slurm [hydra overrides...]
#
# Examples:
#   sbatch experiments/train_belief_encoder.slurm \
#     data.train_path=artifacts_hdf5/encoder_o3/train.h5 \
#     data.val_path=artifacts_hdf5/encoder_o3/validation.h5 \
#     checkpoint.output_dir=checkpoints/belief_encoder/o3
#
#   sbatch experiments/train_belief_encoder.slurm \
#     data.train_path=artifacts_hdf5/encoder_o3/train.h5 \
#     data.val_path=artifacts_hdf5/encoder_o3/validation.h5 \
#     checkpoint.output_dir=checkpoints/belief_encoder/o3 \
#     train.num_epochs=100

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[belief_encoder] host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[belief_encoder] python=$($PYTHON_BIN --version 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[belief_encoder] GPU status:"
  nvidia-smi || true
fi

$PYTHON_BIN -u experiments/train_belief_encoder.py "$@"
