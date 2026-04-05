#!/bin/bash
#SBATCH --job-name=encoder_mae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/slurm_outputs/encoder_mae_%j.out
#SBATCH --error=experiments/slurm_outputs/encoder_mae_%j.err

set -euo pipefail

# Assumptions for this launcher:
# - 1 MIG GPU slice with ~10GB GPU memory
# - ~15% of full A100 compute throughput
# - 16GB CPU/system RAM (via --mem=16G)
# If runs are slower than expected on this slice, increase --time first.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/train_encoder_mae.slurm [hydra overrides...]
#
# Examples:
# sbatch experiments/train_encoder_mae.slurm
# sbatch experiments/train_encoder_mae.slurm train.num_epochs=100 train.batch_size=128
# sbatch experiments/train_encoder_mae.slurm data.test_path=artifacts/encoder_dataset/encoder_test_dataset.pkl

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[encoder_mae] host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[encoder_mae] python=$($PYTHON_BIN --version 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[encoder_mae] GPU status:"
  nvidia-smi || true
fi

$PYTHON_BIN -u experiments/train_encoder_mae.py "$@"
