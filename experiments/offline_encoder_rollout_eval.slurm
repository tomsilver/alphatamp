#!/bin/bash
#SBATCH --job-name=offline_encoder_eval_o2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:55:00
#SBATCH --output=experiments/slurm_outputs/offline_encoder_eval_o2_%j.out
#SBATCH --error=experiments/slurm_outputs/offline_encoder_eval_o2_%j.err

set -euo pipefail

# Runs offline encoder rollout evaluation using dataset/model artifacts.
# Default config already points to artifacts/encoder_o2.

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[offline_encoder_eval] host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[offline_encoder_eval] python=$($PYTHON_BIN --version 2>&1)"

# Usage:
# sbatch experiments/offline_encoder_rollout_eval.slurm [hydra overrides...]
#
# Examples:
# sbatch experiments/offline_encoder_rollout_eval.slurm
# sbatch experiments/offline_encoder_rollout_eval.slurm output.dir=artifacts/encoder_offline_o2_run2
# sbatch experiments/offline_encoder_rollout_eval.slurm budget.seconds=30 generator_baseline.max_generated_skeletons_per_row=200

$PYTHON_BIN -u experiments/offline_encoder_rollout_eval.py "$@"