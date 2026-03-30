#!/bin/bash
#SBATCH --job-name=encoder_dataset
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=23:55:00
#SBATCH --output=experiments/slurm_outputs/encoder_dataset_%j.out
#SBATCH --error=experiments/slurm_outputs/encoder_dataset_%j.err

set -euo pipefail

# Estimated runtime basis (sequential, 1 worker):
# 400 dataset seeds * ~90 sec/seed ~= 10 hours
# + vocabulary build + startup/variability buffer => request 24 hours.
#
# With N workers the dataset phase scales linearly.  Set --cpus-per-task=N
# and pass run.num_workers=N (or omit to let the script infer from
# SLURM_CPUS_PER_TASK).
#
# Two-phase usage (vocab once, then parallel dataset workers):
# -------------------------------------------------------
# Phase 1 – build vocab only:
#   sbatch experiments/build_encoder_dataset.slurm vocab_only \
#     0 1 run.mode=vocab vocab.seed_start=1000 vocab.seed_stop=1300
#
# Phase 2 – build dataset slices in parallel (example: 4 jobs of 100 seeds):
#   VOCAB=artifacts/encoder_dataset/encoder_vocab_vocab_only.pkl
#   for i in 0 100 200 300; do
#     sbatch --cpus-per-task=8 experiments/build_encoder_dataset.slurm train $i $((i+100)) \
#       run.mode=dataset run.vocab_file=$VOCAB
#   done
# -------------------------------------------------------
#
# Simple single-job usage (all in one, with in-process workers):
#   sbatch --cpus-per-task=8 experiments/build_encoder_dataset.slurm validation 0 400

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/build_encoder_dataset.slurm <split_name> <seed_start> <seed_stop> [hydra overrides...]
# Example:
# sbatch experiments/build_encoder_dataset.slurm validation 0 50 vocab.seed_start=1000 vocab.seed_stop=1300

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch experiments/build_encoder_dataset.slurm <split_name> <seed_start> <seed_stop> [hydra overrides...]"
  exit 2
fi

SPLIT_NAME="$1"
SEED_START="$2"
SEED_STOP="$3"
shift 3

# Infer num_workers from SLURM allocation when not explicitly overridden.
# Workers can always be overridden via a Hydra override: run.num_workers=N
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

python -u experiments/build_encoder_dataset.py \
  run.split_name="${SPLIT_NAME}" \
  run.seed_start="${SEED_START}" \
  run.seed_stop="${SEED_STOP}" \
  run.num_workers="${NUM_WORKERS}" \
  "$@"
