#!/bin/bash
# SPECTRE model training on the RT2D-n3-v1 500/100/100 dataset.
#
# Default: array of 3 jobs, one per seed, 20-epoch training each
# (spec §8.4 + §8.7 — multi-seed protocol).
#
#     sbatch experiments/spectre_train.slurm
#
# Single-seed override:
#     sbatch --array=0 experiments/spectre_train.slurm
#
# More seeds (e.g. spec §8.7's "expand to 5 if seed std exceeds gap/2"):
#     sbatch --array=0-4 experiments/spectre_train.slurm
#
# Different env / hyperparams (forwarded to Hydra after `--`):
#     sbatch experiments/spectre_train.slurm env=routedtransport2d_n4_v1
#     sbatch experiments/spectre_train.slurm train.num_f_samples_per_epoch=4
#
# Reads:    data/spectre/raw/<env_variant>/{train,val}/episodes/
#           data/spectre/derived/<env_variant>/train_vocab.json
# Writes:   data/spectre/checkpoints/<env_variant>/seed_<i>/{best.pt,last.pt,
#           log.jsonl,model_meta.json}
#
#SBATCH --job-name=spectre_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/slurm_outputs/spectre_train_%A_%a.out
#SBATCH --error=experiments/slurm_outputs/spectre_train_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
mkdir -p experiments/slurm_outputs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# One seed per array task. Defaults to seed=0 if invoked without an array.
SEED=${SLURM_ARRAY_TASK_ID:-0}

# train.py auto-detects CUDA (falls back to CPU). On CPU-only clusters,
# drop the `--gres=gpu:1` line above and the run will still complete in
# a few hours at the default config.
.venv/bin/python -u experiments/spectre_train.py \
    seed=${SEED} \
    "$@"
