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
#SBATCH --array=0

set -euo pipefail
mkdir -p experiments/slurm_outputs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# One seed per array task. Defaults to seed=0 if invoked without an array.
SEED=${SLURM_ARRAY_TASK_ID:-0}

# Tier 1 anti-overfitting defaults (vs. spec §8.4 baseline):
#   lr 3e-4 → 1e-4               (slower convergence, less train memorization)
#   weight_decay 1e-4 → 5e-4     (stronger L2)
#   num_f_samples_per_epoch 8→4  (halve per-epoch train repetition)
# An initial run at the spec defaults showed val_loss bottoming at epoch 2
# and AUROC(0) stuck at random (~0.50, below the §11.2 bar of 0.55). These
# overrides give the per-step gradient less leverage to overfit before val
# AUROC has time to climb; pair them with the model.py Tier 2 fix (two-SAB
# atom pool) per spec §12.
#
# Override on the command line when needed, e.g.:
#   sbatch experiments/spectre_train.slurm train.lr=2e-4
#
# train.py auto-detects CUDA (falls back to CPU). On CPU-only clusters,
# drop the `--gres=gpu:1` line above and the run will still complete in
# a few hours at the default config.
.venv/bin/python -u experiments/spectre_train.py \
    seed=${SEED} \
    train.lr=1e-4 \
    train.weight_decay=5e-4 \
    train.num_f_samples_per_epoch=4 \
    "$@"
