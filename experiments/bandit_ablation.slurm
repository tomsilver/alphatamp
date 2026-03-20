#!/bin/bash
#SBATCH --job-name=bandit_ablation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:30:00
#SBATCH --array=0-3
#SBATCH --output=experiments/slurm_outputs/bandit_ablation_20_%A_%a.out
#SBATCH --error=experiments/slurm_outputs/bandit_ablation_20_%A_%a.err

# Ablation conditions (SLURM_ARRAY_TASK_ID → flags):
#   0 — full model          (both scorers on)
#   1 — no abstract plan scorer
#   2 — no parameter scorer
#   3 — no scorers          (both off)

cd /home/sl5183/alphatamp

case $SLURM_ARRAY_TASK_ID in
    0) ABLATION_FLAGS="" ;;
    1) ABLATION_FLAGS="--no-abstract-plan-scorer" ;;
    2) ABLATION_FLAGS="--no-parameter-scorer" ;;
    3) ABLATION_FLAGS="--no-abstract-plan-scorer --no-parameter-scorer" ;;
esac

.venv/bin/python -u experiments/bandit_test.py \
    --plot \
    --save experiments/slurm_outputs/bandit_ablation_20_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.png \
    --noise-std 0.05 \
    --max-resamples 2 \
    $ABLATION_FLAGS \
    "$@"
