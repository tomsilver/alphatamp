#!/bin/bash
#SBATCH --job-name=kinder_env_test_ablation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:30:00
#SBATCH --array=0-3
#SBATCH --output=experiments/slurm_outputs/kinder_env_test_ablation_%A_%a.out
#SBATCH --error=experiments/slurm_outputs/kinder_env_test_ablation_%A_%a.err

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

.venv/bin/python -u experiments/kinder_env_test.py \
    --plot \
    --save experiments/slurm_outputs/kinder_env_test_ablation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.png \
    --max-resamples 10 \
    $ABLATION_FLAGS \
    "$@"
