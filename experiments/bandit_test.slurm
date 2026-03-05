#!/bin/bash
#SBATCH --job-name=bandit_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=experiments/slurm_outputs/bandit_test_%j.out
#SBATCH --error=experiments/slurm_outputs/bandit_test_%j.err

cd /home/sl5183/alphatamp
.venv/bin/python -u experiments/bandit_test.py --plot --save experiments/slurm_outputs/bandit_test_${SLURM_JOB_ID}.png --noise-std 0.05 --max-resamples 2 "$@"
