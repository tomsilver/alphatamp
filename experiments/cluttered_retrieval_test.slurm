#!/bin/bash
#SBATCH --job-name=cluttered_retrieval_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:30:00
#SBATCH --output=experiments/slurm_outputs/cluttered_retrieval_test_%j.out
#SBATCH --error=experiments/slurm_outputs/cluttered_retrieval_test_%j.err

cd /home/sl5183/alphatamp
.venv/bin/python -u experiments/cluttered_retrieval_test.py --plot --save experiments/slurm_outputs/cluttered_retrieval_test_${SLURM_JOB_ID}.png --max-resamples 20 "$@"
