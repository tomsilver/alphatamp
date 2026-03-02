#!/bin/bash
#SBATCH --job-name=bandit_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=bandit_test_%j.out
#SBATCH --error=bandit_test_%j.err

cd /home/sl5183/alphatamp
.venv/bin/python -u experiments/bandit_test.py "$@"
