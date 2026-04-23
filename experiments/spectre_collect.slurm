#!/bin/bash
# SPECTRE dataset collection on SLURM.
#
# Two supported shapes:
#
# A) Single-node fanout (default). One SLURM job collects the full
#    [problem_seed_start, problem_seed_end) range with 8 parallel workers
#    inside the job:
#
#        sbatch experiments/spectre_collect.slurm \
#            split=train problem_seed_start=0 problem_seed_end=500
#
# B) Chunked array. Uncomment the `#SBATCH --array=0-9` line and the
#    chunk-size logic below to split the full range across 10 array tasks
#    of 50 problems each (10 jobs × 8 workers = ~500 problems at once):
#
#        sbatch experiments/spectre_collect.slurm split=train
#
# Any trailing arguments are forwarded to Hydra overrides.
#
#SBATCH --job-name=spectre_collect
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/slurm_outputs/spectre_collect_%A_%a.out
#SBATCH --error=experiments/slurm_outputs/spectre_collect_%A_%a.err
##SBATCH --array=0-9          # uncomment for mode B (chunked array)

set -euo pipefail
mkdir -p experiments/slurm_outputs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Chunking (mode B). When run under an array, divide the 500-problem range
# into CHUNK_SIZE-wide windows, one per SLURM_ARRAY_TASK_ID.
CHUNK_SIZE=50
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    CHUNK_ARGS="problem_seed_start=${START} problem_seed_end=${END}"
else
    CHUNK_ARGS=""
fi

.venv/bin/python -u experiments/spectre_collect.py \
    workers=8 \
    ${CHUNK_ARGS} \
    "$@"
