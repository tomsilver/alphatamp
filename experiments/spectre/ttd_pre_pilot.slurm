#!/bin/bash
# TTD §10.0 pre-pilot on SLURM (the first go/no-go gate).
#
# The pre-pilot's expensive unit is the per-(operating-point, ΣA-band) η landscape,
# and those landscapes are independent — so this is a SLURM array job: each array task
# runs one shard (a disjoint subset of the (OP, band) work items) and writes a partial
# JSON report; a final merge step aggregates them and prints the GO/NO-GO verdict.
#
# With the defaults (2 operating points × 3 bands = 6 landscapes), use 6 array tasks:
#
#     mkdir -p data/spectre/ttd/prepilot
#     sbatch experiments/spectre/ttd_pre_pilot.slurm
#     # after all array tasks finish, merge:
#     source .venv/bin/activate
#     python -m alphatamp.approaches.spectre.ttd.ttd_eval.pre_pilot merge \
#         --inputs data/spectre/ttd/prepilot/shard_*.json \
#         --out    data/spectre/ttd/prepilot/report.json
#
# A single η bisection over concave k∈{4,5} subsets is seconds-to-minutes, so budget
# generously (200 subsets × ~5 nester calls per landscape). Tighten --rot-grid-deg to 5
# and raise --node-cap for the final authoritative run; the values below are a balanced
# first pass. Trailing args are forwarded to the `run` subcommand.
#
#SBATCH --job-name=ttd_prepilot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/slurm_outputs/ttd_prepilot_%A_%a.out
#SBATCH --error=experiments/slurm_outputs/ttd_prepilot_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail
source .venv/bin/activate

N_SHARDS=6
OUT_DIR=data/spectre/ttd/prepilot
mkdir -p "${OUT_DIR}" experiments/slurm_outputs

python -m alphatamp.approaches.spectre.ttd.ttd_eval.pre_pilot run \
    --k 5 \
    --n-subsets 200 \
    --n-pool 40 \
    --rot-grid-deg 10 \
    --node-cap 8000 \
    --tol 0.05 \
    --seed 0 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --out "${OUT_DIR}/shard_${SLURM_ARRAY_TASK_ID}.json" \
    "$@"
