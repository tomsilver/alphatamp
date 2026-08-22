#!/usr/bin/env bash
# Adaptive-feature ablation sweep (2026-08-21): 6 arms x 3 envs, one seed at a time.
# Runs the three per-env sweeps SEQUENTIALLY (not concurrently) so the three envs never
# oversubscribe the box together; within each env the 6 arms run concurrently. Restock uses
# fewer parallel arms (heavier scene-3d point-cloud tensorization). Usage:
#   bash experiments/spectre/run_ablation_all.sh [SEED]     # default SEED=0
set -euo pipefail
cd /home/josephxu/Projects/alphatamp
source .venv/bin/activate
SEED="${1:-0}"
echo "=== ablation sweep START seed=$SEED  $(date -Is) ==="
python experiments/spectre/spectre_sweep.py --preset ablation_dd2d \
    --env dd2d_v4 --seeds "$SEED" --max-parallel 6 --num-workers 3
echo "=== dd2d done  $(date -Is) ==="
python experiments/spectre/spectre_sweep.py --preset ablation_sb2d \
    --env stickbutton2d_v1 --seeds "$SEED" --max-parallel 6 --num-workers 3
echo "=== sb2d done  $(date -Is) ==="
python experiments/spectre/spectre_sweep.py --preset ablation_restock \
    --env restock3d_v3 --seeds "$SEED" --max-parallel 4 --num-workers 3
echo "=== ablation sweep DONE seed=$SEED  $(date -Is) ==="
