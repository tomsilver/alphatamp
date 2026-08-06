#!/usr/bin/env bash
# Collect the DD2D shape-SIZE-sweep set: tee/cross shrunk to x0.7 linear (docs/decisions
# 2026-08-06). The confirmatory shrink for the shape-generalization investigation.
#
#   dd2d_v4gen_shapeonly_sz07 : identical to dd2d_v4gen_shapeonly (new tee/cross forced
#                               into every scene at the trained 9-12 blocker count) EXCEPT
#                               tee/cross are uniformly scaled to 0.7 linear (~0.49 area;
#                               convex-hull footprint tee 60->29, cross 68->33), via the
#                               new collector `--family-size-scale` lever.
#
# Read across ALL methods (astar/PIGINet/SPECTRE): a size effect must appear on the
# geometry-free planner too; a SPECTRE-only shift means the deficit is representational,
# not physical. Fresh seed band [6M,7M) disjoint from train/val/test ([0,3M)) and the
# existing gensets (count [3M,4M), shape [4M,5M), shapeonly [5M,6M)); --band=1_000_000 kept
# so compare.stratum_of stays valid.
#
# DD2D generation is PYTHONHASHSEED-dependent; the raw dir is the authoritative record.
#
# Usage:  bash experiments/spectre/collect_dd2d_shapeonly_sz07.sh [WORKERS]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

WORKERS="${1:-12}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
LOGDIR=data/spectre/logs
mkdir -p "$LOGDIR"
echo "# PYTHONHASHSEED=$PYTHONHASHSEED workers=$WORKERS"

echo "=== dd2d_v4gen_shapeonly_sz07 (forced tee/cross at x0.7 linear, 9-12 blockers) ==="
python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.collect \
    --out-root data/dd2d/raw_v4gen_shapeonly_sz07 --splits test --seed-band-base 6 \
    --target-test 40 --shape-set augmented --require-families tee,cross \
    --family-size-scale tee=0.7,cross=0.7 \
    --workers "$WORKERS" 2>&1 | tee "$LOGDIR/collect_dd2d_v4gen_shapeonly_sz07.log"

echo "=== convert (test split only) ==="
python experiments/spectre/dd2d_convert.py env=dd2d_v4gen_shapeonly_sz07 \
    raw_root=data/dd2d/raw_v4gen_shapeonly_sz07 splits=[test]

echo "=== reuse the dd2d_v4 train vocab (op/pred/type set is shape/size-invariant) ==="
mkdir -p data/spectre/derived/dd2d_v4gen_shapeonly_sz07
cp -f data/spectre/derived/dd2d_v4/train_vocab.json \
      data/spectre/derived/dd2d_v4gen_shapeonly_sz07/train_vocab.json

echo "# done. Build the compare_cache with:"
echo "#   python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 \\"
echo "#       --test-variant dd2d_v4gen_shapeonly_sz07 --methods astar piginet spectre3 \\"
echo "#       --no-ablations --seeds 0 1 2 --force"
