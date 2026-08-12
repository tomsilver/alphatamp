#!/usr/bin/env bash
# Collect the DD2D SHAPE-ONLY generalization test set (docs/decisions 2026-08-04).
#
#   dd2d_v4gen_shapeonly : the new tee/cross families forced into every scene (>= 1 of
#                          EACH), at the TRAINED 9-12 blocker count (10-13 items).
#
# This isolates the shape variable. Its sibling dd2d_v4gen_shape confounds the new shapes
# with an unseen 13-15 blocker count, where s2 FP degraded -- later attributed to a
# count-driven pool-composition artifact, not the shapes (docs/notebook/07 2026-08-02).
# Here count is held at the headline band by OMITTING --n-items-* (the collector's default
# is 10-13 items = 9-12 blockers, with no realized-count floor), so only --shape-set /
# --require-families change vs a standard dd2d_v4 collection.
#
# 40 test problems stratified s0-s3 (10 each), on a fresh seed band [5M,6M) disjoint from
# train/val/test ([0,3M)) and the existing gensets (count [3M,4M), shape [4M,5M)), with
# --band=1_000_000 kept so compare.stratum_of stays valid. fill_max is left at its default
# (0.55): at 9-12 blockers the scenes are less dense than the 14-16 gensets, and the forced
# tee/cross place regardless of the fill cap.
#
# DD2D generation is PYTHONHASHSEED-dependent, so a re-run yields a fresh sample -- the raw
# dir is the authoritative record (archive it). We pin PYTHONHASHSEED for a best-effort
# repeatable run and print it into the log.
#
# Usage:  bash experiments/spectre/collect_dd2d_shapeonly.sh [WORKERS]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

WORKERS="${1:-12}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
LOGDIR=data/spectre/logs
mkdir -p "$LOGDIR"
echo "# PYTHONHASHSEED=$PYTHONHASHSEED workers=$WORKERS"

echo "=== dd2d_v4gen_shapeonly (headline 9-12 blockers + forced tee/cross) ==="
python -m alphatamp.approaches.spectre.envs.dd2d.drawer.collect \
    --out-root data/dd2d/raw_v4gen_shapeonly --splits test --seed-band-base 5 \
    --target-test 40 --shape-set augmented --require-families tee,cross \
    --workers "$WORKERS" 2>&1 | tee "$LOGDIR/collect_dd2d_v4gen_shapeonly.log"

echo "=== convert (test split only) ==="
python experiments/spectre/dd2d_convert.py env=dd2d_v4gen_shapeonly \
    raw_root=data/dd2d/raw_v4gen_shapeonly splits=[test]

echo "=== reuse the dd2d_v4 train vocab (op/pred/type set is shape/count-invariant) ==="
mkdir -p data/spectre/derived/dd2d_v4gen_shapeonly
cp -f data/spectre/derived/dd2d_v4/train_vocab.json \
      data/spectre/derived/dd2d_v4gen_shapeonly/train_vocab.json

echo "# done. Score with:"
echo "#   python experiments/spectre/spectre_score.py --env-variant dd2d_v4 \\"
echo "#       --test-variant dd2d_v4gen_shapeonly --arm 'v3:checkpoints_v3_unified' \\"
echo "#       --astar-baseline --seeds 0 1 2"
echo "# and build the compare_cache with:"
echo "#   python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 \\"
echo "#       --test-variant dd2d_v4gen_shapeonly --methods astar piginet spectre3 \\"
echo "#       --no-ablations --seeds 0 1 2 --force"
