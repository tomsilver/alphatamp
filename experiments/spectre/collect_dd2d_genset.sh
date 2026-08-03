#!/usr/bin/env bash
# Collect the two DD2D generalization test sets (docs/decisions 2026-08-01).
#
#   A  dd2d_v4gen_count : unseen item count (14-16 items = 13-15 blockers), OLD shapes.
#   B  dd2d_v4gen_shape : same unseen count + the new tee/cross families, >=1 of each
#                         forced into every scene.
#
# Both are 40 test problems stratified s0-s3 (10 each), on fresh seed bands disjoint from
# train/val/test ([3M,4M) for A, [4M,5M) for B), with --band=1_000_000 kept so
# compare.stratum_of stays valid. A realized-count floor (min_items = n-items-min) makes
# every kept scene genuinely unseen-count; fill_max=0.72 keeps the resample rate low.
#
# DD2D generation is PYTHONHASHSEED-dependent, so a re-run yields a fresh sample -- the raw
# dirs are the authoritative record (archive them). We pin PYTHONHASHSEED for a best-effort
# repeatable run and print it into the log.
#
# Usage:  bash experiments/spectre/collect_dd2d_genset.sh [WORKERS]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

WORKERS="${1:-12}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
LOGDIR=data/spectre/logs
mkdir -p "$LOGDIR"
echo "# PYTHONHASHSEED=$PYTHONHASHSEED workers=$WORKERS"

echo "=== A: dd2d_v4gen_count (unseen count, old shapes) ==="
python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.collect \
    --out-root data/dd2d/raw_v4gen_count --splits test --seed-band-base 3 \
    --target-test 40 --n-items-min 14 --n-items-max 16 --fill-max 0.72 \
    --workers "$WORKERS" 2>&1 | tee "$LOGDIR/collect_dd2d_v4gen_count.log"

echo "=== B: dd2d_v4gen_shape (unseen count + forced tee/cross) ==="
python -m alphatamp.approaches.spectre.envs.dd2d.dd2d.collect \
    --out-root data/dd2d/raw_v4gen_shape --splits test --seed-band-base 4 \
    --target-test 40 --n-items-min 14 --n-items-max 16 --fill-max 0.72 \
    --shape-set augmented --require-families tee,cross \
    --workers "$WORKERS" 2>&1 | tee "$LOGDIR/collect_dd2d_v4gen_shape.log"

echo "=== convert both (test split only) ==="
python experiments/spectre/dd2d_convert.py env=dd2d_v4gen_count \
    raw_root=data/dd2d/raw_v4gen_count splits=[test]
python experiments/spectre/dd2d_convert.py env=dd2d_v4gen_shape \
    raw_root=data/dd2d/raw_v4gen_shape splits=[test]

echo "=== reuse the dd2d_v4 train vocab (op/pred/type set is shape/count-invariant) ==="
for v in dd2d_v4gen_count dd2d_v4gen_shape; do
    mkdir -p "data/spectre/derived/$v"
    cp -f data/spectre/derived/dd2d_v4/train_vocab.json \
          "data/spectre/derived/$v/train_vocab.json"
done

echo "# done. Score with:"
echo "#   python experiments/spectre/spectre_score_v3.py --env-variant dd2d_v4 \\"
echo "#       --test-variant dd2d_v4gen_count --arm 'v3:checkpoints_v3_unified' \\"
echo "#       --astar-baseline --seeds 0 1 2"
