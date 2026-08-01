#!/usr/bin/env bash
# Everything after the StickButton2D collection, in the order the gates require.
#
# Exists so the post-collection phase is one reviewable command rather than six typed at
# the end of a long run: vocab -> pipeline check -> Gate A (does coverage rank?) ->
# baselines -> train 3 seeds -> score. Each stage's output goes to data/spectre/logs/.
#
# The ordering is not cosmetic. Vocab must be rebuilt from the *final* train split (an
# earlier partial vocab is stale and OOV-silent), Gate A must be read before the training
# numbers are trusted, and the baseline bracket must exist before the method number means
# anything.
#
#   bash experiments/spectre/sb2d_finalize.sh [seeds...]     # default: 0 1 2
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
LOG_DIR="$REPO/data/spectre/logs"
mkdir -p "$LOG_DIR"
SEEDS=("${@:-0 1 2}")
[ $# -eq 0 ] && SEEDS=(0 1 2)
ENV_VARIANT="stickbutton2d_v1"

say() { printf '\n=== %s === (%s)\n' "$1" "$(date +%H:%M:%S)"; }

say "census"
for f in b1b2 b3 b5; do
  tail -12 "$LOG_DIR/sb2d_collect_$f.log" 2>/dev/null | grep -E "^b[0-9]|WARNING" || true
done
for split in train val test; do
  n=$(ls "data/spectre/raw/$ENV_VARIANT/$split/episodes" 2>/dev/null | wc -l)
  echo "  $split: $n episodes"
done

say "vocab (train split only, OOV-checks val/test)"
python experiments/spectre/spectre_build_vocab.py env=$ENV_VARIANT 2>&1 \
  | grep -viE "warn|gym.logger" | tee "$LOG_DIR/sb2d_vocab.log"

say "pipeline check"
python experiments/spectre/spectre_check_pipeline.py env=$ENV_VARIANT 2>&1 \
  | grep -viE "warn|gym.logger" | tail -30 | tee "$LOG_DIR/sb2d_check.log"

# Gate A. Read this before trusting any v3 number: if coverage no longer beats the static
# order on the collected pools, the training set is being built for a feature that does
# not rank here, and the headline would be measuring something else.
say "Gate A — coverage re-ranking (test split)"
python experiments/spectre/sb2d_rerank_gate.py --split test 2>&1 \
  | grep -viE "warn|gym.logger" | tee "$LOG_DIR/sb2d_gateA.log"

say "B1-B5 baseline bracket (test split, uncensored)"
python experiments/spectre/sb2d_baselines.py --split test --budget 200 2>&1 \
  | grep -viE "warn|gym.logger" | tee "$LOG_DIR/sb2d_baselines.log"

# Watch the first epoch's wall clock. The selector runs an *uncensored* deployed rollout
# over the val split every epoch, and SB2D pools are 200 candidates against DD2D's ~30, so
# its cost per epoch is roughly an order of magnitude higher here. If epochs run long, the
# lever is `--val-episodes` (a strided subsample — fewer episodes, still uncensored), NOT
# `--select-budget`: censoring the selector below the tail that separates models is the
# mistake R8/G6b were about, and it would be a worse mistake on a 200-deep pool.
# Two arms, not one. Gate A measured `waste` as *exactly* inert at b5 (waste_only ties
# static to two decimal places) and actively harmful at b3 when used as a hand-coded
# tie-break (4.73 vs 2.95 for coverage alone). The deployed model *learns* weights over
# both columns rather than tie-breaking on waste, so that does not condemn the pair -- but
# it makes "does the waste column earn its place on this environment" a question with
# evidence behind it rather than an ablation for completeness. `--coverage-mode coverage`
# zeroes the waste column without changing any tensor shape.
say "train v3 (${SEEDS[*]}) x 2 arms — concurrent; training is CPU-bound, not GPU-bound"
pids=()
for seed in "${SEEDS[@]}"; do
  for arm in both coverage; do
    suffix=""; [ "$arm" = "coverage" ] && suffix="_covonly"
    python -u -m alphatamp.approaches.spectre.train_v3 \
      --env $ENV_VARIANT --seed "$seed" --epochs 30 --num-workers 3 \
      --overlap-mode jaccard --coverage-feats --coverage-mode "$arm" \
      --aggregate-records --evidence-attn --state-delta \
      ${suffix:+--out-suffix "$suffix"} \
      > "$LOG_DIR/sb2d_train${suffix}_s$seed.log" 2>&1 &
    pids+=($!)
    echo "  seed $seed arm=$arm -> $LOG_DIR/sb2d_train${suffix}_s$seed.log"
  done
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
for seed in "${SEEDS[@]}"; do
  for suffix in "" "_covonly"; do
    grep -E "train_v3 done|Traceback|Error" \
      "$LOG_DIR/sb2d_train${suffix}_s$seed.log" | tail -2
  done
done
[ "$fail" -eq 1 ] && echo "WARNING: at least one training run exited non-zero"

# A checkpoint is not a result until its training log says the run finished.
say "scoring (uncensored deployed FP, test split)"
python experiments/spectre/spectre_score_v3.py --env-variant $ENV_VARIANT \
  --arm "v3 coverage+waste:checkpoints_v3" \
  --arm "v3 coverage only:checkpoints_v3_covonly" \
  --baseline "v3 coverage+waste:checkpoints_v3" \
  --seeds "${SEEDS[@]}" 2>&1 \
  | grep -viE "warn|gym.logger" | tee "$LOG_DIR/sb2d_score.log"

say "done"
