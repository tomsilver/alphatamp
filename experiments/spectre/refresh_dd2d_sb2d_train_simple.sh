#!/usr/bin/env bash
# SPECTRE-SIMPLE training: the deployed two-stage residual recipe, but with the
# earlier/simple coverage/waste (`--legacy-coverage`) and `repeat` carrying SB2D.
# Drives compare_methods_simple.py (docs/decisions/07 2026-08-27).
#
# Only the coverage-bearing arms differ from the deployed unified recipe, so ONLY
# +scalars and +full are retrained here (2 arms x 2 envs x seeds). The static /
# +records / +records+jaccard arms have no coverage columns -> definitionally
# identical -> reused from the deployed caches by the notebook (grafted). Each
# residual warm-starts from the EXISTING deployed pure-geometry static trunk.
#
# The simple env_variants reuse the parent episodes:
#   dd2d_v4_simple            -> symlink of dd2d_v4 (repeat inert; simple cov/waste carries it)
#   stickbutton2d_v1_simple   -> COPY of stickbutton2d_v1 with provenance.env_variant rewritten
#                                (so run_training resolves domain `_SB2D_REPEAT`: step_certificate
#                                on the press schemas -> repeat fires; simple cov/waste is 0 there).
#
# Idempotent: skips any (arm,seed) whose best.pt already exists.
#   bash experiments/spectre/refresh_dd2d_sb2d_train_simple.sh [seeds...]   # default 0 1 2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
LOG_DIR="$REPO/data/spectre/logs"; mkdir -p "$LOG_DIR"
SEEDS=("$@"); [ "$#" -eq 0 ] && SEEDS=(0 1 2)
MAX_PARALLEL="${MAX_PARALLEL:-2}"; NW="${NW:-3}"
CK="$REPO/data/spectre"

BB="--use-pca-feats --use-edgeconv --use-point-sab --pma-seeds 4 --atom-mode profiles --select-window 5"
RESID="--residual-adaptive --freeze-static"
RECS="--aggregate-records --evidence-attn --state-delta"
# Simple/legacy coverage/waste (the only change vs the deployed SCAL); repeat still on.
SCAL="--overlap-mode jaccard --coverage-feats --repeat-feats --legacy-coverage"
# Warm-start from the EXISTING deployed pure-geometry static trunks (definition-invariant).
T_DD2D="data/spectre/checkpoints_spectre_norec_noov_atoms_abl_static/dd2d_v4/seed_{seed}/best.pt"
T_SB2D="data/spectre/checkpoints_spectre_norec_noov_atoms_abl_static/stickbutton2d_v1/seed_{seed}/best.pt"

# job = "env|out_suffix|ckpt_dir|tag|<flags>"
ARMS=(
  "dd2d_v4_simple|_simple_scalars|checkpoints_spectre_norec_atoms_simple_scalars|dd2d_simple_scalars|$SCAL --no-records $RESID --init-static-from $T_DD2D"
  "dd2d_v4_simple|_simple_full|checkpoints_spectre_atoms_simple_full|dd2d_simple_full|$SCAL $RECS $RESID --init-static-from $T_DD2D"
  "stickbutton2d_v1_simple|_simple_scalars|checkpoints_spectre_norec_atoms_simple_scalars|sb2d_simple_scalars|$SCAL --no-records $RESID --init-static-from $T_SB2D"
  "stickbutton2d_v1_simple|_simple_full|checkpoints_spectre_atoms_simple_full|sb2d_simple_full|$SCAL $RECS $RESID --init-static-from $T_SB2D"
)

run_one() {
  local env="$1" suf="$2" dir="$3" tag="$4" flags="$5" seed="$6"
  local best="$CK/$dir/$env/seed_$seed/best.pt"
  if [ -f "$best" ]; then echo "[skip] $tag seed=$seed (best.pt exists)"; return 0; fi
  local log="$LOG_DIR/simple_${tag}_s${seed}.log"
  echo "### $(date +%FT%T) START $tag seed=$seed env=$env" >> "$log"
  # shellcheck disable=SC2086
  stdbuf -oL -eL python -u -m alphatamp.approaches.spectre.train \
    --env "$env" --seed "$seed" --epochs 30 --num-workers "$NW" \
    $BB $flags --out-suffix "$suf" >> "$log" 2>&1
  local rc=$?
  echo "### $(date +%FT%T) END $tag seed=$seed rc=$rc" >> "$log"; return "$rc"
}

echo "simple refresh: ${#ARMS[@]} arms x ${#SEEDS[@]} seeds, MAX_PARALLEL=$MAX_PARALLEL, NW=$NW"
for job in "${ARMS[@]}"; do
  IFS='|' read -r env suf dir tag flags <<< "$job"
  for seed in "${SEEDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
    run_one "$env" "$suf" "$dir" "$tag" "$flags" "$seed" &
    echo "[$(date +%T)] launched $tag seed=$seed (pid $!)"; sleep 2
  done
done
wait
echo "ALL DONE $(date +%FT%T)"
