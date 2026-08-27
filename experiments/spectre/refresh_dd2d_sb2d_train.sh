#!/usr/bin/env bash
# Residual-adaptive re-do of the SPECTRE refresh (2026-08-26, round 2).
#
# All adaptive arms are a zero-init |F|-gated residual over a FROZEN pure-geometry trunk
# (X2, docs/failed_records_as_built.md §Round 2): `--residual-adaptive --freeze-static
# --init-static-from <abl_static>`. Step-join dropped. Plan:
# /home/josephxu/.claude/plans/refresh-all-five-sections-scalable-rose.md
#
# Phase 1: holdout static trunks (needed to warm-start the holdout residuals).
# Phase 2: all residual arms (main +records/+scalars/full reuse the EXISTING abl_static
#          trunks; holdout full-residual warm-starts from the Phase-1 hstatic trunks).
# Main abl_static (dd2d_v4 / stickbutton2d_v1) is REUSED from the first refresh — not retrained.
#
# Idempotent: skips any (arm,seed) whose best.pt already exists (so the dd2d full-residual
# seed-0 screen is not redone). Coexists with restock3d-v3: MAX_PARALLEL=2, --num-workers 3.
#   bash experiments/spectre/refresh_dd2d_sb2d_train.sh [seeds...]   # default 0 1 2
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
SCAL="--overlap-mode jaccard --coverage-feats --repeat-feats"
T_DD2D="data/spectre/checkpoints_spectre_norec_noov_atoms_abl_static/dd2d_v4/seed_{seed}/best.pt"
T_SB2D="data/spectre/checkpoints_spectre_norec_noov_atoms_abl_static/stickbutton2d_v1/seed_{seed}/best.pt"
T_HDD2D="data/spectre/checkpoints_spectre_norec_noov_atoms_hstatic/dd2d_v4_holdout_s3/seed_{seed}/best.pt"
T_HSB2D="data/spectre/checkpoints_spectre_norec_noov_atoms_hstatic/stickbutton2d_v1_holdout_b5/seed_{seed}/best.pt"

# job = "env|out_suffix|ckpt_dir|tag|<flags>"
PHASE1=(
  "dd2d_v4_holdout_s3|_hstatic|checkpoints_spectre_norec_noov_atoms_hstatic|dd2d_hstatic|--no-overlap --no-records --train-strata 0 1 2"
  "stickbutton2d_v1_holdout_b5|_hstatic|checkpoints_spectre_norec_noov_atoms_hstatic|sb2d_hstatic|--no-overlap --no-records --train-strata 0 1 2"
)
PHASE2=(
  "dd2d_v4|_resid_records|checkpoints_spectre_noov_atoms_resid_records|dd2d_resid_records|--no-overlap $RECS $RESID --init-static-from $T_DD2D"
  "dd2d_v4|_resid_recjac|checkpoints_spectre_atoms_resid_recjac|dd2d_resid_recjac|--overlap-mode jaccard $RECS $RESID --init-static-from $T_DD2D"
  "dd2d_v4|_resid_scalars|checkpoints_spectre_norec_atoms_resid_scalars|dd2d_resid_scalars|$SCAL --no-records $RESID --init-static-from $T_DD2D"
  "dd2d_v4|_resid_full|checkpoints_spectre_atoms_resid_full|dd2d_resid_full|$SCAL $RECS $RESID --init-static-from $T_DD2D"
  "stickbutton2d_v1|_resid_records|checkpoints_spectre_noov_atoms_resid_records|sb2d_resid_records|--no-overlap $RECS $RESID --init-static-from $T_SB2D"
  "stickbutton2d_v1|_resid_recjac|checkpoints_spectre_atoms_resid_recjac|sb2d_resid_recjac|--overlap-mode jaccard $RECS $RESID --init-static-from $T_SB2D"
  "stickbutton2d_v1|_resid_scalars|checkpoints_spectre_norec_atoms_resid_scalars|sb2d_resid_scalars|$SCAL --no-records $RESID --init-static-from $T_SB2D"
  "stickbutton2d_v1|_resid_full|checkpoints_spectre_atoms_resid_full|sb2d_resid_full|$SCAL $RECS $RESID --init-static-from $T_SB2D"
  "dd2d_v4_holdout_s3|_hfull|checkpoints_spectre_atoms_hfull|dd2d_hfull|$SCAL $RECS --train-strata 0 1 2 $RESID --init-static-from $T_HDD2D"
  "stickbutton2d_v1_holdout_b5|_hfull|checkpoints_spectre_atoms_hfull|sb2d_hfull|$SCAL $RECS --train-strata 0 1 2 $RESID --init-static-from $T_HSB2D"
)

run_one() {
  local env="$1" suf="$2" dir="$3" tag="$4" flags="$5" seed="$6"
  local best="$CK/$dir/$env/seed_$seed/best.pt"
  if [ -f "$best" ]; then echo "[skip] $tag seed=$seed (best.pt exists)"; return 0; fi
  local log="$LOG_DIR/refresh_${tag}_s${seed}.log"
  echo "### $(date +%FT%T) START $tag seed=$seed env=$env" >> "$log"
  # shellcheck disable=SC2086
  stdbuf -oL -eL python -u -m alphatamp.approaches.spectre.train \
    --env "$env" --seed "$seed" --epochs 30 --num-workers "$NW" \
    $BB $flags --out-suffix "$suf" >> "$log" 2>&1
  local rc=$?
  echo "### $(date +%FT%T) END $tag seed=$seed rc=$rc" >> "$log"; return "$rc"
}

run_phase() {
  local -n jobs=$1
  for job in "${jobs[@]}"; do
    IFS='|' read -r env suf dir tag flags <<< "$job"
    for seed in "${SEEDS[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
      run_one "$env" "$suf" "$dir" "$tag" "$flags" "$seed" &
      echo "[$(date +%T)] launched $tag seed=$seed (pid $!)"; sleep 2
    done
  done
  wait
}

echo "residual refresh: phase1=${#PHASE1[@]} arms, phase2=${#PHASE2[@]} arms x ${#SEEDS[@]} seeds, MAX_PARALLEL=$MAX_PARALLEL"
echo "=== PHASE 1: holdout static trunks ==="; run_phase PHASE1
echo "=== PHASE 2: residual arms ==="; run_phase PHASE2
echo "ALL DONE $(date +%FT%T)"
