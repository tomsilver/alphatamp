#!/usr/bin/env bash
set -euo pipefail

# Submit offline eval jobs for all trained arch checkpoints.
#
# Usage:
#   bash experiments/launch_eval_sweep.sh [o2|o3|o4|all] [steps|binary]
#
# Default: all environments, steps mode.
# Skips any arch_* dir that does not yet have encoder_best.pt.
#
# Outputs per (env, arch):
#   steps mode:  artifacts/encoder_{env}/arch_{arch}/offline_eval/...
#   binary mode: artifacts/encoder_{env}/binary_encoder/arch_{arch}/offline_eval/...

cd "$(dirname "$0")/.."

ENVS="${1:-all}"
if [[ "$ENVS" == "all" ]]; then
  ENVS="o2 o3 o4"
fi

MODE="${2:-steps}"
if [[ "$MODE" != "steps" && "$MODE" != "binary" ]]; then
  echo "Unknown mode: $MODE (expected steps or binary)" >&2
  exit 2
fi

for env in $ENVS; do
  if [[ "$MODE" == "binary" ]]; then
    data_root="artifacts/encoder_${env}/binary_encoder"
  else
    data_root="artifacts/encoder_${env}"
  fi
  # Filtered datasets always live in the env root, regardless of mode.
  env_root="artifacts/encoder_${env}"

  if [[ ! -d "$data_root" ]]; then
    echo "Missing env artifact dir: $data_root — skipping" >&2
    continue
  fi

  submitted=0
  skipped=0

  for arch_dir in "${data_root}"/arch_*/; do
    [[ -d "$arch_dir" ]] || continue
    ckpt="${arch_dir}encoder_best.pt"
    if [[ ! -f "$ckpt" ]]; then
      echo "Skipping ${arch_dir} (no encoder_best.pt)" >&2
      skipped=$(( skipped + 1 ))
      continue
    fi
    arch_name="$(basename "$arch_dir")"
    out_dir="${arch_dir}offline_eval"
    echo "Submitting eval env=${env} arch=${arch_name}"
    sbatch experiments/offline_encoder_rollout_eval.slurm \
      "data.train_path=${env_root}/encoder_train_filtered_dataset.pkl" \
      "data.test_path=${env_root}/encoder_test_filtered_dataset.pkl" \
      "checkpoint.path=${ckpt}" \
      "output.dir=${out_dir}"
    submitted=$(( submitted + 1 ))
  done

  echo "env=${env}: submitted=${submitted} skipped=${skipped}"
done

echo "Done submitting eval jobs."
