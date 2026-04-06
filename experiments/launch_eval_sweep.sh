#!/usr/bin/env bash
set -euo pipefail

# Submit offline eval jobs for all trained arch checkpoints.
#
# Usage:
#   bash experiments/launch_eval_sweep.sh [ENVS] [env|binary_encoder] [ARCH_GLOB] [true|false] [MIN_SUPPORT]
#
#   ENVS        Space-separated list of encoder_dataset_difficulty names, or "all".
#               Defaults to "o2 o3 o4".
#               "all" expands to "o2 o3 o4 sb1 sb2 sb3 sb5 sb10".
#               Example: bash experiments/launch_eval_sweep.sh "sb1 sb2 sb3"
#               Example: bash experiments/launch_eval_sweep.sh all
#
#   ARTIFACT_SUBDIR  Which subdirectory under artifacts_{ob|sb}/encoder_{env}/ to scan.
#               "env" (default): scan directly in artifacts_{ob|sb}/encoder_{env}/
#               "binary_encoder": scan in artifacts_{ob|sb}/encoder_{env}/binary_encoder/
#               Skips any matching arch dir that does not yet have encoder_best.pt.
#
#   ARCH_GLOB   Glob pattern for arch subdirs to evaluate (default: arch_*).
#               Example: bash experiments/launch_eval_sweep.sh "o2 o3 o4" env arch_transformer
#
#   GREEDY_ORACLE_ENABLED  Enable greedy_conditional_success_oracle baseline.
#               Default: true.
#               Example: bash experiments/launch_eval_sweep.sh "o2 o3 o4" env arch_* false
#
#   MIN_SUPPORT   min_support for greedy_conditional_success_oracle (default: 5).
#               Example: bash experiments/launch_eval_sweep.sh "o2 o3 o4" env arch_* true 20
#
# Outputs per (env, arch):
#   env subdir:            artifacts_{ob|sb}/encoder_{env}/arch_{arch}/offline_eval/...
#   binary_encoder subdir: artifacts_{ob|sb}/encoder_{env}/binary_encoder/arch_{arch}/offline_eval/...

cd "$(dirname "$0")/.."

# Map difficulty name to its artifact root directory.
artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

ENVS="${1:-o2 o3 o4}"
if [[ "$ENVS" == "all" ]]; then
  ENVS="o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
fi

ARTIFACT_SUBDIR="${2:-env}"
if [[ "$ARTIFACT_SUBDIR" != "env" && "$ARTIFACT_SUBDIR" != "binary_encoder" ]]; then
  echo "Unknown artifact subdir: $ARTIFACT_SUBDIR (expected env or binary_encoder)" >&2
  exit 2
fi

ARCH_GLOB="${3:-arch_*}"
GREEDY_ORACLE_ENABLED="${4:-true}"
GREEDY_ORACLE_MIN_SUPPORT="${5:-5}"

if [[ "$GREEDY_ORACLE_ENABLED" != "true" && "$GREEDY_ORACLE_ENABLED" != "false" ]]; then
  echo "Invalid GREEDY_ORACLE_ENABLED: $GREEDY_ORACLE_ENABLED (expected true or false)" >&2
  exit 2
fi

if ! [[ "$GREEDY_ORACLE_MIN_SUPPORT" =~ ^[0-9]+$ ]]; then
  echo "Invalid MIN_SUPPORT: $GREEDY_ORACLE_MIN_SUPPORT (expected positive integer)" >&2
  exit 2
fi

if [[ "$GREEDY_ORACLE_MIN_SUPPORT" -lt 1 ]]; then
  echo "Invalid MIN_SUPPORT: $GREEDY_ORACLE_MIN_SUPPORT (expected >= 1)" >&2
  exit 2
fi

for env in $ENVS; do
  env_root="$(artifact_dir "$env")"
  if [[ "$ARTIFACT_SUBDIR" == "binary_encoder" ]]; then
    data_root="${env_root}/binary_encoder"
  else
    data_root="${env_root}"
  fi

  if [[ ! -d "$data_root" ]]; then
    echo "Missing env artifact dir: $data_root — skipping" >&2
    continue
  fi

  submitted=0
  skipped=0

  for arch_dir in "${data_root}"/${ARCH_GLOB}/; do
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
      "output.dir=${out_dir}" \
      "greedy_conditional_success_oracle.enabled=${GREEDY_ORACLE_ENABLED}" \
      "greedy_conditional_success_oracle.min_support=${GREEDY_ORACLE_MIN_SUPPORT}"
    submitted=$(( submitted + 1 ))
  done

  echo "env=${env}: submitted=${submitted} skipped=${skipped}"
done

echo "Done submitting eval jobs (${ENVS})."
