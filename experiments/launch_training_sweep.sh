#!/usr/bin/env bash
set -euo pipefail

# Submit encoder MAE training jobs for all bottleneck architectures.
#
# Usage:
#   bash experiments/launch_training_sweep.sh [o2|o3|o4|all] [steps|binary]
#
# Default: all environments, steps mode.
#
# Modes:
#   steps   Soft-label BCE on steps_completed_fraction (default)
#   binary  Hard BCE on binary success matrix
#
# Architectures (hidden_dims = [128, B, 128]):
#   no_bottleneck  B=128 (unconstrained MLP)
#   full_M         B=M
#   half_M         B=ceil(M/2)
#   quarter_M      B=ceil(M/4)
#   eighth_M       B=ceil(M/8)
#
# Outputs per (env, arch):
#   steps mode:  artifacts/encoder_{env}/arch_{arch}/encoder_best.pt
#   binary mode: artifacts/encoder_{env}/binary_encoder/arch_{arch}/encoder_best.pt
#   ...

cd "$(dirname "$0")/.."

# Vocabulary size M per environment
declare -A ENV_M=([o2]=14 [o3]=19 [o4]=34)

# ceil_div A B  →  ceil(A / B)
ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

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
  if [[ -z "${ENV_M[$env]+x}" ]]; then
    echo "Unknown environment: $env (expected o2, o3, or o4)" >&2
    exit 2
  fi

  M=${ENV_M[$env]}
  if [[ "$MODE" == "binary" ]]; then
    data_root="artifacts/encoder_${env}/binary_encoder"
  else
    data_root="artifacts/encoder_${env}"
  fi

  # Filtered datasets always live in the env root, regardless of mode.
  env_root="artifacts/encoder_${env}"

  # Verify filtered data artifacts exist before submitting
  for split in train validation test; do
    artifact="${env_root}/encoder_${split}_filtered_dataset.pkl"
    if [[ ! -f "$artifact" ]]; then
      echo "Missing data artifact: $artifact" >&2
      echo "Run launch_encoder_refilter_train_matrix.sh first to build filtered datasets." >&2
      exit 2
    fi
  done

  declare -A ARCHS
  ARCHS[no_bottleneck]=128
  ARCHS[full_M]=$M
  ARCHS[half_M]=$(ceil_div $M 2)
  ARCHS[quarter_M]=$(ceil_div $M 4)
  ARCHS[eighth_M]=$(ceil_div $M 8)

  for arch_name in "${!ARCHS[@]}"; do
    B=${ARCHS[$arch_name]}
    out_dir="${data_root}/arch_${arch_name}"
    echo "Submitting env=${env} arch=${arch_name} bottleneck=${B} mode=${MODE} out=${out_dir}"
    sbatch experiments/train_encoder_mae.slurm \
      "data.train_path=${env_root}/encoder_train_filtered_dataset.pkl" \
      "data.val_path=${env_root}/encoder_validation_filtered_dataset.pkl" \
      "data.test_path=${env_root}/encoder_test_filtered_dataset.pkl" \
      "train.target_mode=${MODE}" \
      "model.hidden_dims=[128,${B},128]" \
      "checkpoint.output_dir=${out_dir}" \
      "checkpoint.best_filename=encoder_best.pt" \
      "checkpoint.last_filename=encoder_last.pt"
  done
  unset ARCHS
done

echo "Submitted all training jobs."
