#!/usr/bin/env bash
set -euo pipefail

# Submit encoder MAE training jobs for all bottleneck architectures.
#
# Usage:
#   bash experiments/launch_training_sweep.sh [ENVS] [steps|binary]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_training_sweep.sh "sb1 sb2 sb3"
#         Example: bash experiments/launch_training_sweep.sh "o2 o3 o4 sb1 sb2 sb3" binary
#
#   MODE  Target mode: steps (default) or binary.
#           steps   Soft-label BCE on steps_completed_fraction
#           binary  Hard BCE on binary success matrix
#
# Architectures (hidden_dims = [128, B, 128]) where M is the filtered vocab size:
#   no_bottleneck  B=128 (unconstrained MLP)
#   full_M         B=M
#   half_M         B=ceil(M/2)
#   quarter_M      B=ceil(M/4)
#   eighth_M       B=ceil(M/8)
#
# M is read dynamically from the filtered vocab artifact for each env, so it
# reflects the actual post-filter vocabulary size without any hardcoding.
#
# Outputs per (env, arch):
#   steps mode:  artifacts_{ob|sb}/encoder_{env}/arch_{arch}/encoder_best.pt
#   binary mode: artifacts_{ob|sb}/encoder_{env}/binary_encoder/arch_{arch}/encoder_best.pt

cd "$(dirname "$0")/.."

# Map difficulty name to its artifact root directory.
artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

# ceil_div A B  →  ceil(A / B)
ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

ENVS="${1:-o2 o3 o4}"

MODE="${2:-steps}"
if [[ "$MODE" != "steps" && "$MODE" != "binary" ]]; then
  echo "Unknown mode: $MODE (expected steps or binary)" >&2
  exit 2
fi

for env in $ENVS; do
  env_root="$(artifact_dir "$env")"
  if [[ "$MODE" == "binary" ]]; then
    data_root="${env_root}/binary_encoder"
  else
    data_root="${env_root}"
  fi

  # Verify filtered data artifacts exist before submitting.
  for split in train validation test; do
    artifact="${env_root}/encoder_${split}_filtered_dataset.pkl"
    if [[ ! -f "$artifact" ]]; then
      echo "Missing data artifact: $artifact" >&2
      echo "Run launch_encoder_refilter_train_matrix.sh first to build filtered datasets." >&2
      exit 2
    fi
  done

  # Read M (filtered vocab size) dynamically from the train-filtered vocab artifact.
  vocab_artifact="${env_root}/encoder_vocab_filtered_train_filtered.pkl"
  if [[ ! -f "$vocab_artifact" ]]; then
    echo "Missing vocab artifact: $vocab_artifact" >&2
    echo "Run launch_encoder_refilter_train_matrix.sh first." >&2
    exit 2
  fi
  M=$(python -c "import dill; d=dill.load(open('${vocab_artifact}','rb')); print(len(d['vocabulary']))")
  echo "env=${env}: filtered vocab size M=${M}"

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

echo "Submitted all training jobs (${ENVS})."
