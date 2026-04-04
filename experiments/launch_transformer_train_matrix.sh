#!/usr/bin/env bash
set -euo pipefail

# Submit SkeletonTransformer training jobs for one or more environments.
#
# Usage:
#   bash experiments/launch_transformer_train_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of env names (default "o2 o3 o4")
#
# Outputs per env:
#   artifacts_{ob|sb}/encoder_{env}/arch_transformer/encoder_best.pt
#   artifacts_{ob|sb}/encoder_{env}/arch_transformer/encoder_last.pt
#
# These sit alongside existing MLP arch_* dirs and do NOT overwrite them.
# Run launch_eval_sweep.sh after training to evaluate.

cd "$(dirname "$0")/.."

artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

ENVS="${1:-o2 o3 o4}"

for env in $ENVS; do
  env_root="$(artifact_dir "$env")"

  for split in train validation test; do
    artifact="${env_root}/encoder_${split}_filtered_dataset.pkl"
    if [[ ! -f "$artifact" ]]; then
      echo "Missing data artifact: $artifact" >&2
      echo "Run launch_encoder_refilter_train_matrix.sh first." >&2
      exit 2
    fi
  done

  out_dir="${env_root}/arch_transformer"
  echo "Submitting transformer training: env=${env} out=${out_dir}"
  sbatch experiments/train_encoder_mae.slurm \
    "data.train_path=${env_root}/encoder_train_filtered_dataset.pkl" \
    "data.val_path=${env_root}/encoder_validation_filtered_dataset.pkl" \
    "data.test_path=${env_root}/encoder_test_filtered_dataset.pkl" \
    "checkpoint.output_dir=${out_dir}" \
    "checkpoint.best_filename=encoder_best.pt" \
    "checkpoint.last_filename=encoder_last.pt"
done

echo "Submitted transformer training jobs (${ENVS})."
echo "After completion, run: bash experiments/launch_eval_sweep.sh \"${ENVS}\""
