#!/usr/bin/env bash
set -euo pipefail

# Submit encoder MAE training jobs for all bottleneck architectures.
#
# Usage:
#   bash experiments/launch_training_sweep.sh [o2|o3|o4|all]
#
# Default: all environments.
#
# Architectures (hidden_dims = [128, B, 128]):
#   no_bottleneck  B=128 (unconstrained MLP)
#   full_M         B=M
#   half_M         B=ceil(M/2)
#   quarter_M      B=ceil(M/4)
#   eighth_M       B=ceil(M/8)
#
# Outputs per (env, arch):
#   artifacts/encoder_{env}/arch_{arch}/encoder_best.pt
#   artifacts/encoder_{env}/arch_{arch}/encoder_mae_summary.json
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

for env in $ENVS; do
  if [[ -z "${ENV_M[$env]+x}" ]]; then
    echo "Unknown environment: $env (expected o2, o3, or o4)" >&2
    exit 2
  fi

  M=${ENV_M[$env]}
  data_root="artifacts/encoder_${env}"

  # Verify filtered data artifacts exist before submitting
  for split in train validation test; do
    artifact="${data_root}/encoder_${split}_filtered_dataset.pkl"
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
    echo "Submitting env=${env} arch=${arch_name} bottleneck=${B} out=${out_dir}"
    sbatch experiments/train_encoder_mae.slurm \
      "data.train_path=${data_root}/encoder_train_filtered_dataset.pkl" \
      "data.val_path=${data_root}/encoder_validation_filtered_dataset.pkl" \
      "data.test_path=${data_root}/encoder_test_filtered_dataset.pkl" \
      "model.hidden_dims=[128,${B},128]" \
      "checkpoint.output_dir=${out_dir}" \
      "checkpoint.best_filename=encoder_best.pt" \
      "checkpoint.last_filename=encoder_last.pt"
  done
  unset ARCHS
done

echo "Submitted all training jobs."
