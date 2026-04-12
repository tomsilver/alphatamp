#!/usr/bin/env bash
set -euo pipefail

# Submit belief encoder training jobs for one or more environments.
#
# Trains the full belief encoder pipeline (SkeletonEncoder -> TokenBuilder ->
# BeliefEncoder -> prediction heads) using DAgger-scheduled rollout-consistent
# prefixes on HDF5 data.
#
# Usage:
#   bash experiments/launch_belief_train_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty config names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_belief_train_matrix.sh "sb1 sb2 sb3"
#
# Prerequisites: Stage 5 (HDF5 conversion) must have completed.
#
# Outputs per difficulty:
#   checkpoints/belief_encoder/<difficulty>/belief_best.pt
#   checkpoints/belief_encoder/<difficulty>/belief_last.pt

cd "$(dirname "$0")/.."

ENVS="${1:-o2 o3 o4}"

# Pre-flight: verify HDF5 files exist for all envs.
for env in $ENVS; do
  hdf5_dir="artifacts_hdf5/encoder_${env}"
  for split in train validation; do
    if [[ ! -f "${hdf5_dir}/${split}.h5" ]]; then
      echo "Missing: ${hdf5_dir}/${split}.h5" >&2
      echo "Run launch_hdf5_convert_matrix.sh first." >&2
      exit 2
    fi
  done
done

for env in $ENVS; do
  hdf5_dir="artifacts_hdf5/encoder_${env}"
  ckpt_dir="checkpoints/belief_encoder/${env}"

  echo "Submitting belief encoder training: env=${env} ckpt_dir=${ckpt_dir}"
  sbatch experiments/train_belief_encoder.slurm \
    "data.train_path=${hdf5_dir}/train.h5" \
    "data.val_path=${hdf5_dir}/validation.h5" \
    "checkpoint.output_dir=${ckpt_dir}"
done

echo "Submitted all belief encoder training jobs (${ENVS})."
