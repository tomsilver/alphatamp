#!/usr/bin/env bash
set -euo pipefail

# Submit HDF5 conversion jobs for one or more environments.
#
# Converts filtered pickle artifacts (from Stage 2) to self-contained HDF5
# files that the belief encoder training pipeline reads.
#
# Usage:
#   bash experiments/launch_hdf5_convert_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty config names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_hdf5_convert_matrix.sh "sb1 sb2 sb3"
#
# Prerequisites: Stages 0-2 must have completed (filtered pkl artifacts must exist).
#
# Outputs per difficulty:
#   artifacts_hdf5/encoder_<difficulty>/{train,validation,test}.h5

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

# Pre-flight: verify filtered pkl artifacts exist for all envs.
for difficulty in $ENVS; do
  for split in train validation test; do
    artifact="$(artifact_dir "$difficulty")/encoder_${split}_filtered_dataset.pkl"
    if [[ ! -f "$artifact" ]]; then
      echo "Missing: $artifact" >&2
      echo "Run Stages 0-2 (vocab build + dataset build + refilter) first." >&2
      exit 2
    fi
  done
done

for difficulty in $ENVS; do
  echo "Submitting HDF5 conversion for difficulty=$difficulty"
  sbatch experiments/convert_hdf5.slurm "$difficulty"
done

echo "Submitted all HDF5 conversion jobs (${ENVS})."
