#!/usr/bin/env bash
set -euo pipefail

# Chain belief encoder pipeline stages 5-7 with SLURM job dependencies.
#
# Dependency chain PER DIFFICULTY (no cross-difficulty dependencies):
#   Stage 5 (HDF5 convert) -> Stage 6 (belief train) -> Stage 7 (comparison)
#
# Prerequisites: Stages 0-2 must have completed (filtered pkl artifacts exist).
#
# Usage:
#   bash experiments/launch_full_belief_pipeline.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty config names.
#         Defaults to "o2 o3 o4".
#
# Dry-run (prints commands without submitting):
#   DRY_RUN=1 bash experiments/launch_full_belief_pipeline.sh "o2 o3 o4"
#   bash experiments/launch_full_belief_pipeline.sh --dry-run "o2 o3 o4"
#
# Examples:
#   bash experiments/launch_full_belief_pipeline.sh "o2 o3 o4"
#   bash experiments/launch_full_belief_pipeline.sh "sb1 sb2 sb3 sb5 sb10"
#   bash experiments/launch_full_belief_pipeline.sh "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"

cd "$(dirname "$0")/.."

# Map difficulty name to its artifact root directory.
artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

# Parse --dry-run flag.
DRY_RUN="${DRY_RUN:-0}"
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    ARGS+=("$arg")
  fi
done

ENVS="${ARGS[0]:-o2 o3 o4}"

for difficulty in $ENVS; do
  echo "=== Pipeline for difficulty=${difficulty} ==="

  # Pre-flight: verify Stage 2 outputs (filtered pkl) exist.
  for split in train validation test; do
    artifact="$(artifact_dir "$difficulty")/encoder_${split}_filtered_dataset.pkl"
    if [[ ! -f "$artifact" ]]; then
      echo "Missing prerequisite: $artifact" >&2
      echo "Run Stages 0-2 first." >&2
      exit 2
    fi
  done

  hdf5_dir="artifacts_hdf5/encoder_${difficulty}"
  ckpt_dir="checkpoints/belief_encoder/${difficulty}"

  # --- Stage 5: HDF5 Conversion ---
  CMD_5="sbatch --parsable experiments/convert_hdf5.slurm ${difficulty}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] ${CMD_5}"
    HDF5_JOB="DRY_RUN"
  else
    HDF5_JOB=$(${CMD_5} | cut -d';' -f1)
    echo "  Stage 5 (HDF5 convert): job ${HDF5_JOB}"
  fi

  # --- Stage 6: Belief Encoder Training (depends on Stage 5) ---
  CMD_6="sbatch --parsable --dependency=afterok:${HDF5_JOB} experiments/train_belief_encoder.slurm data.train_path=${hdf5_dir}/train.h5 data.val_path=${hdf5_dir}/validation.h5 checkpoint.output_dir=${ckpt_dir}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] ${CMD_6}"
    TRAIN_JOB="DRY_RUN"
  else
    TRAIN_JOB=$(${CMD_6} | cut -d';' -f1)
    echo "  Stage 6 (belief train): job ${TRAIN_JOB} (after ${HDF5_JOB})"
  fi

  # --- Stage 7: Comparison (depends on Stage 6) ---
  CMD_7="sbatch --parsable --dependency=afterok:${TRAIN_JOB} experiments/belief_comparison.slurm ${difficulty}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] ${CMD_7}"
  else
    COMPARE_JOB=$(${CMD_7} | cut -d';' -f1)
    echo "  Stage 7 (comparison):   job ${COMPARE_JOB} (after ${TRAIN_JOB})"
  fi

  echo ""
done

echo "All pipeline jobs submitted for: ${ENVS}"
if [[ "$DRY_RUN" != "1" ]]; then
  echo "Monitor with: squeue -u \$USER"
fi
