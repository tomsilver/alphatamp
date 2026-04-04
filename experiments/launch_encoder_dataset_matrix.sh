#!/usr/bin/env bash
set -euo pipefail

# Submit train/validation/test dataset-build jobs for one or more environments
# using pre-built filtered vocab files.
#
# Usage:
#   bash experiments/launch_encoder_dataset_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty config names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_encoder_dataset_matrix.sh "sb1 sb2 sb3"
#
# Notes:
# - Uses config groups:
#     encoder_dataset_difficulty=<difficulty>
#     encoder_dataset_split={train,validation,test}
# - Assumes filtered vocab artifacts already exist at:
#     artifacts_{ob|sb}/encoder_<difficulty>/encoder_vocab_filtered_all_filtered.pkl

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

for difficulty in $ENVS; do
  vocab_path="$(artifact_dir "$difficulty")/encoder_vocab_filtered_all_filtered.pkl"
  if [[ ! -f "$vocab_path" ]]; then
    echo "Missing filtered vocab artifact: $vocab_path" >&2
    echo "Run: bash experiments/launch_encoder_all_filtered_matrix.sh \"${ENVS}\"" >&2
    exit 2
  fi
done

for difficulty in $ENVS; do
  for split in train validation test; do
    case "$split" in
      train)
        seed_start=0
        seed_stop=500
        ;;
      validation)
        seed_start=1000
        seed_stop=1100
        ;;
      test)
        seed_start=2000
        seed_stop=2100
        ;;
      *)
        echo "Unknown split: $split" >&2
        exit 2
        ;;
    esac

    echo "Submitting difficulty=$difficulty split=$split seeds=[$seed_start,$seed_stop)"
    sbatch experiments/build_encoder_dataset.slurm \
      "$split" "$seed_start" "$seed_stop" \
      run.mode=dataset \
      encoder_dataset_difficulty="$difficulty" \
      encoder_dataset_split="$split"
  done
done

echo "Submitted all dataset jobs (${ENVS})."
