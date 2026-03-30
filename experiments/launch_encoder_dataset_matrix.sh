#!/usr/bin/env bash
set -euo pipefail

# Submit train/validation/test dataset-build jobs for o2/o3/o4 using
# pre-built filtered vocab files.
#
# Usage:
#   bash experiments/launch_encoder_dataset_matrix.sh
#
# Notes:
# - Uses config groups:
#     encoder_dataset_difficulty={o2,o3,o4}
#     encoder_dataset_split={train,validation,test}
# - Assumes filtered vocab artifacts already exist at:
#     artifacts/encoder_o{2,3,4}/encoder_vocab_filtered_all_filtered.pkl

cd "$(dirname "$0")/.."

for difficulty in o2 o3 o4; do
  vocab_path="artifacts/encoder_${difficulty}/encoder_vocab_filtered_all_filtered.pkl"
  if [[ ! -f "$vocab_path" ]]; then
    echo "Missing filtered vocab artifact: $vocab_path" >&2
    echo "Run: bash experiments/launch_encoder_all_filtered_matrix.sh" >&2
    exit 2
  fi
done

for difficulty in o2 o3 o4; do
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

echo "Submitted all dataset jobs."
