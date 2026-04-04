#!/usr/bin/env bash
set -euo pipefail

# Submit Stage A/B/C (all_filtered mode) jobs for one or more environments.
#
# all_filtered does:
#   A) full vocab build on vocab.seed range
#   B) small filter-seed dataset build on full vocab
#   C) offline vocab filtering (remove never-successful-when-applicable for
#      threshold=0.0)
#
# Usage:
#   bash experiments/launch_encoder_all_filtered_matrix.sh [ENVS]
#
#   ENVS  Space-separated list of encoder_dataset_difficulty config names.
#         Defaults to "o2 o3 o4".
#         Example: bash experiments/launch_encoder_all_filtered_matrix.sh "sb1 sb2 sb3"
#         Example: bash experiments/launch_encoder_all_filtered_matrix.sh "o2 o3 o4 sb1 sb2 sb3"
#
# Outputs per difficulty (routed to artifacts_ob/ or artifacts_sb/ automatically):
#   artifacts_{ob|sb}/encoder_<difficulty>/encoder_vocab_filtered_all_filtered.pkl
#   artifacts_{ob|sb}/encoder_<difficulty>/encoder_filter_dataset.pkl
#   artifacts_{ob|sb}/encoder_<difficulty>/encoder_filter_dataset_filtered.pkl

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
  echo "Submitting all_filtered for difficulty=$difficulty"
  sbatch experiments/build_encoder_dataset.slurm \
    all_filtered 0 1 \
    run.mode=all_filtered \
    encoder_dataset_difficulty="$difficulty" \
    vocab.seed_start=0 \
    vocab.seed_stop=500 \
    vocab.filter_seed_start=500 \
    vocab.filter_seed_stop=550 \
    vocab.filter_success_rate_threshold=0.0

done

echo "Submitted all all_filtered jobs (${ENVS})."
