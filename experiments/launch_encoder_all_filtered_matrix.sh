#!/usr/bin/env bash
set -euo pipefail

# Submit Stage A/B/C (all_filtered mode) jobs for o2/o3/o4.
#
# all_filtered does:
#   A) full vocab build on vocab.seed range
#   B) small filter-seed dataset build on full vocab
#   C) offline vocab filtering (remove never-successful-when-applicable for
#      threshold=0.0)
#
# Usage:
#   bash experiments/launch_encoder_all_filtered_matrix.sh
#
# Outputs per difficulty:
#   artifacts/encoder_oX/encoder_vocab_filtered_all_filtered.pkl
#   artifacts/encoder_oX/encoder_filter_dataset.pkl
#   artifacts/encoder_oX/encoder_filter_dataset_filtered.pkl

cd "$(dirname "$0")/.."

for difficulty in o2 o3 o4; do
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

echo "Submitted all all_filtered jobs (o2/o3/o4)."
