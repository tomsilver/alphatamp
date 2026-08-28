#!/usr/bin/env bash
# Train SPECTRE on the restock3d_v3_real REAL (hybrid-prune PyBullet) dataset — 3 seeds.
#
# This is the SPECTRE-only intermediate: PIGINet/LAZY are NOT retrained on the real data yet
# (the SPECTRE-vs-PIGINet real-label representation-crossover audit is deferred to the full
# collection). It reuses the deployed synthetic-v3 recipe VERBATIM (restock3d_v3_train.sh) so
# the intermediate is directly comparable once PIGINet/LAZY land:
#   --scene-3d (3D point cloud) + --atom-mode profiles (init+goal atoms) + PointSet/coverage/
#   records + the F3 exact-step certificate --repeat-feats (the load-bearing adaptivity here).
# Lands in checkpoints_spectre_atoms_repeat/restock3d_v3_real/seed_<s>/best.pt (train.py appends
# `--out-suffix _repeat` and inserts /<env>/seed_<n>/).
#
# NOTE: domain.py must map "restock3d_v3_real" -> _RESTOCK3D_V3 or --repeat-feats is inert
# (the val-selection rollout fetches the spec by this variant name).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

SPECTRE_FLAGS="--epochs 30 --scene-3d --atom-mode profiles --use-pca-feats --use-edgeconv \
--use-point-sab --pma-seeds 4 --overlap-mode jaccard --coverage-feats --coverage-mode both \
--aggregate-records --evidence-attn --state-delta --select-window 5 --repeat-feats --step-join \
--out-suffix _repeat"

echo "### SPECTRE 3 seeds (concurrent) on restock3d_v3_real $(date -Is)"
for s in 0 1 2; do
  # shellcheck disable=SC2086
  python -m alphatamp.approaches.spectre.train --env restock3d_v3_real --seed "$s" $SPECTRE_FLAGS \
    > "$REPO/data/spectre/logs/restock3d_v3_real_spectre_s$s.log" 2>&1 &
done
wait
echo "### SPECTRE done $(date -Is)"
