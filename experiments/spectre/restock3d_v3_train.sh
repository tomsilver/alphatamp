#!/usr/bin/env bash
# Train SPECTRE / PIGINet / LAZY on the restock3d_v3 SYNTHETIC dataset (3 seeds each).
#
# SPECTRE uses the deployed restock3d_v2 recipe verbatim (--scene-3d 3D point cloud + --atom-mode
# profiles + PointSet/coverage/records), landing in checkpoints_spectre_atoms/restock3d_v3/. LAZY
# needs no per-variant code (geom_dim 9 auto via startswith("restock3d")). PIGINet reconstructs the
# oblique crops per seed via oracle_v3.build_v3_bundle. The dataset is tiny (400 train), so each run
# is fast; SPECTRE's 3 seeds run concurrently (CPU-bound), PIGINet's share one CLIP cache (built by
# the first seed) so they run sequentially.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
CKPT="data/spectre/checkpoints/restock3d_v3"

# Deployed recipe: the F3 exact-step certificate `--repeat-feats` (2026-08-21) revives the
# otherwise-inert adaptivity (adaptive 3.13 vs coverage-only 12.18, ~97% of the P2 oracle
# ceiling). Lands in checkpoints_spectre_atoms_repeat/ via `--out-suffix _repeat`. `regroup`
# is deprecated/off (adds nothing on v3; to be removed in a later refactor).
# `--step-join` (2026-08-23, learned-pathway workstream): the pre-pooling step-join is
# scalar-free + off-byte-identical / on-zero-init-additive. Adopted from the deployed recipe
# for parity; on restock3d_v3 its effect is UNMEASURED (measured only on DD2D) -- re-measure
# on the next v3 retrain.
SPECTRE_FLAGS="--epochs 30 --scene-3d --atom-mode profiles --use-pca-feats --use-edgeconv \
--use-point-sab --pma-seeds 4 --overlap-mode jaccard --coverage-feats --coverage-mode both \
--aggregate-records --evidence-attn --state-delta --select-window 5 --repeat-feats --step-join \
--out-suffix _repeat"

echo "### SPECTRE 3 seeds (concurrent) $(date -Is)"
for s in 0 1 2; do
  # shellcheck disable=SC2086
  python -m alphatamp.approaches.spectre.train --env restock3d_v3 --seed "$s" $SPECTRE_FLAGS \
    > "$REPO/data/spectre/logs/restock3d_v3_spectre_s$s.log" 2>&1 &
done
wait
echo "### SPECTRE done $(date -Is)"

echo "### PIGINet 3 seeds (sequential; shared CLIP cache) $(date -Is)"
for s in 0 1 2; do
  python -m alphatamp.approaches.spectre.baselines.piginet.train \
    --domain restock3d --env-variant restock3d_v3 --data-root data/spectre \
    --cache-dir "$CKPT/piginet_clip_cache" --out "$CKPT/piginet_s$s" --seed "$s"
done
echo "### PIGINet done $(date -Is)"

echo "### LAZY 3 seeds (sequential) $(date -Is)"
for s in 0 1 2; do
  python -m alphatamp.approaches.spectre.baselines.lazy.train --env-variant restock3d_v3 --seed "$s"
done
echo "### training done $(date -Is)"
