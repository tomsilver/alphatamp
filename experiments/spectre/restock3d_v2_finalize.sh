#!/usr/bin/env bash
# Restock3D v2: after the full collection finishes, build vocab and low-epoch smoke-train
# all three trainable methods (readiness gate). Waits on the collector PID so it can be
# launched immediately after the collection and completes the scope unattended.
#
#   bash experiments/spectre/restock3d_v2_finalize.sh <collector_pid>
#
# Logs to data/spectre/logs/restock3d_v2_finalize.log (spectre_status-visible). Every stage
# runs even if an earlier one fails (set +e); each stage's exit code is logged. The smoke
# checkpoints are throwaway -- the gate is "a checkpoint appears + the run exits 0", NOT FP.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
PIDARG="${1:-}"

log() { echo "[finalize $(date -Is)] $*"; }

# 1. Wait for the collection to finish (poll the collector PID if given).
if [ -n "$PIDARG" ]; then
  log "waiting for collector pid $PIDARG to exit ..."
  while kill -0 "$PIDARG" 2>/dev/null; do sleep 300; done
  log "collector pid $PIDARG has exited; proceeding."
fi

# Census snapshot so a SHORTFALL is visible in this log too.
for sp in train val test; do
  n=$(ls "data/spectre/raw/restock3d_v2/$sp/episodes/" 2>/dev/null | wc -l)
  log "collected $sp: $n episodes"
done

set +e
# 2. Vocab (train split only; OOV-checks val/test).
log "=== vocab ==="
python experiments/spectre/spectre_build_vocab.py env=restock3d_v2
log "vocab exit=$?"

CKPT="data/spectre/checkpoints/restock3d_v2"
mkdir -p "$CKPT"

# 3a. SPECTRE 3D + PointSetEncoder (the load-bearing readiness proof).
log "=== smoke: SPECTRE (3D PointSetEncoder) ==="
python -m alphatamp.approaches.spectre.train --env restock3d_v2 --seed 0 --epochs 3 \
    --scene-3d --use-pca-feats --use-edgeconv --out-suffix smoke
log "spectre exit=$?"

# 3b. LAZY (9-dim height graph).
log "=== smoke: LAZY (9-dim height graph) ==="
python -m alphatamp.approaches.spectre.baselines.lazy.train \
    --env-variant restock3d_v2 --seed 0 --epochs 3 --out-dir "$CKPT/lazy_smoke_s0"
log "lazy exit=$?"

# 3c. PIGINet (oblique height-visible crops).
log "=== smoke: PIGINet ==="
python -m alphatamp.approaches.spectre.baselines.piginet.train \
    --domain restock3d --env-variant restock3d_v2 --data-root data/spectre \
    --cache-dir "$CKPT/piginet_clip_cache" --out "$CKPT/piginet_smoke" --epochs 3
log "piginet exit=$?"

log "=== DONE: readiness gate complete (check exit codes + that checkpoints exist) ==="
ls -la "data/spectre/checkpoints_"*"/restock3d_v2/"* 2>/dev/null
ls -la "$CKPT"/*/ 2>/dev/null
