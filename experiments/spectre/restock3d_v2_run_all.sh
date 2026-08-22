#!/usr/bin/env bash
# Restock3D v2 -- SEQUENTIAL, GATED, single-stratum collection, then vocab + smoke-train.
#
# Runs the collector once per banding stratum in strata_v2.SEQUENTIAL_ORDER (2x2, 3x3, 4x3,
# 3x4, 4x4), FOREGROUND/blocking so stratum n+1 starts only after n finishes. A fresh process
# per stratum guarantees full RAM reclamation, so each job auto-sizes workers to its own
# uniform, predictable per-worker RAM peak (min(0.85*CPU, 0.85*freeRAM/per_worker_gb),
# floor-guarded). Then builds vocab + smoke-trains SPECTRE/LAZY/PIGINet (the readiness gate).
# Resumes automatically: the collector pre-scans on-disk episodes, so re-running skips whatever
# is already collected.
#
#   bash experiments/spectre/spectre_run.sh restock3d_v2_collect \
#        bash experiments/spectre/restock3d_v2_run_all.sh
#
# Logs (via spectre_run.sh) to data/spectre/logs/restock3d_v2_collect.log.
set -u  # NOT -e: a per-stratum SHORTFALL / nonzero exit must not abort the remaining strata.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

log() { echo "[run_all $(date -Is)] $*"; }

ORDER=$(python -c 'from alphatamp.approaches.spectre.envs.restock3d import strata_v2 as S; print(*S.SEQUENTIAL_ORDER)')
N=$(echo "$ORDER" | wc -w)
log "sequential collection order (banding strata): $ORDER   ($N jobs)"
t_all=$(date +%s)

i=0
for s in $ORDER; do
  i=$((i + 1))
  free=$(free -g | awk 'NR==2{print $7}')
  log "=== stratum s=$s ($i/$N) START  freeRAM=${free}GB ==="
  t0=$(date +%s)
  python experiments/spectre/restock3d_v2_collect.py --strata "$s"
  rc=$?
  el=$(( ($(date +%s) - t0) / 60 ))
  log "=== stratum s=$s ($i/$N) DONE in ${el}m (exit=$rc; census above) ==="
done

# R4 guard: vocab's max_pool_size / max_skeleton_length are computed over the train split; a
# stratum with 0 train episodes would silently undersize downstream models. Warn loudly.
log "per-stratum train-episode census (0 => vocab would undersize):"
python - <<'PY'
from pathlib import Path
from alphatamp.approaches.spectre.envs.restock3d import strata_v2 as S
ep = Path("data/spectre/raw") / S.ENV_VARIANT / "train" / "episodes"
counts = {s: 0 for s in S.STRATA}
if ep.exists():
    for f in ep.glob("ep_*.pkl.gz"):
        try:
            pid = int(f.name.removeprefix("ep_").split(".")[0])
        except ValueError:
            continue
        counts[S.stratum_of(pid)] += 1
for s in S.STRATA:
    flag = "  <-- WARNING: 0 train episodes" if counts[s] == 0 else ""
    print(f"  s{s} ({'x'.join(map(str, S.CONFIGS[s]))}): train={counts[s]}/{S.sizes(s)['train']}{flag}")
PY

el_all=$(( ($(date +%s) - t_all) / 60 ))
log "=== all strata collected in ${el_all}m; running finalize (vocab + smoke-train) ==="
bash experiments/spectre/restock3d_v2_finalize.sh   # no pid arg -> skips wait, runs vocab + train
log "=== run_all COMPLETE ==="
