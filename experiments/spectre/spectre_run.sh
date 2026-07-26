#!/usr/bin/env bash
# Launch a long SPECTRE job in the background with its output in a known place.
#
# Long runs (collections, training sweeps, cache builds) otherwise scatter their stdout
# somewhere ephemeral, so checking on one means remembering a path. This puts every job at
# data/spectre/logs/<name>.log, which is where spectre_status.py looks.
#
#   bash experiments/spectre/spectre_run.sh v3_train \
#        python -m alphatamp.approaches.spectre.train_v2 --env dd2d_v4 --seed 0 --evidence
#   python experiments/spectre/spectre_status.py
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <job-name> <command...>" >&2
  exit 2
fi
NAME="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO/data/spectre/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$NAME.log"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
{
  echo "### $NAME started $(date -Is)"
  echo "### cmd: $*"
} >> "$LOG"
# stdbuf keeps the heartbeats line-buffered so the log is readable *while* running
nohup stdbuf -oL -eL "$@" >> "$LOG" 2>&1 &
echo "launched $NAME (pid $!) -> $LOG"
echo "check with: python experiments/spectre/spectre_status.py"
