"""One command to answer "what is running and when will it finish?".

Long SPECTRE jobs (collections, training sweeps, cache builds) are launched in the
background, so their stdout lands somewhere ephemeral and checking on them means
remembering a path. This reads a canonical log directory plus the process table and prints
a compact status with ETAs.

It deliberately *reads* rather than *instruments*: ``train_v2`` and the DD2D collector are
frozen under the v3 migration's D-7 rule, and both already emit periodic heartbeats with an
ETA. So this parses what they print instead of adding calls inside them, which also means it
works for jobs that were already running when it was written.

Usage::

    python experiments/spectre/spectre_status.py          # snapshot
    python experiments/spectre/spectre_status.py --watch  # refresh every 20 s

Launch long jobs with ``experiments/spectre/spectre_run.sh <name> <cmd...>`` so their output
lands in ``data/spectre/logs/<name>.log`` where this can find it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "data" / "spectre" / "logs"

# Training heartbeat from either trainer, e.g.
#   [train_v2] seed=0 epoch 15/30 ... relrank=1.428 best=1.428 * ... ETA 4.2m
#   [train_v3] seed=0 epoch 15/30 ... val_fp=17.20 ma=17.9 best=17.9 * ... ETA 4.2m
# The selection metric differs between them (relrank vs deployed val FP), so it is
# captured generically as "whatever this trainer calls best".
_TRAIN = re.compile(
    r"\[(?P<who>train_v[23])\]\s+seed=(?P<seed>\d+)\s+epoch\s+(?P<ep>\d+)/(?P<tot>\d+)"
    r".*?(?:best=(?P<best>[\d.]+))?.*?(?:ETA\s+(?P<eta>[\d.]+)m)?$"
)
# collector heartbeat, e.g. "  [train] 12m30s | kept 204/400  (s0 100/100, ...) | ETA 33m"
_COLLECT = re.compile(
    r"\[(?P<split>train|val|test)\]\s+(?P<elapsed>[\dhms]+)\s+\|\s+kept\s+"
    r"(?P<kept>\d+)/(?P<target>\d+)"
)


def _running() -> list[tuple[str, str, str]]:
    """``(pid, elapsed, cmd)`` for live SPECTRE jobs."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etime,cmd"], capture_output=True, text=True, check=True
        ).stdout
    except Exception:  # pragma: no cover - ps should exist
        return []
    keys = (
        "spectre.train",
        "dd2d.drawer.collect",
        "sb2d_render_convert",
        "precompute_dd2d_cache",
        "vlmplan_",
    )
    rows = []
    for line in out.splitlines()[1:]:
        if any(k in line for k in keys) and "spectre_status" not in line:
            pid, etime, cmd = line.split(None, 2)
            rows.append((pid, etime, cmd.strip()))
    return rows


def _tail(path: Path, n: int = 4000) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def _log_status(path: Path) -> str | None:
    """The most informative recent line of a job log, condensed."""
    for line in reversed(_tail(path)):
        m = _TRAIN.search(line)
        if m:
            eta = f"ETA {m['eta']}m" if m["eta"] else "ETA ?"
            best = f" best={m['best']}" if m["best"] else ""
            return (
                f"{m['who']} seed={m['seed']} epoch {m['ep']}/{m['tot']}{best}  {eta}"
            )
        m = _COLLECT.search(line)
        if m:
            return (
                f"collect [{m['split']}] kept {m['kept']}/{m['target']} "
                f"({m['elapsed']} elapsed)"
            )
        if "done:" in line or "complete" in line.lower():
            return line.strip()[:110]
    return None


def _checkpoints() -> list[tuple[str, str]]:
    """Completed training runs, newest first, with their selection metric."""
    root = REPO / "data" / "spectre"
    rows: list[tuple[float, str, str]] = []
    for log in root.glob("checkpoints*/*/seed_*/log.jsonl"):
        try:
            recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        if not recs:
            continue
        key = "val_fp" if "val_fp" in recs[0] else "val_relrank"
        best = min(r.get(key, float("inf")) for r in recs)
        rel = log.relative_to(root).parent
        rows.append(
            (
                log.stat().st_mtime,
                str(rel),
                f"{len(recs)} epochs, best {key} {best:.3f}",
            )
        )
    rows.sort(reverse=True)
    return [(name, info) for _, name, info in rows[:8]]


def snapshot() -> str:
    out = [f"=== SPECTRE status @ {datetime.now():%H:%M:%S} ==="]

    live = _running()
    out.append(f"\n-- running ({len(live)}) --")
    if not live:
        out.append("   (nothing running)")
    for pid, etime, cmd in live:
        short = cmd.replace("python -m alphatamp.approaches.spectre.", "")
        out.append(f"   pid {pid:<8} up {etime:>9}  {short[:96]}")

    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: -p.stat().st_mtime)
    out.append(f"\n-- job logs ({LOG_DIR}) --")
    if not logs:
        out.append("   (none; launch via experiments/spectre/spectre_run.sh)")
    for log in logs[:6]:
        age = (time.time() - log.stat().st_mtime) / 60
        status = _log_status(log) or "(no parsable progress line yet)"
        fresh = "*" if age < 2 else " "
        out.append(f"  {fresh}{log.stem:<28} {status}   [{age:.0f}m since write]")

    out.append("\n-- recent completed checkpoints --")
    for name, info in _checkpoints():
        out.append(f"   {name:<52} {info}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--watch", action="store_true", help="refresh until interrupted")
    ap.add_argument("--interval", type=float, default=20.0)
    a = ap.parse_args(argv)
    if not a.watch:
        print(snapshot())
        return 0
    try:
        while True:
            print("\033[2J\033[H" + snapshot(), flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
