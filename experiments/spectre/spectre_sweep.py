"""Run several SPECTRE v3 training configurations concurrently on one GPU.

Training here is **CPU-bound, not GPU-bound**: measured on dd2d_v4, tensorization is ~79%
of a step and the model needs 173 MB of the card's 32 GB. Running arms one after another
therefore leaves both the GPU and 30-odd CPU threads idle. This launches them together.

Two knobs matter and they interact:

- ``--max-parallel`` -- how many training processes run at once. Each is its own CUDA
  context (~300-500 MB of overhead), so VRAM is not the limit; CPU is.
- ``--num-workers`` -- dataloader workers *inside* each run. Total load is roughly
  ``max_parallel * (1 + num_workers)`` processes, so keep that under the core count or
  the runs start fighting each other and the wall-clock stops improving.

Each arm writes to ``data/spectre/logs/<name>.log``, which is where
``spectre_status.py`` looks, so a sweep stays checkable mid-flight.

Usage::

    python experiments/spectre/spectre_sweep.py --preset g6
    python experiments/spectre/spectre_sweep.py --preset g7 --max-parallel 4
    python experiments/spectre/spectre_sweep.py \\
        --arm "recA:--no-overlap" --arm "recB:--no-overlap --no-records"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import IO

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "data" / "spectre" / "logs"

#: Named sweeps. Each entry is ``name -> extra CLI args for train_v3``.
#: G6 holds ``cand_overlap`` out of *both* the record and no-record arms, so the evidence
#: increment is not measured against a bar contaminated by the same set-overlap signal.
PRESETS: dict[str, dict[str, str]] = {
    "g6": {
        "g6_recON_ovOFF": "--no-overlap",
        "g6_recOFF_ovOFF": "--no-overlap --no-records",
        "g6_recON_ovON": "",
    },
    # G6b re-runs G6 with the *only* change being an uncensored, whole-split selector
    # (now the `train_v3` default, hence no extra args). G6's censored-at-30 selector
    # scored v2.2 and v3 within 0.3 FP of each other while they were 4+ FP apart on test,
    # so it was ranking epochs by noise. Separate output dirs: G6's checkpoints are kept
    # so the two selectors can be compared rather than one quietly overwriting the other.
    "g6b": {
        "g6b_recON_ovOFF": "--no-overlap",
        "g6b_recOFF_ovOFF": "--no-overlap --no-records",
        "g6b_recON_ovON": "",
    },
    # G7's 2x2 isolates `jaccard` (genuinely uncertain) from `dead` (redundant with the
    # demotion applied outside the net). The demotion half of the 2x2 is an
    # evaluation-time switch, not a training one, so only the two training arms appear.
    "g7": {
        "g7_ovON": "",
        "g7_ovOFF": "--no-overlap",
    },
    # G8 (performance push): close the s1 and s3 gaps to v2.2 without giving up s2.
    # `jac` drops the `dead` column, which is a disguised shortness cue and the suspected
    # cause of the s1 regression; `tailF` shows training the |F| ~ 20-40 regime an s3
    # rollout actually visits; `both` combines them.
    "g8": {
        "g8_jac": "--overlap-mode jaccard",
        "g8_tailF": "--tail-max-f 40",
        "g8_jac_tailF": "--overlap-mode jaccard --tail-max-f 40",
    },
    # P2: the missing cell of the G6 ablation, plus record aggregation.
    # G6's "no records" bar also had overlap off, so it conflated two removals; `norec`
    # is the honest bar (overlap ON, records OFF) and is the closest v3 analogue of the
    # v2.2 yardstick, whose FactEncoder is inert on dd2d_v4 (no harvested facts).
    # `agg` collapses one-record-per-failed-sample to one per (schema, args): -88.7%
    # tokens, max 2045 -> 37.
    "p2": {
        "p2_norec": "--no-records",
        "p2_agg": "--aggregate-records",
        "p2_agg_tailF": "--aggregate-records --tail-max-f 40",
        "p2_agg_jac_tailF": "--aggregate-records --overlap-mode jaccard --tail-max-f 40",
    },
    # P3: the same record content, routed through the tag join instead of as free tokens
    # -- `obj_evidence` is computed purely from FailureRecord fields, so it is a record
    # *consumption* mechanism, not a replacement. `objev_norec` is its cleanest form.
    "p3": {
        "p3_objev": "--obj-evidence",
        "p3_objev_norec": "--obj-evidence --no-records",
        "p3_objev_tailF": "--obj-evidence --tail-max-f 40",
    },
    # P4: the records-first fix, and the one that targets the cause directly.
    # `suppress_records` showed the trained model discards its own record tokens;
    # CrossAttentionScorerV3 says why -- one softmax over [scene ; global ; records] makes
    # evidence compete with geometry for attention mass, and geometry wins because it is
    # reliably useful. A separate channel removes the competition, so records can drive
    # adaptiveness rather than being ignored.
    "p4": {
        "p4_evattn": "--evidence-attn",
        "p4_evattn_agg": "--evidence-attn --aggregate-records",
        "p4_evattn_agg_tailF": "--evidence-attn --aggregate-records --tail-max-f 40",
    },
}


def launch(name: str, extra: str, env: str, seed: int, epochs: int, workers: int):
    """Start one training arm detached, returning ``(process, log path, handle)``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{name}.log"
    cmd = [
        "python",
        "-m",
        "alphatamp.approaches.spectre.train_v3",
        "--env",
        env,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--num-workers",
        str(workers),
        "--out-suffix",
        f"_{name}",
        *extra.split(),
    ]
    fh = open(log, "a", encoding="utf-8")
    fh.write(f"### {name} started {time.strftime('%FT%T')}\n### cmd: {' '.join(cmd)}\n")
    fh.flush()
    # stdbuf keeps heartbeats line-buffered so the log is readable while running
    proc = subprocess.Popen(
        ["stdbuf", "-oL", "-eL", *cmd],
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=REPO,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc, log, fh


def main(argv: list[str] | None = None) -> int:
    """Run the requested arms, at most ``--max-parallel`` at a time."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        help='extra arm as "name:args", repeatable',
    )
    ap.add_argument("--env", default="dd2d_v4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=3)
    a = ap.parse_args(argv)

    arms: dict[str, str] = dict(PRESETS.get(a.preset, {})) if a.preset else {}
    for spec in a.arm:
        name, _, extra = spec.partition(":")
        arms[name.strip()] = extra.strip()
    if not arms:
        ap.error("nothing to run: pass --preset and/or --arm")

    jobs = [
        (f"{name}_s{seed}" if len(a.seeds) > 1 else name, extra, seed)
        for seed in a.seeds
        for name, extra in arms.items()
    ]
    load = a.max_parallel * (1 + a.num_workers)
    print(
        f"sweep: {len(jobs)} jobs, {a.max_parallel} at a time, "
        f"{a.num_workers} loader workers each (~{load} procs vs {os.cpu_count()} cores)"
    )
    for name, extra, seed in jobs:
        print(f"  {name:<24} seed={seed} args={extra or '(defaults)'}")
    print("\ncheck progress: python experiments/spectre/spectre_status.py\n")

    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, Path, IO[str]]] = []
    failed: list[str] = []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.max_parallel:
            name, extra, seed = pending.pop(0)
            proc, log, fh = launch(name, extra, a.env, seed, a.epochs, a.num_workers)
            running.append((name, proc, log, fh))
            print(f"[{time.strftime('%T')}] started {name} (pid {proc.pid}) -> {log}")
        time.sleep(5)
        for entry in list(running):
            name, proc, log, fh = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            fh.close()
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            if proc.returncode != 0:
                failed.append(name)
            print(f"[{time.strftime('%T')}] {name}: {status}")

    print(f"\nsweep done in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"FAILED arms: {', '.join(failed)} (see their logs)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
