"""Run several SPECTRE v3 training configurations concurrently on one GPU.

Training here is **CPU-bound, not GPU-bound**: measured on dd2d_v4, tensorization is
~79% of a step and the model needs 173 MB of the card's 32 GB. Running arms one after
another therefore leaves both the GPU and 30-odd CPU threads idle. This launches them
together.

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

#: Shared backbone for the adaptive-feature ablation (2026-08-21): the deployed recipe
#: minus every failure-conditioned feature -- jaccard overlap kept as the constant backbone,
#: plus the point-set-encoder upgrade, atom profiles and the ma5 selector. Each ablation arm
#: adds exactly one feature on top, trained from scratch (not train-all-then-disable), so a
#: feature that looks inert is not being masked by another compensating for it.
_ABL_BACKBONE = (
    "--overlap-mode jaccard --use-pca-feats --use-edgeconv --use-point-sab "
    "--pma-seeds 4 --atom-mode profiles --select-window 5"
)


def _ablation_arms(extra: str = "") -> dict[str, str]:
    """The six ablation arms (``extra`` = env-specific backbone add, e.g. ``" --scene-3d"``).

    ``abl_floor`` is the Δ reference (jaccard backbone only); ``abl_all`` is the
    current-architecture "all adaptive features on" (the stale DD2D/SB2D deployed
    checkpoints predate the point-set upgrade, so nothing is reused). ``repeat`` fires on
    DD2D/SB2D via the ``domain.py`` ``step_certificate`` declarations (place-buffer / press
    schemas). Run ONE seed at a time (``--seeds 0``, then ``--seeds 1`` / ``2``) so each
    seed lands in ``checkpoints_spectre[_norec]_atoms_<arm>/<env>/seed_<n>/`` -- a multi-seed
    sweep would instead suffix ``_s<n>`` onto the dir name and split the seeds apart.
    """
    bb = _ABL_BACKBONE + extra
    return {
        "abl_floor": f"{bb} --no-records",
        "abl_only_cov": f"{bb} --coverage-feats --coverage-mode coverage --no-records",
        "abl_only_waste": f"{bb} --coverage-feats --coverage-mode waste --no-records",
        "abl_only_repeat": f"{bb} --repeat-feats --no-records",
        "abl_only_records": f"{bb} --aggregate-records --evidence-attn --state-delta",
        "abl_all": (
            f"{bb} --coverage-feats --coverage-mode both --repeat-feats "
            "--aggregate-records --evidence-attn --state-delta"
        ),
    }


#: Named sweeps. Each entry is ``name -> extra CLI args for train.py``.
#: G6 holds ``cand_overlap`` out of *both* the record and no-record arms, so the
#: evidence increment is not measured against a bar contaminated by the same
#: set-overlap signal.
PRESETS: dict[str, dict[str, str]] = {
    "g6": {
        "g6_recON_ovOFF": "--no-overlap",
        "g6_recOFF_ovOFF": "--no-overlap --no-records",
        "g6_recON_ovON": "",
    },
    # G6b re-runs G6 with the *only* change being an uncensored, whole-split selector
    # (now the `train.py` default, hence no extra args). G6's censored-at-30 selector
    # scored v2.2 and v3 within 0.3 FP of each other while they were 4+ FP apart on
    # test, so it was ranking epochs by noise. Separate output dirs: G6's checkpoints
    # are kept so the two selectors can be compared rather than one quietly overwriting
    # the other.
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
    # `jac` drops the `dead` column, which is a disguised shortness cue and the
    # suspected cause of the s1 regression.
    "g8": {
        "g8_jac": "--overlap-mode jaccard",
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
    },
    # P4: the records-first fix, and the one that targets the cause directly.
    # `suppress_records` showed the trained model discards its own record tokens;
    # EvidenceCrossAttentionScorer says why -- one softmax over [scene ; global ;
    # records] makes
    # evidence compete with geometry for attention mass, and geometry wins because it is
    # reliably useful. A separate channel removes the competition, so records can drive
    # adaptiveness rather than being ignored.
    # P5: the s3 fix. G8 showed `dead` is a length proxy -- right at s3, wrong at s1.
    # Rather than tune the proxy, state the thing it proxies for: at s3 three objects
    # block and the right candidate removes all three. `coverage`/`waste` say that
    # directly, from culprits the refiner REPORTED (so no predicted necessity head, and
    # no geometry routine of ours). Measured separation feasible-vs-infeasible grows
    # with stratum: coverage 0.139/0.160 at s0 -> 0.412/0.168 at s3.
    "p5": {
        "p5_jac_cov": "--overlap-mode jaccard --coverage-feats",
        "p5_cov": "--coverage-feats",
        "p5_jac_cov_evattn": "--overlap-mode jaccard --coverage-feats --evidence-attn",
    },
    # THE v3 deployed configuration (2026-07-28). Beats deployed v2.2 at every stratum:
    # 7.20 +/- 0.62 vs 17.27 +/- 3.02 over 3 seeds, -10.06 FP, CI [-12.83, -7.59].
    # Run with `--seeds 0 1 2` for the reportable number.
    #
    # `--state-delta` joined on 2026-07-28 (proposal §6.1's `s_j`). It is a TIE with the
    # pre-delta config, not a win -- 7.20 +/- 0.62 vs 7.44 +/- 0.23 at 3 seeds, and
    # 8.23 +/- 1.36 vs 7.90 +/- 0.61 at 6 -- and it is deployed because it completes the
    # record schema at no cost in a new environment, not because it improved the number.
    # `decisions.md` 2026-07-28. The pre-delta checkpoints survive as
    # `checkpoints_spectre_v3final_s{0..5}`; this preset now writes over that name, so use
    # `--out-suffix` if you need both on disk at once.
    # 2026-07-31: coverage/waste now use the **unified** definitions by default
    # (`TrainConfig.unified_coverage=True`), so this preset needs no extra flag and a
    # fresh run of it reproduces 5.78 +/- 0.10, not the 7.44 above.
    # `--select-window 5` added 2026-08-08: the domain-agnostic (narrowed-input) model
    # is higher-variance, and the default ma3 selection window locked onto unlucky val
    # epochs (the s1 regression). Widening to ma5 recovers parity with the frozen
    # baseline (5.92 vs 5.78, CI includes 0) and collapses the variance -- see
    # docs/decisions 2026-08-08. The `TrainConfig` default stays 3 so the frozen
    # baseline's provenance is untouched; the deployed recipe opts in here.
    # 2026-08-19: the PointSetEncoder upgrade and the AtomProfileEncoder are switched ON
    # in the deployed recipe (`--use-pca-feats --use-edgeconv --use-point-sab --pma-seeds 4
    # --atom-mode profiles`). The `SpectreConfig`/`TrainConfig` defaults stay off so the
    # config-off equivalence tests and old-checkpoint strict loads remain valid -- the
    # deployed recipe opts in here, exactly like every other feature flag above. NOTE:
    # `--atom-mode profiles` makes `train.py` append an `_atoms` suffix to the checkpoint
    # dir, so the frozen 5.78/6.29 numbers predate this change and a retrain is pending;
    # reconcile the downstream `spectre_score.py`/`compare.py` checkpoint-dir names when
    # that retrain lands. `--scene-3d` is intentionally NOT here (Restock3D-only widening).
    "v3final": {
        "v3final": (
            "--overlap-mode jaccard --coverage-feats "
            "--aggregate-records --evidence-attn --state-delta --select-window 5 "
            "--use-pca-feats --use-edgeconv --use-point-sab --pma-seeds 4 "
            "--atom-mode profiles"
        ),
    },
    # EMA weight-averaging arm of the deployed config, to recover the domain-agnostic
    # (narrowed-input) model's inflated across-seed variance without touching inputs or
    # architecture (docs/decisions 2026-08-08). Same flag set as v3final plus
    # `--weight-avg ema`; writes `checkpoints_spectre_v3ema_s{seed}` (a NEW location, so the
    # deployed `checkpoints_spectre_unified` is never touched until an arm is verified and
    # promoted). `sb2dema` is the StickButton2D twin for the cheap variance-attribution
    # triage. `--select-window 5` can be appended ad-hoc via `--arm` for the selector
    # probe.
    "v3ema": {
        "v3ema": (
            "--overlap-mode jaccard --coverage-feats --aggregate-records "
            "--evidence-attn --state-delta --weight-avg ema"
        ),
    },
    "sb2dema": {
        "sb2dema": (
            "--overlap-mode jaccard --coverage-feats --aggregate-records "
            "--evidence-attn --state-delta --weight-avg ema"
        ),
    },
    "p4": {
        "p4_evattn": "--evidence-attn",
        "p4_evattn_agg": "--evidence-attn --aggregate-records",
    },
    # `v3delta` is FOLDED INTO `v3final` above -- it is the same flag set, and keeping a
    # second name for the deployed config is how two "current" arms end up on disk. Its
    # checkpoints (`checkpoints_spectre_v3final_s{0..5}`, 6 seeds) are what the comparison
    # cache reads; `_V3_ARMS["spectre3"]` in `precompute_dd2d_cache.py` points at them.
    #
    # StickButton2D's ablation set, mirroring the six arms the DD2D notebook's section 4
    # reads. Each is the deployed config with exactly one thing toggled, so a
    # difference is
    # attributable. Run as:
    #
    #     python experiments/spectre/spectre_sweep.py --preset sb2dabl \
    #         --env stickbutton2d_v1 --seeds 0 1 2
    #
    # The **demotion arms are deliberately absent.** Proof-tier demotion was cut
    # from the method on 2026-07-30, and StickButton2D resolves to `EMPTY_SPEC`, so
    # `licenses_demotion` is always False there and a demotion arm would be
    # bit-identical to its base. Omitted because it is vacuous on this environment,
    # not overlooked.
    "sb2dabl": {
        # coverage x records, 2x2. `abl_cov_rec` is the deployed config itself; it is
        # trained under its own name so section 4's grid has all four cells from one
        # sweep
        # rather than three cells plus a cross-reference.
        "abl_cov_rec": (
            "--overlap-mode jaccard --coverage-feats "
            "--aggregate-records --evidence-attn --state-delta"
        ),
        "abl_cov_norec": (
            "--overlap-mode jaccard --coverage-feats "
            "--aggregate-records --evidence-attn --state-delta --no-records"
        ),
        "abl_nocov_rec": (
            "--overlap-mode jaccard --aggregate-records --evidence-attn --state-delta"
        ),
        "abl_nocov_norec": (
            "--overlap-mode jaccard --aggregate-records --evidence-attn "
            "--state-delta --no-records"
        ),
        # coverage vs waste, separated by zeroing one column (shape unchanged).
        "abl_cov_only": (
            "--overlap-mode jaccard --coverage-feats --coverage-mode coverage "
            "--aggregate-records --evidence-attn --state-delta"
        ),
        "abl_waste_only": (
            "--overlap-mode jaccard --coverage-feats --coverage-mode waste "
            "--aggregate-records --evidence-attn --state-delta"
        ),
    },
    # Single-feature-isolation ablation (2026-08-21) -- one preset per environment because
    # restock3d_v3 adds `--scene-3d` to the backbone and DD2D/SB2D do not. Run per env with a
    # single seed:  --preset ablation_dd2d --env dd2d_v4 --seeds 0  (SB2D trains under
    # `--env stickbutton2d_v1`; SPECTRE is image-free). See the ablation ADR / repeat census.
    "ablation_dd2d": _ablation_arms(),
    "ablation_sb2d": _ablation_arms(),
    "ablation_restock": _ablation_arms(" --scene-3d"),
    # Learned-pathway workstream (docs/failed_records_fix.md F-A/F-B2). On the SAME ablation
    # backbone as `abl_only_records` (the rung-0 tokens-only baseline), the increments are:
    # rung-1 evidence STEPS (`--record-mode steps`) and the pre-pooling StepJoin
    # (`--step-join`, the C2 fix — candidate step tokens join over the evidence memory before
    # pooling). `fr_summary` retrains the rung-0 baseline under identical code as the matched
    # control. Run per env one seed at a time (restock adds ` --scene-3d`):
    #   --preset failed_records --env dd2d_v4 --seeds 0
    "failed_records": {
        "fr_summary": (
            f"{_ABL_BACKBONE} --aggregate-records --evidence-attn --state-delta"
        ),
        "fr_steps": (
            f"{_ABL_BACKBONE} --aggregate-records --evidence-attn --state-delta "
            "--record-mode steps"
        ),
        "fr_join": (
            f"{_ABL_BACKBONE} --aggregate-records --evidence-attn --state-delta "
            "--step-join"
        ),
        "fr_steps_join": (
            f"{_ABL_BACKBONE} --aggregate-records --evidence-attn --state-delta "
            "--record-mode steps --step-join"
        ),
    },
    "failed_records_restock": {
        "fr_summary": (
            f"{_ABL_BACKBONE} --scene-3d --aggregate-records --evidence-attn --state-delta"
        ),
        "fr_steps": (
            f"{_ABL_BACKBONE} --scene-3d --aggregate-records --evidence-attn "
            "--state-delta --record-mode steps"
        ),
        "fr_join": (
            f"{_ABL_BACKBONE} --scene-3d --aggregate-records --evidence-attn "
            "--state-delta --step-join"
        ),
        "fr_steps_join": (
            f"{_ABL_BACKBONE} --scene-3d --aggregate-records --evidence-attn "
            "--state-delta --record-mode steps --step-join"
        ),
    },
}


def launch(name: str, extra: str, env: str, seed: int, epochs: int, workers: int):
    """Start one training arm detached, returning ``(process, log path, handle)``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{name}.log"
    cmd = [
        "python",
        "-m",
        "alphatamp.approaches.spectre.train",
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
