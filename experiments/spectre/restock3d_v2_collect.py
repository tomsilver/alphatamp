"""SPECTRE collector for **restock3d_v2** (continuous packing) -- the full dataset.

Collects the five configs -- symmetric 2x2, 3x3 at 50/15/15 and crowded 3x4, 4x3, 4x4 at
**25/10/10** (``strata_v2.SIZES``; 175 train / 60 val / 60 test = 295) into
``data/spectre/raw/restock3d_v2/{train,val,test}`` -- full ``EpisodeRecord``s (pool +
per-candidate labels + 3D scene geometry + instrumented F2/F3 failures), the same schema
DD2D/SB2D train on. **No oracle anywhere** -- the pool comes from the geometry-informed
nearest-first generator (the deployed v2 prior), refined non-short-circuiting; a problem is
kept iff >=1 candidate refines (``num_success >= 1``).

**Deployed as SEQUENTIAL single-stratum jobs** (``restock3d_v2_run_all.sh`` invokes this once
per stratum in ``strata_v2.SEQUENTIAL_ORDER``). One block count per process => a uniform,
predictable per-worker RAM peak, so each job is sized to its own safe concurrency and fully
reclaims memory before the next. This replaced a single mixed job whose peak was unpredictable
and OOM-killed the desktop (decisions/07 2026-08-19).

**Per-stratum budgets** come from ``strata_v2.BUDGETS`` (``(K_max, r_cap_s)``), calibrated on
the pilot's real collection-path feasible-solve tails (BacktrackingRefiner, not the oracle
certifier). Because infeasible candidates do NOT fail fast (backtracking re-descends), each
burns ~= r_cap, so per-problem cost ~= K_max x r_cap.

**Concurrency is RAM-bound, not CPU-bound.** ``--workers`` defaults to
``min(0.85*CPU, 0.85*freeRAM / strata_v2.PER_WORKER_GB[stratum])`` (floor-guarded), computed
per invocation from live free RAM -- ~27 for 2x2, ~15 for 3x3, ~10 for a crowded stratum on a
~55 GB box. A memory watchdog pauses submissions if free RAM dips below ``--mem-floor-gb``; the
heartbeat prints a per-stratum ETA, free RAM, and ``wRSSmax`` (max worker RSS, the live check on
``PER_WORKER_GB``). Full run ~1.5 days.

**Reject-resample is a dynamic top-up loop:** each (split, stratum) cell keeps ``target``
tasks in flight until ``target`` are kept (or a hard draw cap trips a SHORTFALL), so any
per-config yield self-corrects (4x4 is ~40-55% packable). The census trims each cell to the
first ``target`` kept *in index order*, so the dataset is exact + reproducible.

Refinement is real PyBullet motion planning, so tasks run in a spawn process pool; each worker
is single-thread-BLAS. Re-running resumes: a pre-scan seeds each cell from on-disk episodes so
a completed cell submits nothing and continues from the next index.

    bash experiments/spectre/restock3d_v2_run_all.sh            # deployed: all 5 strata, gated
    python experiments/spectre/restock3d_v2_collect.py --strata 3   # one stratum, auto-sized
    python experiments/spectre/restock3d_v2_collect.py --strata 0 --train 1 --val 1 --test 1 --workers 2  # smoke
"""

from __future__ import annotations

# Single-thread BLAS per worker (set before numpy/torch import) so N workers do not
# oversubscribe the cores.
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import math
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import psutil

from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.restock3d import strata_v2 as S

_HEARTBEAT_S = 30.0

# Absolute reserve (GB) below which the RAM-based worker cap never dips -- this is what keeps a
# job's peak headroom above the `--mem-floor-gb` watchdog floor even when the 0.85-utilization
# term would otherwise size too aggressively at low free RAM. See decisions/07 2026-08-19.
RESERVE_GB = 10.0


def _sized_workers(strata, cpu: int, avail_gb: float, mem_floor_gb: float) -> int:
    """Per-stratum RAM-sized worker count: ``min(0.85*CPU, 0.85*freeRAM /
    per_worker_peak)``.

    A worker's peak RSS is one problem's ``bpg`` scratchpad at its heaviest, which is uniform
    within a single-stratum job (one block count) but differs sharply across strata -- so we
    size from THAT stratum's ``per_worker_gb`` estimate. The RAM term targets 85% free-RAM
    utilization but is floor-guarded: it never leaves less than ``mem_floor_gb + 3`` GB free, so
    the peak stays above the watchdog floor. When several strata are passed (legacy path) the
    most memory-hungry one sets the cap. The per-stratum estimates are conservative upper bounds
    validated live by the collector's ``wRSSmax`` heartbeat.
    """
    pwg = max(S.per_worker_gb(s) for s in strata)
    ram_budget = min(0.85 * avail_gb, avail_gb - (mem_floor_gb + 3.0))
    by_ram = int(ram_budget / pwg)
    by_cpu = int(0.85 * cpu)
    return max(1, min(by_cpu, by_ram))


def _config(
    stratum: int, split: str, k_max: int, timeout_s: float, samples: int
) -> CollectionConfig:
    key = S.recipe_key(stratum)  # committed generator.STRATA key (11-15)
    n_tall, n_short = S.CONFIGS[stratum]
    return CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant=S.ENV_VARIANT,
        model_name="restock3d_v2",
        # Recipe rides in model_kwargs (stratum key) so config_hash + git_sha pin the
        # composition; n_tall/n_short are documentation (create_restock3d_v2_models reads
        # only `stratum`).
        model_kwargs={"stratum": key, "n_tall": n_tall, "n_short": n_short},
        split=split,
        num_problems=1,
        problem_seed_start=S.problem_id(split, stratum, 0),
        problem_seed_end=S.problem_id(split, stratum, 0) + 1,
        K_max=k_max,
        plan_generator="closed_form",  # -> v2 geometry-guided generator (collect.py dispatch)
        abstract_plan_timeout_s=30.0,
        refinement_timeout_s=timeout_s,
        num_sampling_attempts_per_step=samples,
        max_trajectory_steps=500,
    )


def _task(args) -> dict:
    """Worker: collect one problem, keep iff it has a feasible skeleton, write if
    kept."""
    stratum, split, index, k_max, timeout_s, samples, data_root = args
    from alphatamp.approaches.spectre.collect import collect_episode, episode_path
    from alphatamp.approaches.spectre.io import atomic_write_pickle_gz

    cfg = _config(stratum, split, k_max, timeout_s, samples)
    pid = S.problem_id(split, stratum, index)
    path = episode_path(Path(data_root), cfg.env_variant, split, pid)
    if path.exists():
        return {
            "stratum": stratum,
            "split": split,
            "index": index,
            "cached": True,
            "kept": True,
        }
    try:
        t0 = time.perf_counter()
        ep = collect_episode(cfg, pid)
        wall = time.perf_counter() - t0
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        return {
            "stratum": stratum,
            "split": split,
            "index": index,
            "error": f"{type(exc).__name__}: {exc}",
        }
    kept = ep.summary.num_success >= 1
    if kept:
        atomic_write_pickle_gz(ep, path)
    # Worst *feasible-candidate* refine wall (the collection-path tail the r_cap must clear);
    # surfaced so the heartbeat can flag a stratum whose feasible solves approach r_cap.
    feas = [
        getattr(o, "refinement_wall_clock_s", 0.0)
        for o in ep.outcomes
        if o.outcome == "success"
    ]
    return {
        "stratum": stratum,
        "split": split,
        "index": index,
        "kept": kept,
        "cached": False,
        "n_skel": len(ep.skeleton_pool),
        "n_succ": ep.summary.num_success,
        "n_fail": ep.summary.num_fail,
        "wall": round(wall, 1),
        "feas_max": round(max(feas), 1) if feas else None,
    }


def _p90(xs: list[float]) -> float:
    """Simple p90 of a non-empty list."""
    ss = sorted(xs)
    return ss[min(len(ss) - 1, int(0.9 * (len(ss) - 1)))]


def _worker_rss_gb() -> tuple[float, float]:
    """(max, sum) RSS in GB across this process's worker children -- the empirical check
    on ``strata_v2.PER_WORKER_GB``.

    Read-only; returns (0, 0) if children can't be read.
    """
    try:
        kids = psutil.Process().children(recursive=True)
        rss = [k.memory_info().rss for k in kids if k.is_running()]
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        Exception,
    ):  # pylint: disable=broad-except
        return 0.0, 0.0
    if not rss:
        return 0.0, 0.0
    return max(rss) / 1e9, sum(rss) / 1e9


def _heartbeat(cells: dict, t_start: float, workers: int) -> None:
    """Per-cell progress + a rough ETA from the observed keep-rate throughput."""
    el = time.perf_counter() - t_start
    total_target = sum(c["target"] for c in cells.values())
    total_kept = sum(c["kept"] for c in cells.values())
    total_done = sum(len([r for r in c["results"]]) for c in cells.values())

    # ETA: PER-STRATUM mean per-task wall (2x2 ~13min vs 4x4 ~100min -- a global mean badly
    # misprices the heavy tail), falling back to the nominal K_max*r_cap prior before a
    # stratum has completions; summed over the estimated remaining tasks, spread over workers.
    def _stratum_mean_wall(stratum: int) -> float:
        ws = [
            r["wall"]
            for ck, c2 in cells.items()
            if ck[1] == stratum
            for r in c2["results"]
            if "wall" in r
        ]
        if ws:
            return sum(ws) / len(ws)
        k, rc = S.budget(stratum)
        # Prior: most candidates burn ~r_cap (+ cooperative-timeout overshoot). Matches the
        # 2x2 probe (20*40*1.05=840s vs observed 789s).
        return k * rc * 1.05

    rem_core_s = 0.0
    for ck, c in cells.items():
        stratum = ck[1]
        drawn_done = len([r for r in c["results"] if not r.get("cached")])
        yield_ = max((c["kept"] / drawn_done) if drawn_done else 0.5, 0.05)
        rem = max(0, c["target"] - c["kept"])
        rem_tasks = min(rem / yield_, c["cap"] - c["drawn"])
        rem_core_s += rem_tasks * _stratum_mean_wall(stratum)
    eta = rem_core_s / max(1, workers)
    eta_txt = f"{eta/3600:.1f}h" if eta >= 5400 else f"{eta/60:.0f}m"
    avail_gb = psutil.virtual_memory().available / 1e9
    w_max, w_sum = _worker_rss_gb()
    print(
        f"[collect] {total_kept}/{total_target} kept ({total_done} done) "
        f"elapsed={el/60:.1f}m eta~{eta_txt} freeRAM={avail_gb:.1f}GB "
        f"wRSSmax={w_max:.1f}GB wRSSsum={w_sum:.1f}GB",
        flush=True,
    )
    for cell_key in cells:
        split, stratum = cell_key
        c = cells[cell_key]
        nt, ns = S.CONFIGS[stratum]
        _kmax, rcap = S.budget(stratum)
        drawn_done = len([r for r in c["results"] if not r.get("cached")])
        yld = f"{100*c['kept']/drawn_done:.0f}%" if drawn_done else "-"
        fmaxes = [r["feas_max"] for r in c["results"] if r.get("feas_max") is not None]
        if fmaxes:
            worst = max(fmaxes)
            near = "  <-- feasible-solve NEAR r_cap" if worst >= 0.9 * rcap else ""
            ftxt = (
                f"feasμ_p90={_p90(fmaxes):.0f}s worst={worst:.0f}s/rcap{rcap:.0f}{near}"
            )
        else:
            ftxt = "feas=-"
        print(
            f"    {split:5s} s{stratum}({nt}x{ns}): kept={c['kept']}/{c['target']} "
            f"drawn={c['drawn']} inflight={c['inflight']} yield={yld} err={c['errors']} "
            f"{ftxt}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    # Cell listing order. Submission is ROUND-ROBIN across cells (see the initial-fill loop
    # in main), so the running set is a MIX of strata -- memory ramps smoothly and the
    # never-collected 3x4/4x3 budgets are still exercised in the first heartbeats, without the
    # all-heavy spike that OOM-killed the first run. Order here only breaks ties in the mix.
    # Deployed as SEQUENTIAL single-stratum jobs (restock3d_v2_run_all.sh passes one `--strata`),
    # so each process has one block count and a uniform, predictable per-worker RAM peak. A list
    # still works (legacy mixed mode) but sizing then uses the most memory-hungry stratum.
    ap.add_argument("--strata", type=int, nargs="+", default=list(S.SEQUENTIAL_ORDER))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    # Sizes default to the PER-STRATUM strata_v2.SIZES (light 50/15/15, heavy 25/10/10); a
    # non-None value overrides that split's target across every passed stratum (dev/smoke).
    ap.add_argument(
        "--train", type=int, default=None, help="override per-stratum train size"
    )
    ap.add_argument(
        "--val", type=int, default=None, help="override per-stratum val size"
    )
    ap.add_argument(
        "--test", type=int, default=None, help="override per-stratum test size"
    )
    # K_max / r_cap come from strata_v2.BUDGETS per stratum; these override ALL strata (dev).
    ap.add_argument(
        "--k-max", type=int, default=None, help="override per-stratum K_max"
    )
    ap.add_argument(
        "--refinement-timeout",
        type=float,
        default=None,
        help="override per-stratum r_cap",
    )
    ap.add_argument("--samples-per-step", type=int, default=18)
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="concurrency; default is per-stratum RAM-sized "
        "(min(0.85*CPU, 0.85*freeRAM/per_worker_gb[stratum]), floor-guarded)",
    )
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument(
        "--draw-cap-factor",
        type=float,
        default=6.0,
        help="hard cap on draws per cell = ceil(factor*target); trips SHORTFALL if hit",
    )
    # Memory backstop: the overnight OOM took down gnome-shell + the terminal. If free RAM
    # drops below the floor, pause new submissions (inflight drains, freeing memory); below
    # critical, stop entirely. Kept episodes persist on disk, so a relaunch resumes.
    ap.add_argument("--mem-floor-gb", type=float, default=6.0)
    ap.add_argument("--mem-critical-gb", type=float, default=3.0)
    a = ap.parse_args()

    # RAM-size workers from THIS invocation's strata (one, in the deployed sequential mode) and
    # the live free RAM. Computed here, not as an argparse default, so a fresh process measures
    # fresh free RAM after the previous stratum's job fully released its memory.
    if a.workers is None:
        avail_gb = psutil.virtual_memory().available / 1e9
        a.workers = _sized_workers(
            a.strata, os.cpu_count() or 4, avail_gb, a.mem_floor_gb
        )
        pwg = max(S.per_worker_gb(s) for s in a.strata)
        print(
            f"[collect] auto-sized workers={a.workers} "
            f"(freeRAM={avail_gb:.1f}GB, per_worker={pwg:.1f}GB, "
            f"0.85*CPU={int(0.85*(os.cpu_count() or 4))})",
            flush=True,
        )

    # Per-split size override sentinels: None => use the per-stratum strata_v2.SIZES target.
    _size_override = {"train": a.train, "val": a.val, "test": a.test}

    def _target(split: str, stratum: int) -> int:
        ov = _size_override[split]
        return ov if ov is not None else S.sizes(stratum)[split]

    def _budget(stratum: int) -> tuple[int, float]:
        k, r = S.budget(stratum)
        return (
            a.k_max if a.k_max is not None else k,
            a.refinement_timeout if a.refinement_timeout is not None else r,
        )

    # Per-cell state for the dynamic top-up loop.
    cells: dict = {}
    for split in a.splits:
        for stratum in a.strata:
            target = _target(split, stratum)
            cells[(split, stratum)] = {
                "target": target,
                "cap": max(target + 3, math.ceil(a.draw_cap_factor * target)),
                "kept": 0,
                "drawn": 0,
                "inflight": 0,
                "next_idx": 0,
                "errors": 0,
                "results": [],
            }

    _sizes_txt = ", ".join(
        f"{s}:{_target('train', s)}/{_target('val', s)}/{_target('test', s)}"
        for s in a.strata
    )
    print(
        f"[collect] restock3d_v2 strata={a.strata} splits={a.splits} workers={a.workers} "
        f"sizes(tr/val/test per s)={{{_sizes_txt}}} "
        f"budgets={{{', '.join(f'{s}:{_budget(s)}' for s in a.strata)}}}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    t_start = time.perf_counter()
    last_hb = t_start
    paused = False  # watchdog: True while free RAM is below the floor
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        inflight: dict = {}  # future -> cell_key

        def submit_one(cell_key: tuple) -> bool:
            """Submit ONE task for this cell if it still wants more; else False."""
            split, stratum = cell_key
            c = cells[cell_key]
            if not (
                (c["kept"] + c["inflight"] < c["target"]) and (c["drawn"] < c["cap"])
            ):
                return False
            kmax, rcap = _budget(stratum)
            idx = c["next_idx"]
            c["next_idx"] += 1
            c["drawn"] += 1
            c["inflight"] += 1
            fut = ex.submit(
                _task,
                (stratum, split, idx, kmax, rcap, a.samples_per_step, a.data_root),
            )
            inflight[fut] = cell_key
            return True

        def submit_more(cell_key: tuple) -> None:
            """Top up one cell until its inflight target (or draw cap) is met."""
            while submit_one(cell_key):
                pass

        # RESUME PRE-SCAN: seed each cell from episodes already on disk so a completed cell
        # submits ZERO tasks and an interrupted heavy stratum does not re-pay its rejected
        # refinements. Existing indices become kept-cached and the cell continues from
        # max(index)+1. (The per-task path.exists check in _task is the base guarantee; this
        # just skips the wasted re-draws of rejected gaps below the last kept index.)
        raw_dir = Path(a.data_root) / "raw" / S.ENV_VARIANT
        for cell_key, c in cells.items():
            split, stratum = cell_key
            ep_dir = raw_dir / split / "episodes"
            if not ep_dir.exists():
                continue
            found: list[int] = []
            for f in ep_dir.glob("ep_*.pkl.gz"):
                try:
                    pid = int(f.name.removeprefix("ep_").split(".")[0])
                except ValueError:
                    continue
                sp, st, ix = S.decode(pid)
                if sp == split and st == stratum:
                    found.append(ix)
            if not found:
                continue
            found = sorted(set(found))
            for ix in found:
                c["results"].append(
                    {
                        "split": split,
                        "stratum": stratum,
                        "index": ix,
                        "kept": True,
                        "cached": True,
                    }
                )
            c["kept"] = len(found)
            c["drawn"] = len(found)
            c["next_idx"] = max(found) + 1
            print(
                f"[collect] resume {split} s{stratum}: {len(found)} on disk "
                f"(next_idx={c['next_idx']}, target={c['target']})",
                flush=True,
            )

        # ROUND-ROBIN initial fill: one task per cell, cycling, so the running set is a MIX of
        # strata rather than all-heavy (the OOM schedule) or all-light. The executor is FIFO,
        # so submission order == run order; interleaving keeps memory smooth in the mixed phase
        # and still starts a crowded-stratum problem in the first workers.
        progressing = True
        while progressing:
            progressing = False
            for cell_key in cells:
                if submit_one(cell_key):
                    progressing = True

        while inflight:
            done_set, _ = wait(
                list(inflight), timeout=_HEARTBEAT_S, return_when=FIRST_COMPLETED
            )
            for fut in done_set:
                cell_key = inflight.pop(fut)
                c = cells[cell_key]
                c["inflight"] -= 1
                r = fut.result()
                c["results"].append(r)
                if r.get("error"):
                    c["errors"] += 1
                elif r.get("kept"):
                    c["kept"] += 1
                # Only top up while RAM is healthy; when paused, inflight drains (freeing
                # memory) until it recovers. Kept episodes are already on disk.
                if not paused:
                    submit_more(cell_key)
            # Memory watchdog: never let this crash the desktop again.
            avail_gb = psutil.virtual_memory().available / 1e9
            if avail_gb < a.mem_critical_gb:
                if inflight:
                    print(
                        f"[collect] !! CRITICAL: free RAM {avail_gb:.1f}GB < "
                        f"{a.mem_critical_gb}GB -- halting submissions, draining "
                        f"{len(inflight)} inflight. Kept episodes persist; relaunch resumes.",
                        flush=True,
                    )
                paused = True
            elif avail_gb < a.mem_floor_gb:
                if not paused:
                    print(
                        f"[collect] ! low RAM {avail_gb:.1f}GB < {a.mem_floor_gb}GB -- "
                        f"pausing new submissions until it recovers.",
                        flush=True,
                    )
                paused = True
            elif paused and avail_gb > a.mem_floor_gb + 3.0:
                print(
                    f"[collect] RAM recovered ({avail_gb:.1f}GB) -- resuming submissions.",
                    flush=True,
                )
                paused = False
                for cell_key in cells:  # refill after a pause
                    submit_more(cell_key)
            now = time.perf_counter()
            if now - last_hb >= _HEARTBEAT_S or not inflight:
                _heartbeat(cells, t_start, a.workers)
                last_hb = now

    # Census: trim each cell to the first `target` KEPT in index order (delete surplus so the
    # dataset is exactly target/config); report kept/drawn/errors + worst feasible-solve wall.
    from alphatamp.approaches.spectre.collect import episode_path

    print("\n=== census ===", flush=True)
    n_final = 0
    for cell_key in cells:
        split, stratum = cell_key
        c = cells[cell_key]
        target = c["target"]
        kept_rs = sorted(
            (r for r in c["results"] if r.get("kept")),
            key=lambda r: r.get("index", 0),
        )
        surplus = kept_rs[target:]
        for r in surplus:
            p = episode_path(
                Path(a.data_root),
                S.ENV_VARIANT,
                split,
                S.problem_id(split, stratum, r["index"]),
            )
            if p.exists():
                p.unlink()
        kept_final = min(len(kept_rs), target)
        n_final += kept_final
        nt, ns = S.CONFIGS[stratum]
        _kmax, rcap = _budget(stratum)
        fmaxes = [r["feas_max"] for r in c["results"] if r.get("feas_max") is not None]
        worst = f"{max(fmaxes):.0f}s/rcap{rcap:.0f}" if fmaxes else "-"
        flag = "" if kept_final >= target else "  <-- SHORTFALL"
        print(
            f"  {split:5s} s{stratum} ({nt}x{ns}): kept={kept_final}/{target} "
            f"drawn={c['drawn']} err={c['errors']} trimmed={len(surplus)} "
            f"worst_feas={worst}{flag}",
            flush=True,
        )
    print(
        f"\n[collect] total kept={n_final} in {(time.perf_counter()-t_start)/60:.1f}m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
