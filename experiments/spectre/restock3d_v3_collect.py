"""SPECTRE collector for **restock3d_v3** -- the SYNTHETIC dataset.

Collects the four block-count strata (n=6/7/8/9) at **100/25/25** train/val/test
(``strata_v3.SIZES``; 400 train / 100 val / 100 test) into
``data/spectre/raw/restock3d_v3/{train,val,test}`` -- full ``EpisodeRecord``s (pool +
per-candidate labels + 3D scene geometry + F2/F3/reach-over failures), the SAME schema
DD2D/SB2D/restock3d_v2 train on. The **only** difference is how each candidate is labelled:

* pools come from the geometry-informed nearest-first generator (``plan_generator="closed_form"``),
  the deployed prior -- NOT hff;
* labels come from the **analytic refiner** (``feasibility_v3.classify_skeleton``, pure geometry,
  no motion planning), set via ``CollectionConfig.refiner_mode="analytic"``. Wall-clock is
  SYNTHESIZED per candidate: a fail costs the full ``r_cap``; a success costs ``U[0.6,0.8]*r_cap``
  (seeded, deterministic). See ``collect._restock3d_analytic_outcome``.

A problem is kept iff >=1 candidate is analytically feasible (``num_success >= 1``); some n=8/9
problems have their feasible skeleton beyond ``K_max`` (geom solve% ~83/85%), so a **dynamic
top-up loop** keeps ``target`` tasks in flight per (split, stratum) cell until ``target`` are kept
(or a hard draw cap trips a SHORTFALL). The census trims each cell to the first ``target`` kept
*in index order*, so the dataset is exact + reproducible. A pre-scan resumes from on-disk episodes.

Because there is no motion planning, this is **fast and light** (the only cost is the geometry
pool draw; the n=9 K_max=200 A* enumeration is the heaviest). Workers default to a fixed
conservative count; a memory watchdog is kept as cheap insurance. Per-stratum ``(K_max, r_cap)``
come from ``strata_v3.BUDGETS``.

    python experiments/spectre/restock3d_v3_collect.py                 # all 4 strata, 400/100/100
    python experiments/spectre/restock3d_v3_collect.py --strata 0 --train 2 --val 1 --test 1 --workers 2  # smoke
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
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S

_HEARTBEAT_S = 30.0


def _config(
    stratum: int, split: str, k_max: int, timeout_s: float, samples: int
) -> CollectionConfig:
    return CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant=S.ENV_VARIANT,
        model_name="restock3d_v3",
        # Recipe key = the stratum itself (create_restock3d_v3_models reads only `stratum`).
        model_kwargs={"stratum": stratum},
        split=split,
        num_problems=1,
        problem_seed_start=S.problem_id(split, stratum, 0),
        problem_seed_end=S.problem_id(split, stratum, 0) + 1,
        K_max=k_max,
        plan_generator="closed_form",  # -> v3 geometry-guided generator (collect.py dispatch)
        # Analytic labels: no motion planning; feasibility_v3.classify_skeleton + synthetic wall.
        refiner_mode="analytic",
        # Generous, so a full K_max pool of goal-reaching plans can be enumerated for n=9.
        abstract_plan_timeout_s=120.0,
        refinement_timeout_s=timeout_s,  # = r_cap; the synthetic fail-cost / success-cost base.
        num_sampling_attempts_per_step=samples,
        max_trajectory_steps=500,
    )


def _task(args) -> dict:
    """Worker: collect one problem analytically, keep iff feasible, write if kept."""
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
    return {
        "stratum": stratum,
        "split": split,
        "index": index,
        "kept": kept,
        "cached": False,
        "n_skel": len(ep.skeleton_pool),
        "n_succ": ep.summary.num_success,
        "n_fail": ep.summary.num_fail,
        "first_succ": ep.summary.first_success_idx,
        "wall": round(wall, 2),
    }


def _worker_rss_gb() -> tuple[float, float]:
    """(max, sum) RSS in GB across this process's worker children.

    Read-only.
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
    total_done = sum(len(c["results"]) for c in cells.values())

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
        return 5.0  # analytic prior: a few seconds/problem before any completion

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
        n = S.params(stratum).n
        drawn_done = len([r for r in c["results"] if not r.get("cached")])
        yld = f"{100*c['kept']/drawn_done:.0f}%" if drawn_done else "-"
        print(
            f"    {split:5s} s{stratum}(n={n}): kept={c['kept']}/{c['target']} "
            f"drawn={c['drawn']} inflight={c['inflight']} yield={yld} err={c['errors']}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", type=int, nargs="+", default=list(S.STRATA))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    # Sizes default to strata_v3.SIZES (100/25/25 per stratum); a non-None value overrides.
    ap.add_argument(
        "--train", type=int, default=None, help="override per-stratum train size"
    )
    ap.add_argument(
        "--val", type=int, default=None, help="override per-stratum val size"
    )
    ap.add_argument(
        "--test", type=int, default=None, help="override per-stratum test size"
    )
    # K_max / r_cap come from strata_v3.BUDGETS per stratum; these override ALL strata (dev).
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
    # Analytic collection is light (no motion planning); the n=9 K_max=200 A* enumeration is
    # the only heavy step. A fixed conservative default keeps the peak well under the box RAM;
    # the watchdog is the backstop. Raise if the heartbeat shows spare freeRAM.
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(int(0.6 * (os.cpu_count() or 4)), 12)),
        help="concurrency (default min(0.6*CPU, 12))",
    )
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument(
        "--draw-cap-factor",
        type=float,
        default=6.0,
        help="hard cap on draws per cell = ceil(factor*target); trips SHORTFALL if hit",
    )
    ap.add_argument("--mem-floor-gb", type=float, default=6.0)
    ap.add_argument("--mem-critical-gb", type=float, default=3.0)
    a = ap.parse_args()

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
        f"[collect] restock3d_v3 (ANALYTIC) strata={a.strata} splits={a.splits} "
        f"workers={a.workers} sizes(tr/val/test per s)={{{_sizes_txt}}} "
        f"budgets={{{', '.join(f'{s}:{_budget(s)}' for s in a.strata)}}}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    t_start = time.perf_counter()
    last_hb = t_start
    paused = False
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        inflight: dict = {}

        def submit_one(cell_key: tuple) -> bool:
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
            while submit_one(cell_key):
                pass

        # RESUME PRE-SCAN: seed each cell from episodes already on disk.
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

        def _wants_more() -> bool:
            return any(
                c["kept"] < c["target"] and c["drawn"] < c["cap"]
                for c in cells.values()
            )

        # OUTER loop: refill + drain, repeating after a memory pause. When the watchdog pauses
        # under CRITICAL RAM, the inflight set drains to empty (freeing all worker RAM); the
        # inner loop then exits, and this outer loop clears the pause and refills any cell still
        # under target. Without it, a paused-drain silently shortfalls (bug seen on n=9 K=200).
        while True:
            # ROUND-ROBIN (re)fill.
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
                    if not paused:
                        submit_more(cell_key)
                avail_gb = psutil.virtual_memory().available / 1e9
                if avail_gb < a.mem_critical_gb:
                    if inflight and not paused:
                        print(
                            f"[collect] !! CRITICAL: free RAM {avail_gb:.1f}GB < "
                            f"{a.mem_critical_gb}GB -- halting submissions, draining "
                            f"{len(inflight)} inflight (outer loop resumes after RAM frees).",
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
                    for cell_key in cells:
                        submit_more(cell_key)
                now = time.perf_counter()
                if now - last_hb >= _HEARTBEAT_S or not inflight:
                    _heartbeat(cells, t_start, a.workers)
                    last_hb = now

            # Inflight drained. If it drained because of a memory pause, worker RAM is now
            # freed -- clear the pause, settle briefly, and let the outer loop refill.
            if not _wants_more():
                break
            if paused:
                print(
                    "[collect] drained under pause -- RAM freed, resuming top-up.",
                    flush=True,
                )
            paused = False
            time.sleep(2.0)

    # Census: trim each cell to the first `target` KEPT in index order.
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
        n = S.params(stratum).n
        flag = "" if kept_final >= target else "  <-- SHORTFALL"
        print(
            f"  {split:5s} s{stratum} (n={n}): kept={kept_final}/{target} "
            f"drawn={c['drawn']} err={c['errors']} trimmed={len(surplus)}{flag}",
            flush=True,
        )
    print(
        f"\n[collect] total kept={n_final} in {(time.perf_counter()-t_start)/60:.1f}m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
