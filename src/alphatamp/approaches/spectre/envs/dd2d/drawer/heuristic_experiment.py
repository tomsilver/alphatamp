"""DD2D heuristic experiment -- does the *kind* of search heuristic reorder the diverse
plan set toward the feasible (blocker-clearing) plan?

Five enumeration arms, from uninformed through off-the-shelf symbolic heuristics to a simple
hand-written *geometric* prior, compared on **first-feasible rank** (# skeletons refined until
the first feasible one) over the same set of subset-requiring DD2D problems:

    arm         search  heuristic     geometry?   hand-written?
    ---------   ------  ------------  ----------  -------------
    bfs         bfs     -             no          no (current baseline)
    astar-hff   astar   pyperplan hFF no          no (off-the-shelf)
    gbf-hff     gbf     pyperplan hFF no          no (off-the-shelf)
    astar-dist  astar   distance      yes(coarse) yes (simple)
    gbf-dist    gbf     distance      yes(coarse) yes (simple)

Narrative: if the *symbolic* heuristics (hFF) don't help but the *geometric* prior does, the
useful signal is geometric -- precisely what PIGINet learns; the distance arms are a crude,
hand-written stand-in for it. See docs/dd2d.md "Fair baselines" and notebook.md.

This is the heavy compute layer: it generates problems, enumerates all arms, and refines
(the expensive step) with a **lazy per-problem memo** -- each distinct skeleton is refined at
most once (fixed per-skeleton seed so its feasibility is identical across arms), and only if
some arm reaches it before that arm's first feasible. It writes one CSV row per (problem, arm)
to ``data/dd2d/out_dd2d/heuristic_experiment/results.csv`` (override with ``--output``) plus a sibling
run-meta JSON. The marimo notebook ``heuristic_notebook.py`` reads that CSV and renders the
charts (it does not compute).

    python -m blocks_tamp.dd2d.heuristic_experiment --smoke                       # tiny serial check
    python -m blocks_tamp.dd2d.heuristic_experiment \\
        --num-items 13 --lambda 0.8 --margin 1 --k 200 --num-problems 50 --crowd 5 \\
        --diverse-crowd --min-subset 3 --retry-cap 10 --samples-per-step 15 --time-budget 10 --workers 8
    # custom output path (relative to the envsearch dir); writes results_minsubset2_meta.json alongside:
    python -m blocks_tamp.dd2d.heuristic_experiment --min-subset 2 \\
        --output data/dd2d/out_dd2d/heuristic_experiment/results_minsubset2.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass

from .eda import _progress, wilson_ci

OUT_DIR = os.path.join("data", "dd2d", "out_dd2d", "heuristic_experiment")
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")
RUN_META = os.path.join(OUT_DIR, "run_meta.json")

# (arm label, search, heuristic) -- heuristic None => blind bfs (the established baseline).
ARMS: tuple[tuple[str, str, str | None], ...] = (
    ("bfs", "bfs", None),
    ("astar-hff", "astar", "hff"),
    ("gbf-hff", "gbf", "hff"),
    ("astar-dist", "astar", "dist"),
    ("gbf-dist", "gbf", "dist"),
)

FIELDS = [
    "problem_id",
    "seed",
    "min_feasible_subset",
    "arm",
    "search",
    "heuristic",
    "num_skeletons",
    "first_feasible_rank",
    "solved",
    "cum_calls",
    "expansions",
    "starved",
    "n_refined_problem",
    "elapsed_s",
]


@dataclass(frozen=True)
class Config:
    # generation / problem distribution (defaults = the requested config)
    num_problems: int = 50
    lam: float = 0.8
    crowd: int = 5
    diverse_crowd: bool = True
    margin: float = 1.0
    n_items: int | None = 13
    min_subset: int = 3
    # planner
    k: int = 200
    max_expansions: int = 200_000  # best-first frontier cap (gbf-starvation backstop)
    # refiner budget (spec P13/P14/P15)
    budget: int = 500
    retry_cap: int = 10
    samples_per_step: int = 15
    time_budget: float = 10.0
    # coordinator
    max_scan_seeds: int = (
        6000  # cap on seeds tried while collecting num_problems successes
    )

    def gen_kwargs(self) -> dict:
        return dict(
            lam=self.lam,
            crowd=self.crowd,
            diverse_crowd=self.diverse_crowd,
            margin=self.margin,
            n_items=self.n_items,
            require_subset=True,
            min_subset=self.min_subset,
            certify=True,
            budget=self.budget,
            retry_cap=self.retry_cap,
            samples_per_step=self.samples_per_step,
            time_budget=self.time_budget,
        )

    def ref_kwargs(self) -> dict:
        return dict(
            budget=self.budget,
            retry_cap=self.retry_cap,
            samples_per_step=self.samples_per_step,
            time_budget=self.time_budget,
        )


def _stable_seed(key) -> int:
    """Deterministic (cross-process) refiner seed for a skeleton, so its feasibility
    label is identical no matter which arm/rank first reaches it."""
    return int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# worker (top-level so ProcessPoolExecutor can pickle it)
# --------------------------------------------------------------------------- #
def _run_problem(task) -> dict:
    """Generate one subset-requiring instance for ``seed``, enumerate every arm, and
    compute each arm's first-feasible rank via a lazy per-problem refinement memo.

    Returns
    ``{"seed", "ok", "reason", "rows"}``.
    """
    seed, gkw, ref_kw, k, max_expansions = task
    from alphatamp.approaches.spectre.envs.dd2d.drawer.planning import make_dd2d_planner
    from alphatamp.approaches.spectre.envs.dd2d.drawer.problem import (
        generate_dd2d_problem,
    )
    from alphatamp.approaches.spectre.envs.dd2d.drawer.refine import DD2DRefiner

    t0 = time.perf_counter()
    try:
        problem = generate_dd2d_problem(seed=seed, **gkw)
    except Exception as e:  # this seed's scene band never yielded a min_subset instance
        return {
            "seed": seed,
            "ok": False,
            "reason": f"gen:{type(e).__name__}",
            "rows": [],
        }

    scene = problem.scene
    refiner = DD2DRefiner(**ref_kw)
    cache: dict = (
        {}
    )  # skeleton.key() -> RefineResult (lazy memo, one refine per distinct plan)

    def label(sk):
        key = sk.key()
        res = cache.get(key)
        if res is None:
            res = refiner.refine(sk, scene, seed=_stable_seed(key))
            cache[key] = res
        return res

    # enumerate all arms up front (cheap relative to refinement)
    arm_lists = {}
    for arm, search, heur in ARMS:
        planner = make_dd2d_planner(
            prefer="pyperplan",
            search=search,
            heuristic=heur,
            max_expansions=max_expansions,
        )
        sks = planner.plan(problem, k)
        arm_lists[arm] = (
            sks,
            getattr(planner, "last_expansions", 0),
            getattr(planner, "last_starved", False),
        )

    rows = []
    for arm, search, heur in ARMS:
        sks, exp, starved = arm_lists[arm]
        rank = None
        cum = 0
        for i, sk in enumerate(sks):
            res = label(sk)
            cum += res.n_attempts
            if res.feasible:
                rank = i + 1  # 1-indexed first-feasible rank
                break
        rows.append(
            {
                "problem_id": problem.problem_id,
                "seed": seed,
                "min_feasible_subset": problem.min_feasible_subset,
                "arm": arm,
                "search": search,
                "heuristic": heur or "none",
                "num_skeletons": len(sks),
                "first_feasible_rank": rank,
                "solved": rank is not None,
                "cum_calls": cum,
                "expansions": exp,
                "starved": starved,
                "n_refined_problem": 0,  # filled below
                "elapsed_s": 0.0,
            }
        )
    n_refined = len(cache)
    elapsed = round(time.perf_counter() - t0, 2)
    for r in rows:
        r["n_refined_problem"] = n_refined
        r["elapsed_s"] = elapsed
    return {"seed": seed, "ok": True, "reason": "ok", "rows": rows}


# --------------------------------------------------------------------------- #
# coordinator
# --------------------------------------------------------------------------- #
def _meta_path_for(out_csv: str) -> str:
    """Sibling run-meta path for a results CSV.

    The default CSV keeps the historical
    ``run_meta.json`` name; any custom ``--output`` gets ``<stem>_meta.json`` next to it so a
    custom run never clobbers the default meta.
    """
    if os.path.abspath(out_csv) == os.path.abspath(RESULTS_CSV):
        return RUN_META
    return os.path.splitext(out_csv)[0] + "_meta.json"


def run(cfg: Config, workers: int, out_csv: str = RESULTS_CSV) -> list[dict]:
    """Collect ``cfg.num_problems`` successful problems (consuming seeds as needed) and
    stream per-(problem, arm) rows to ``out_csv``.

    Resumable-friendly: overwrites fresh each run.
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    gkw, ref_kw = cfg.gen_kwargs(), cfg.ref_kwargs()
    all_rows: list[dict] = []
    n_ok = 0
    n_gen_fail = 0
    t0 = time.time()
    print(
        f"# DD2D heuristic experiment: collecting {cfg.num_problems} problems "
        f"(min_subset>={cfg.min_subset}, crowd={cfg.crowd}"
        f"{' diverse' if cfg.diverse_crowd else ''}, lam={cfg.lam}, n_items={cfg.n_items}), "
        f"{len(ARMS)} arms, k={cfg.k}, budget={cfg.budget} calls / {cfg.time_budget}s per plan, "
        f"{workers} workers",
        flush=True,
    )
    wave = max(workers * 2, 8)
    seed = 0
    inflight: set = set()
    # Manage the pool explicitly (not via `with`): once we have enough problems we must NOT
    # let __exit__ call shutdown(wait=True), which would block until every overshoot task
    # (expensive generation-scans / refinements still running for seeds we no longer need)
    # finishes -- that is the "hang after collect" symptom. See `_shutdown_now`.
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            while n_ok < cfg.num_problems and (inflight or seed < cfg.max_scan_seeds):
                # keep ~`wave` tasks in flight (small overshoot beyond num_problems is fine)
                while len(inflight) < wave and seed < cfg.max_scan_seeds:
                    inflight.add(
                        ex.submit(
                            _run_problem, (seed, gkw, ref_kw, cfg.k, cfg.max_expansions)
                        )
                    )
                    seed += 1
                if not inflight:
                    break
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    res = fut.result()
                    if not res["ok"]:
                        n_gen_fail += 1
                        continue
                    if n_ok >= cfg.num_problems:
                        continue  # overshoot from in-flight tasks; ignore
                    for row in res["rows"]:
                        writer.writerow(row)
                    f.flush()
                    all_rows.extend(res["rows"])
                    n_ok += 1
                    _progress("collect", n_ok, cfg.num_problems, t0)
    finally:
        _shutdown_now(ex, inflight)  # never block on the unneeded in-flight overshoot
    print(
        f"\n# collected {n_ok}/{cfg.num_problems} problems "
        f"({n_gen_fail} seeds failed generation, {seed} seeds tried) "
        f"in {(time.time()-t0)/60:.1f} min -> {out_csv}",
        flush=True,
    )
    _write_meta(cfg, n_ok, n_gen_fail, seed, _meta_path_for(out_csv))
    return all_rows


def _shutdown_now(ex: ProcessPoolExecutor, inflight: set) -> None:
    """Stop the pool immediately without waiting on in-flight work.

    Cancels queued futures, returns from shutdown without joining running tasks, and
    hard-terminates any worker still busy with an overshoot task -- so a finished
    collection never hangs on stragglers (nor at interpreter exit, where
    ProcessPoolExecutor's atexit hook would otherwise join them).
    """
    for fut in inflight:
        fut.cancel()
    # snapshot workers BEFORE shutdown -- shutdown() clears ex._processes to None
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()  # workers only compute (no output writes) -> safe to kill


def _write_meta(
    cfg: Config, n_ok: int, n_gen_fail: int, seeds_tried: int, meta_path: str = RUN_META
) -> None:
    meta = {
        "config": asdict(cfg),
        "arms": [{"arm": a, "search": s, "heuristic": h or "none"} for a, s, h in ARMS],
        "n_problems_collected": n_ok,
        "n_gen_failures": n_gen_fail,
        "seeds_tried": seeds_tried,
        "metric": "first_feasible_rank = # skeletons refined until first feasible; "
        "cum_calls = refiner sample-calls up to and incl. first feasible",
        "caveat": (
            "EDA/diagnostic; DD2D negative certificate is the Day-1 fallback "
            "(docs/dd2d.md, notebook.md) -- not a label-dependent research claim."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"# Wrote {meta_path}")


# --------------------------------------------------------------------------- #
# summary (console; the notebook owns the charts)
# --------------------------------------------------------------------------- #
def summarize(rows: list[dict]) -> None:
    import statistics

    print("\n# FIRST-FEASIBLE RANK by arm (success = solved within k; Wilson 95% CI)")
    print(
        f"  {'arm':<12} {'n':>4} {'solve%':>8} {'95% CI':>16} {'mean rank':>10} "
        f"{'median':>7} {'starved':>8}"
    )
    for arm, _s, _h in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n = len(arm_rows)
        solved = [r for r in arm_rows if r["solved"]]
        p, lo, hi = wilson_ci(len(solved), n)
        ranks = sorted(int(r["first_feasible_rank"]) for r in solved)
        mean_r = f"{statistics.mean(ranks):.1f}" if ranks else "-"
        med_r = f"{statistics.median(ranks):.0f}" if ranks else "-"
        starved = sum(1 for r in arm_rows if r["starved"])
        print(
            f"  {arm:<12} {n:>4} {p:>7.1%} [{lo:>5.1%},{hi:>6.1%}] {mean_r:>10} "
            f"{med_r:>7} {starved:>8}"
        )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--num-problems", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--crowd", type=int, default=5)
    ap.add_argument(
        "--diverse-crowd", dest="diverse_crowd", action="store_true", default=True
    )
    ap.add_argument("--no-diverse-crowd", dest="diverse_crowd", action="store_false")
    ap.add_argument("--num-items", type=int, default=13)
    ap.add_argument("--min-subset", type=int, default=3)
    ap.add_argument("--max-stream-calls", dest="budget", type=int, default=500)
    ap.add_argument("--retry-cap", type=int, default=10)
    ap.add_argument("--samples-per-step", type=int, default=15)
    ap.add_argument("--time-budget", type=float, default=10.0)
    ap.add_argument("--max-expansions", type=int, default=200_000)
    ap.add_argument("--max-scan-seeds", type=int, default=6000)
    ap.add_argument(
        "--output",
        "-o",
        default=RESULTS_CSV,
        help="results CSV path, relative to the envsearch dir "
        f"(default: {RESULTS_CSV}). A sibling <stem>_meta.json is written "
        "alongside; --analyze-only re-summarizes this same path",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny serial plumbing check")
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="re-summarize the existing results CSV at --output",
    )
    args = ap.parse_args(argv)

    if args.analyze_only:
        rows = _load_rows(args.output)
        summarize(rows)
        return 0

    if args.smoke:
        cfg = Config(
            num_problems=3,
            k=40,
            n_items=9,
            min_subset=2,
            max_expansions=20_000,
            budget=200,
            time_budget=3.0,
            max_scan_seeds=200,
        )
        rows = run(cfg, workers=1, out_csv=args.output)
        summarize(rows)
        return 0

    cfg = Config(
        num_problems=args.num_problems,
        lam=args.lam,
        crowd=args.crowd,
        diverse_crowd=args.diverse_crowd,
        margin=args.margin,
        n_items=args.num_items,
        min_subset=args.min_subset,
        k=args.k,
        max_expansions=args.max_expansions,
        budget=args.budget,
        retry_cap=args.retry_cap,
        samples_per_step=args.samples_per_step,
        time_budget=args.time_budget,
        max_scan_seeds=args.max_scan_seeds,
    )
    rows = run(cfg, args.workers, out_csv=args.output)
    summarize(rows)
    return 0


def _load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["solved"] = str(r["solved"]).lower() in ("true", "1")
            r["first_feasible_rank"] = (
                int(r["first_feasible_rank"]) if r["first_feasible_rank"] else None
            )
            r["starved"] = str(r["starved"]).lower() in ("true", "1")
            rows.append(r)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
