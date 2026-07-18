"""DD2D difficulty EDA -- the geometry-blind pyperplan baseline.

Quantifies how hard DD2D is for the **fair baseline** planner (pyperplan, geometry-blind),
against which PIGINet-style plan ranking is meant to pay off. Two deliverables:

  1. **Attempts-until-success distribution** = the *first-feasible rank* -- the index of the
     first pyperplan-enumerated staging skeleton the refiner actually solves (docs/dd2d.md
     "Fair baselines"; the spec's D1 diagnostic / excess ``E = rank - 1``).
  2. **Success probability with error bars** = fraction of problems solved within the ``k``
     plan budget, reported **per blocker stratum and pooled overall**, each with a binomial
     (Wilson) 95% CI.

Blocker axis. DD2D has no total-blocker knob (total non-target items = ``num_items - 1``,
8-13 at the default 9-14 sampling). The 1-3 "blocker" axis is ``min_feasible_subset`` -- the
number of blockers that MUST be moved to clear the target (a property of each generated
instance, computed from the *sound* positive packing certificate). We stratify by it into
{1,2,3} via rejection sampling at the default crowd/lambda.

Protocol per episode (matching the demo's baseline path):

  * generate one filtered+certified instance for a seed (``generate_dd2d_problem``);
  * enumerate up to ``k`` diverse skeletons with pyperplan (``length_slack=None``, k-driven);
  * refine them in enumerated (ascending-length) order with ``DD2DRefiner`` under
    ``budget`` stream calls AND ``time_budget`` s/plan, **stopping at the first feasible**;
  * record the first-feasible rank (or None = unsolved within k).

These are EDA/diagnostic numbers, NOT label-dependent research claims (DD2D's negative
certificate is still the Day-1 fallback -- see docs/dd2d.md, notebook.md 2026-07-06).

    python -m blocks_tamp.dd2d.eda --calibrate            # measure the min_feasible_subset mix
    python -m blocks_tamp.dd2d.eda --smoke                # tiny plumbing check (serial)
    python -m blocks_tamp.dd2d.eda --episodes 200 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

OUT_DIR = os.path.join("out", "dd2d_eda")
EPISODES_CSV = os.path.join(OUT_DIR, "episodes.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "summary.json")
STRATA = (1, 2, 3)  # min_feasible_subset values we stratify over


@dataclass(frozen=True)
class Config:
    # generation (defaults per the request: num-items/order/retry-cap/samples-per-step/crowd/lambda)
    lam: float = 0.8
    crowd: int = 10
    margin: float = 1.0
    n_items: int | None = None  # None -> sampled 9-14
    # refiner budget (the request): 500 stream calls OR 10 s/plan, whichever first
    budget: int = 500
    retry_cap: int = 10
    samples_per_step: int = 15
    time_budget: float = 10.0
    # planner
    k: int = 200  # diverse plans / attempted refinements per problem
    # stratified sample
    episodes: int = 200  # total planning episodes (~episodes/3 per stratum)
    max_scan_seeds: int = (
        4000  # cap on seeds scanned while filling the rare subset-3 stratum
    )


def _progress(prefix: str, done: int, total: int, t0: float) -> None:
    """A flushed progress bar with ETA.

    Updates in place on a TTY; throttled newlines (~5% steps) when stdout is redirected
    to a file/pipe so a log stays readable.
    """
    frac = done / total if total else 1.0
    filled = int(round(frac * 24))
    bar = "#" * filled + "-" * (24 - filled)
    el = time.time() - t0
    eta = (el / done * (total - done)) if done else 0.0
    line = (
        f"{prefix} [{bar}] {frac*100:3.0f}%  {done}/{total}  |  "
        f"{el/60:4.1f} min elapsed  |  ETA {eta/60:4.1f} min"
    )
    if sys.stdout.isatty():
        print("\r" + line, end=("\n" if done >= total else ""), flush=True)
    elif done >= total or total == 0 or done % max(1, total // 20) == 0:
        print(line, flush=True)


def _scan_progress(
    prefix: str, collected: dict, targets: dict, scanned: int, t0: float
) -> None:
    """Phase-A indicator: total work is unknown (we scan until every stratum fills), so drive
    the bar off the slowest-filling stratum and estimate ETA from its fill rate."""
    fracs = [len(collected[s]) / targets[s] for s in targets if targets[s]]
    frac = min(fracs) if fracs else 1.0
    filled = int(round(frac * 24))
    bar = "#" * filled + "-" * (24 - filled)
    el = time.time() - t0
    eta = (el / frac * (1 - frac)) if frac > 0 else 0.0
    got = {s: len(collected[s]) for s in targets}
    line = (
        f"{prefix} [{bar}] {frac*100:3.0f}%  collected {got} / {dict(targets)}  |  "
        f"{scanned} seeds  |  {el/60:4.1f} min  |  ETA {eta/60:4.1f} min"
    )
    if sys.stdout.isatty():
        print("\r" + line + "   ", end=("\n" if frac >= 1.0 else ""), flush=True)
    else:
        print(line, flush=True)


def gen_kwargs(cfg: Config, **overrides) -> dict:
    kw = dict(
        lam=cfg.lam,
        crowd=cfg.crowd,
        margin=cfg.margin,
        n_items=cfg.n_items,
        budget=cfg.budget,
        retry_cap=cfg.retry_cap,
        samples_per_step=cfg.samples_per_step,
        time_budget=cfg.time_budget,
    )
    kw.update(overrides)
    return kw


# --------------------------------------------------------------------------- #
# workers (top-level so ProcessPoolExecutor can pickle them)
# --------------------------------------------------------------------------- #
def _classify_seed(task) -> tuple[int, int | None, str]:
    """Generate the instance for ``seed`` and return (seed, min_feasible_subset,
    err)."""
    seed, gkw = task
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import (
        generate_dd2d_problem,
    )

    try:
        p = generate_dd2d_problem(seed=seed, **gkw)
        return seed, p.min_feasible_subset, ""
    except Exception as e:  # a seed that never yields a filtered instance; skip it
        return seed, None, f"{type(e).__name__}"


def _run_episode(task) -> dict:
    """Regenerate the instance for ``seed`` and run the pyperplan baseline episode."""
    seed, stratum, gkw, k = task
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.planning import make_dd2d_planner
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import (
        generate_dd2d_problem,
    )
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.refine import DD2DRefiner

    row = {
        "seed": seed,
        "min_feasible_subset": stratum,
        "num_skeletons": 0,
        "first_feasible_rank": None,
        "solved": False,
        "stream_calls": 0,
        "elapsed_s": 0.0,
        "reason": "",
    }
    t0 = time.perf_counter()
    try:
        problem = generate_dd2d_problem(seed=seed, **gkw)
        # stratum is fixed at scan time; regeneration is deterministic, but guard anyway
        row["min_feasible_subset"] = problem.min_feasible_subset
        planner = make_dd2d_planner(
            prefer="pyperplan", order="published", length_slack=None
        )
        skeletons = planner.plan(problem, k)
        row["num_skeletons"] = len(skeletons)
        if not skeletons:
            row["reason"] = "no_plans"
            row["elapsed_s"] = round(time.perf_counter() - t0, 3)
            return row
        refiner = DD2DRefiner(
            budget=gkw["budget"],
            retry_cap=gkw["retry_cap"],
            samples_per_step=gkw["samples_per_step"],
            time_budget=gkw["time_budget"],
        )
        total_calls = 0
        for i, sk in enumerate(skeletons):
            res = refiner.refine(
                sk, problem.scene, seed=1000 + i
            )  # match the demo's seeding
            total_calls += res.n_attempts
            if res.feasible:
                row["first_feasible_rank"] = i + 1  # 1-indexed rank
                row["solved"] = True
                row["reason"] = "solved"
                break
        else:
            row["reason"] = "unsolved_within_k"
        row["stream_calls"] = total_calls
    except Exception as e:  # keep the sweep alive; record the failure
        row["reason"] = f"error:{type(e).__name__}"
    row["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return row


# --------------------------------------------------------------------------- #
# phase A: stratified sampling by min_feasible_subset
# --------------------------------------------------------------------------- #
def _targets(episodes: int) -> dict[int, int]:
    """Split ``episodes`` as evenly as possible across the three strata (e.g.
    67/67/66)."""
    base, rem = divmod(episodes, len(STRATA))
    return {s: base + (1 if i < rem else 0) for i, s in enumerate(STRATA)}


def scan_strata(cfg: Config, workers: int) -> tuple[dict[int, list[int]], dict]:
    """Scan seeds (parallel) until each stratum has enough members; trim to targets.

    Returns (selected_seeds_by_stratum, scan_stats). Falls back to ``require_subset`` for
    subset-3 only if pure rejection can't fill it within ``max_scan_seeds`` (the deviation
    is recorded in scan_stats and printed).
    """
    targets = _targets(cfg.episodes)
    collected: dict[int, list[int]] = {s: [] for s in STRATA}
    seen_mfs: dict[int, int] = (
        {}
    )  # mfs value -> count seen (incl. >=4 / None), for the mix report
    gkw = gen_kwargs(cfg)
    scanned = 0
    wave = max(workers * 4, 16)
    t0 = time.time()
    print(
        f"# Phase A: scanning seeds for strata {dict(targets)} (crowd={cfg.crowd}, lam={cfg.lam})",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as ex:
        seed = 0
        while seed < cfg.max_scan_seeds and not _full(collected, targets):
            batch = list(range(seed, min(seed + wave, cfg.max_scan_seeds)))
            seed += wave
            for fut in as_completed(
                [ex.submit(_classify_seed, (s, gkw)) for s in batch]
            ):
                s, mfs, err = fut.result()
                scanned += 1
                key = mfs if mfs is not None else -1
                seen_mfs[key] = seen_mfs.get(key, 0) + 1
                if mfs in collected and len(collected[mfs]) < targets[mfs]:
                    collected[mfs].append(s)
            _scan_progress("Phase A scan", collected, targets, scanned, t0)
    if not sys.stdout.isatty():
        print(
            f"  scan mix over {scanned} seeds: {_mix_str(seen_mfs, scanned)}",
            flush=True,
        )

    fallback = {}
    if not _full(collected, targets):
        fallback = _fill_fallback(
            collected, targets, cfg, workers, start_seed=cfg.max_scan_seeds
        )

    for s in STRATA:  # deterministic: smallest seeds first
        collected[s] = sorted(collected[s])[: targets[s]]
    stats = {
        "targets": targets,
        "seeds_scanned": scanned,
        "mfs_mix": {
            (str(k) if k != -1 else ">=4/None"): v for k, v in sorted(seen_mfs.items())
        },
        "collected": {s: len(collected[s]) for s in STRATA},
        "subset3_fallback": fallback,
    }
    return collected, stats


def _fill_fallback(collected, targets, cfg: Config, workers, start_seed: int) -> dict:
    """require_subset=True, min_subset=3 to top up the subset-3 bucket (keep only
    mfs==3)."""
    need = targets[3] - len(collected[3])
    if need <= 0:
        return {}
    print(
        f"# Phase A: subset-3 short by {need}; topping up with require_subset "
        f"(min_subset=3) from seed {start_seed} -- DEVIATES from pure default-crowd rejection"
    )
    gkw = gen_kwargs(cfg, require_subset=True, min_subset=3)
    got = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        seed = start_seed
        while (
            len(collected[3]) < targets[3]
            and seed < start_seed + 4 * cfg.max_scan_seeds
        ):
            batch = list(range(seed, seed + max(workers * 2, 8)))
            seed += max(workers * 2, 8)
            for fut in as_completed(
                [ex.submit(_classify_seed, (s, gkw)) for s in batch]
            ):
                s, mfs, err = fut.result()
                if mfs == 3 and len(collected[3]) < targets[3]:
                    collected[3].append(s)
                    got += 1
    return {"added": got, "min_subset": 3, "note": "require_subset top-up for subset-3"}


def _full(collected, targets) -> bool:
    return all(len(collected[s]) >= targets[s] for s in STRATA)


def _mix_str(seen_mfs: dict, scanned: int) -> str:
    if scanned == 0:
        return "-"
    parts = []
    for k in sorted(seen_mfs):
        label = str(k) if k != -1 else ">=4/None"
        parts.append(f"{label}:{100 * seen_mfs[k] / scanned:.0f}%")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# phase B: run episodes
# --------------------------------------------------------------------------- #
FIELDS = [
    "seed",
    "min_feasible_subset",
    "num_skeletons",
    "first_feasible_rank",
    "solved",
    "stream_calls",
    "elapsed_s",
    "reason",
]


def run_episodes(
    selected: dict[int, list[int]],
    cfg: Config,
    workers: int,
    out_csv: str = EPISODES_CSV,
) -> list[dict]:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    gkw = gen_kwargs(cfg)
    tasks = [(s, stratum, gkw, cfg.k) for stratum in STRATA for s in selected[stratum]]
    total = len(tasks)
    rows: list[dict] = []
    done = 0
    t0 = time.time()
    print(
        f"# Phase B: {total} episodes (pyperplan k={cfg.k}, budget={cfg.budget} calls / "
        f"{cfg.time_budget}s per plan), {workers} workers",
        flush=True,
    )
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_run_episode, t) for t in tasks]):
                row = fut.result()
                writer.writerow(row)
                f.flush()
                rows.append(row)
                done += 1
                _progress("Phase B refine", done, total, t0)
    print(f"# Phase B complete in {(time.time()-t0)/60:.1f} min -> {out_csv}")
    return rows


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate + Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _attempts_stats(ranks: list[int]) -> dict:
    if not ranks:
        return {"n_solved": 0}
    xs = sorted(ranks)
    return {
        "n_solved": len(xs),
        "mean": round(statistics.mean(xs), 2),
        "median": statistics.median(xs),
        "p25": _quantile(xs, 0.25),
        "p75": _quantile(xs, 0.75),
        "min": xs[0],
        "max": xs[-1],
    }


def _quantile(sorted_xs: list[int], q: float):
    if not sorted_xs:
        return None
    idx = min(len(sorted_xs) - 1, int(round(q * (len(sorted_xs) - 1))))
    return sorted_xs[idx]


def _summarize_group(rows: list[dict], k: int) -> dict:
    n = len(rows)
    solved_rows = [r for r in rows if r["solved"]]
    n_solved = len(solved_rows)
    p, lo, hi = wilson_ci(n_solved, n)
    ranks = [int(r["first_feasible_rank"]) for r in solved_rows]
    return {
        "n": n,
        "n_solved": n_solved,
        "success_prob": round(p, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "unsolved_frac": round((n - n_solved) / n, 4) if n else 0.0,
        "attempts_until_success": _attempts_stats(ranks),
        "k": k,
    }


def summarize(rows: list[dict], cfg: Config, scan_stats: dict | None = None) -> dict:
    by_stratum = {}
    for s in STRATA:
        grp = [r for r in rows if r.get("min_feasible_subset") == s]
        if grp:
            by_stratum[str(s)] = _summarize_group(grp, cfg.k)
    summary = {
        "config": {
            "lam": cfg.lam,
            "crowd": cfg.crowd,
            "k": cfg.k,
            "budget": cfg.budget,
            "time_budget": cfg.time_budget,
            "retry_cap": cfg.retry_cap,
            "samples_per_step": cfg.samples_per_step,
            "planner": "pyperplan",
            "episodes": len(rows),
        },
        "overall": _summarize_group(rows, cfg.k),
        "by_stratum": by_stratum,
        "scan": scan_stats or {},
        "caveat": (
            "EDA/diagnostic only; DD2D negative certificate is still the Day-1 fallback "
            "(docs/dd2d.md, notebook.md 2026-07-06) -- do not promote to research claims."
        ),
    }
    return summary


def print_summary(summary: dict) -> None:
    def line(name: str, g: dict) -> None:
        a = g["attempts_until_success"]
        att = (
            f"mean {a['mean']} / median {a['median']} (p25 {a['p25']}, p75 {a['p75']}, max {a['max']})"
            if a.get("n_solved")
            else "-"
        )
        print(
            f"  {name:<10} n={g['n']:<4} solve={g['success_prob']:.2%} "
            f"[{g['ci95_low']:.2%}, {g['ci95_high']:.2%}]  "
            f"unsolved={g['unsolved_frac']:.0%}  attempts-to-first-success: {att}"
        )

    print(
        "\n# SUCCESS PROBABILITY (Wilson 95% CI) and ATTEMPTS-UNTIL-SUCCESS "
        f"(first-feasible rank), pyperplan k={summary['config']['k']}"
    )
    for s in ("1", "2", "3"):
        if s in summary["by_stratum"]:
            line(f"subset={s}", summary["by_stratum"][s])
    line("OVERALL", summary["overall"])
    print(f"# {summary['caveat']}")


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def calibrate(cfg: Config, workers: int, n: int = 200) -> None:
    """Just scan ``n`` seeds and print the min_feasible_subset mix (informs feasibility
    of pure rejection for the rare subset-3 stratum)."""
    gkw = gen_kwargs(cfg)
    seen: dict[int, int] = {}
    t0 = time.time()
    print(
        f"# Calibration: classifying {n} seeds at crowd={cfg.crowd}, lam={cfg.lam}, {workers} workers"
    )
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed(
            [ex.submit(_classify_seed, (s, gkw)) for s in range(n)]
        ):
            _, mfs, _ = fut.result()
            key = mfs if mfs is not None else -1
            seen[key] = seen.get(key, 0) + 1
    dt = time.time() - t0
    print(
        f"# min_feasible_subset mix over {n} seeds ({dt:.0f}s, {dt/max(n,1)*1000:.0f} ms/seed):"
    )
    for k in sorted(seen):
        label = str(k) if k != -1 else ">=4 or None"
        print(f"    mfs={label:<12} {seen[k]:>4}  ({100*seen[k]/n:.1f}%)")
    for target_s in (1, 2, 3):
        rate = seen.get(target_s, 0) / n
        need = _targets(cfg.episodes)[target_s]
        est = (
            f"~{need/rate:.0f} seeds"
            if rate > 0
            else "NOT SEEN (need require_subset fallback)"
        )
        print(f"    -> to collect {need} subset-{target_s}: {est}")


def run_eda(cfg: Config, workers: int) -> dict:
    selected, scan_stats = scan_strata(cfg, workers)
    rows = run_episodes(selected, cfg, workers)
    summary = summarize(rows, cfg, scan_stats)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"# Wrote {SUMMARY_JSON}")
    print_summary(summary)
    return summary


def analyze_only(cfg: Config) -> dict:
    """Re-summarize from an existing episodes.csv (no re-running)."""
    rows = []
    with open(EPISODES_CSV) as f:
        for r in csv.DictReader(f):
            r["min_feasible_subset"] = (
                int(r["min_feasible_subset"]) if r["min_feasible_subset"] else None
            )
            r["solved"] = r["solved"] in ("True", "true", "1")
            r["first_feasible_rank"] = (
                int(r["first_feasible_rank"]) if r["first_feasible_rank"] else None
            )
            rows.append(r)
    summary = summarize(rows, cfg)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="total planning episodes (~/3 per stratum)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--k",
        type=int,
        default=200,
        help="diverse plans / attempted refinements per problem",
    )
    ap.add_argument("--lambda", dest="lam", type=float, default=0.8)
    ap.add_argument("--crowd", type=int, default=10)
    ap.add_argument(
        "--max-stream-calls",
        dest="max_stream_calls",
        type=int,
        default=500,
        help="refiner cap on total stream calls per plan",
    )
    ap.add_argument(
        "--time-budget", type=float, default=10.0, help="wall-clock seconds per plan"
    )
    ap.add_argument("--retry-cap", type=int, default=10)
    ap.add_argument("--samples-per-step", type=int, default=15)
    ap.add_argument("--max-scan-seeds", type=int, default=4000)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="print the min_feasible_subset mix, don't run",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny serial plumbing check")
    ap.add_argument(
        "--analyze-only", action="store_true", help="re-summarize from episodes.csv"
    )
    args = ap.parse_args(argv)

    cfg = Config(
        lam=args.lam,
        crowd=args.crowd,
        budget=args.max_stream_calls,
        time_budget=args.time_budget,
        retry_cap=args.retry_cap,
        samples_per_step=args.samples_per_step,
        k=args.k,
        episodes=args.episodes,
        max_scan_seeds=args.max_scan_seeds,
    )
    if args.smoke:
        cfg = Config(
            lam=args.lam,
            crowd=args.crowd,
            budget=args.max_stream_calls,
            time_budget=args.time_budget,
            retry_cap=args.retry_cap,
            samples_per_step=args.samples_per_step,
            k=40,
            episodes=6,
            max_scan_seeds=400,
        )
        run_eda(cfg, workers=1)
        return 0
    if args.calibrate:
        calibrate(cfg, args.workers)
        return 0
    if args.analyze_only:
        analyze_only(cfg)
        return 0
    run_eda(cfg, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
