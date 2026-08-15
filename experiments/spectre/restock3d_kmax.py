"""Restock3D K_max estimation: first-feasible index under plain hff vs eager order (no
refinement).

For each (stratum, problem) it enumerates the skeleton pool under BOTH orders and records the index of
the first ``is_feasible_skeleton`` member (the no-refinement K_max method). The **plain hff** index is
the meaningful, FP-rich K_max lower bound (a feasibility-agnostic pool that still contains a feasible);
the **eager** index (expected ~0) validates the heuristic and measures the collection short-circuit
depth. Also reports F3 presence per order (V3). Worker-parallel across problems (spawn). Writes
``data/spectre/derived/restock3d_v1/kmax_estimate.json``.

    python experiments/spectre/restock3d_kmax.py --strata 0,1,2,3 --problems 20 --k-max 200 --workers 24
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from restock3d_harness import (
    build_cfg,
    enumerate_pool,
    first_feasible_index,
    n_f3_candidates,
    run_parallel,
)

from alphatamp.approaches.spectre.envs.restock3d import strata as S

_OUT = Path("data/spectre/derived/restock3d_v1/kmax_estimate.json")
_EAGER_K = (
    50  # eager index is ~0; a small cap suffices to confirm it (saves enumeration time)
)


def _kmax_worker(job: object) -> dict:
    """Top-level worker: enumerate plain (K) and eager (small K) pools for one
    problem."""
    stratum, problem_id, k_plain = job  # type: ignore[misc]
    plain_cfg = build_cfg(
        stratum,
        k_max=k_plain,
        plan_generator="closed_form",
        abstract_plan_timeout_s=90.0,
    )
    eager_cfg = build_cfg(
        stratum,
        k_max=_EAGER_K,
        plan_generator="astar_eager",
        abstract_plan_timeout_s=90.0,
    )
    plain_pool, plain_tables = enumerate_pool(plain_cfg, problem_id)
    eager_pool, eager_tables = enumerate_pool(eager_cfg, problem_id)
    return {
        "stratum": stratum,
        "problem_id": problem_id,
        "plain_ff": first_feasible_index(plain_pool, plain_tables),
        "eager_ff": first_feasible_index(eager_pool, eager_tables),
        "plain_pool": len(plain_pool),
        "eager_pool": len(eager_pool),
        "plain_n_f3": n_f3_candidates(plain_pool, plain_tables),
        "eager_n_f3": n_f3_candidates(eager_pool, eager_tables),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--problems", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=200)
    # Enumeration to K=200 is more memory-heavy than the oracle refine; 24 workers can OOM a
    # worker on the heavy strata (r2/r3) and break the pool. 12 is safe on the 59 GB box.
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    strata = [int(s) for s in args.strata.split(",")]
    jobs: list[object] = []
    for stratum in strata:
        for i in range(args.problems):
            jobs.append((stratum, S.problem_id("train", stratum, i), args.k_max))

    print(
        f"[kmax] {len(jobs)} jobs ({len(strata)} strata x {args.problems}), "
        f"K_plain={args.k_max}, K_eager={_EAGER_K}, {args.workers} workers",
        flush=True,
    )
    results = run_parallel(
        _kmax_worker, jobs, workers=args.workers, heartbeat_s=30.0, label="kmax"
    )
    # Self-heal: a worker OOM on a heavy stratum can break the pool and fail the pending jobs.
    # Resubmit the failed jobs at low concurrency (up to 2 passes).
    for _ in range(2):
        failed = [
            r[1]  # type: ignore[index]
            for r in results
            if isinstance(r, tuple) and r and r[0] == "ERROR"
        ]
        if not failed:
            break
        print(f"[kmax] resubmitting {len(failed)} failed jobs at 4 workers", flush=True)
        results = [
            r for r in results if not (isinstance(r, tuple) and r and r[0] == "ERROR")
        ]
        results += run_parallel(
            _kmax_worker, failed, workers=4, heartbeat_s=30.0, label="kmax-retry"
        )
    rows = [r for r in results if isinstance(r, dict)]
    n_err = len(results) - len(rows)

    per_stratum: dict[str, dict[str, object]] = {}
    for stratum in strata:
        srows = [r for r in rows if r["stratum"] == stratum]
        plain = [r["plain_ff"] for r in srows]
        eager = [r["eager_ff"] for r in srows]
        plain_found = [v for v in plain if v is not None]
        n_censored = sum(1 for v in plain if v is None)  # feasible not in plain top-K
        kmax_r = math.ceil(max(plain_found) * 1.2) if plain_found else None
        rec: dict[str, object] = {
            "n": len(srows),
            "plain_ff_found": sorted(plain_found),
            "plain_ff_max": (max(plain_found) if plain_found else None),
            "plain_ff_median": (
                statistics.median(plain_found) if plain_found else None
            ),
            "plain_censored_beyond_K": n_censored,
            "K_max_r": kmax_r,  # max(plain first-feasible) * 1.2, rounded up
            "eager_ff_max": (
                max(v for v in eager if v is not None)
                if any(v is not None for v in eager)
                else None
            ),
            "eager_ff_all_zero": all(v == 0 for v in eager),
            "plain_has_f3": all(r["plain_n_f3"] > 0 for r in srows) if srows else False,
            "plain_n_f3_median": (
                statistics.median(r["plain_n_f3"] for r in srows) if srows else 0
            ),
            "eager_n_f3_median": (
                statistics.median(r["eager_n_f3"] for r in srows) if srows else 0
            ),
        }
        per_stratum[str(stratum)] = rec
        print(
            f"  r{stratum}: plain_ff max={rec['plain_ff_max']} "
            f"(censored {n_censored}/{len(srows)}), K_max_r={kmax_r}; "
            f"eager_ff all0={rec['eager_ff_all_zero']}; "
            f"plain_F3~{rec['plain_n_f3_median']} eager_F3~{rec['eager_n_f3_median']}",
            flush=True,
        )

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "config": {
                    "problems_per_stratum": args.problems,
                    "K_plain": args.k_max,
                    "K_eager": _EAGER_K,
                    "kmax_rule": "ceil(max(plain first-feasible index) * 1.2)",
                    "note": (
                        "eager order front-loads feasibles (ff~0) and buries F3; "
                        "training-pool membership + reported baseline use the PLAIN order."
                    ),
                },
                "per_stratum": per_stratum,
                "n_worker_errors": n_err,
            },
            indent=2,
        )
    )
    print(f"[kmax] wrote {_OUT} (worker errors: {n_err})", flush=True)


if __name__ == "__main__":
    main()
