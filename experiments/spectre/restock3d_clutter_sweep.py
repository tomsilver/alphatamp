"""Restock3D Gate-3 clutter generation sweep.

Varies the number of F1-blocked goals ``k`` per clutter stratum (r1/r3) and measures, over N problems
per (stratum, k):

  * **solvability** -- fraction whose top-K_max plain-hff pool contains a feasible (relocate-first)
    skeleton (``first_feasible_index`` found); a censored problem has no feasible skeleton in K_max,
  * **baseline FP** -- the plain-hff first-feasible index (how many candidates the naive length order
    tries before a feasible one; this is what clutter is meant to inflate),
  * **eager FP** -- the eager first-feasible index (T5 should surface a relocate-first plan early),
  * **n_block** -- how many goals are actually F1-blocked (sanity: == k).

``k=0`` is the no-clutter baseline. The recipe we want makes baseline FP rise measurably over k=0
while keeping solvability ~100%. Reuses the kmax enumeration harness; the clutter count is overridden
in-process per worker (spawn workers re-import, so the override is applied inside the worker).

    python experiments/spectre/restock3d_clutter_sweep.py --strata 1,3 --problems 20 --ks 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# restock3d_harness applies the IKFast BLAS shim + kept-first import ordering.
from restock3d_harness import (  # noqa: E402
    build_cfg,
    enumerate_pool,
    first_feasible_index,
    run_parallel,
)

import alphatamp.approaches.spectre.envs.restock3d.strata as S  # noqa: E402

_OUT = Path("data/spectre/derived/restock3d_v1/clutter_recipe.json")
_K_MAX = 200
_EAGER_K = 50


def _set_clutter(stratum: int, k: int) -> None:
    """Override BOTH clutter-count tables (specs + positions) for this stratum, in-
    process."""
    import alphatamp.approaches.spectre.envs.restock3d.generator as gen
    import alphatamp.approaches.spectre.envs.restock3d.kinematic_env as ke

    gen._CLUTTER_PER_STRATUM[stratum] = k  # pylint: disable=protected-access
    ke.CLUTTER_PER_STRATUM[stratum] = k


def _sweep_worker(job: object) -> dict:
    stratum, problem_id, k = job  # type: ignore[misc]
    _set_clutter(stratum, k)
    plain_cfg = build_cfg(
        stratum,
        k_max=_K_MAX,
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
        "k": k,
        "problem_id": problem_id,
        "plain_ff": first_feasible_index(plain_pool, plain_tables),
        "eager_ff": first_feasible_index(eager_pool, eager_tables),
        "plain_pool": len(plain_pool),
        "n_block": len(plain_tables.blockers),
    }


def _agg(vals: list) -> dict:
    present = [v for v in vals if v is not None]
    return {
        "n": len(vals),
        "solved": len(present),
        "solve_rate": round(len(present) / max(1, len(vals)), 3),
        "mean": round(statistics.mean(present), 2) if present else None,
        "median": round(statistics.median(present), 1) if present else None,
        "max": max(present) if present else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="1,3")
    parser.add_argument("--ks", default="0,1,2,3")
    parser.add_argument("--problems", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    strata = [int(s) for s in args.strata.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    jobs: list[object] = []
    for stratum in strata:
        for k in ks:
            for i in range(args.problems):
                jobs.append((stratum, S.problem_id("train", stratum, i), k))
    print(
        f"[clutter-sweep] {len(jobs)} jobs (strata {strata} x ks {ks} x {args.problems}), "
        f"K_plain={_K_MAX} K_eager={_EAGER_K}, {args.workers} workers",
        flush=True,
    )
    results = run_parallel(
        _sweep_worker,
        jobs,
        workers=args.workers,
        heartbeat_s=30.0,
        label="clutter-sweep",
    )
    for _ in range(2):
        failed = [
            r[1] for r in results if isinstance(r, tuple) and r and r[0] == "ERROR"
        ]
        if not failed:
            break
        print(f"[clutter-sweep] resubmitting {len(failed)} at 4 workers", flush=True)
        results = [
            r for r in results if not (isinstance(r, tuple) and r and r[0] == "ERROR")
        ]
        results += run_parallel(
            _sweep_worker, failed, workers=4, heartbeat_s=30.0, label="clutter-retry"
        )
    ok_results = [r for r in results if isinstance(r, dict)]
    n_err = len(results) - len(ok_results)

    summary: dict = {
        "config": {"K_max": _K_MAX, "eager_k": _EAGER_K, "problems": args.problems},
        "per_cell": {},
    }
    print(f"\n==== GATE-3 CLUTTER SWEEP ({n_err} worker errors) ====", flush=True)
    for stratum in strata:
        for k in ks:
            cell = [r for r in ok_results if r["stratum"] == stratum and r["k"] == k]
            if not cell:
                continue
            plain = _agg([r["plain_ff"] for r in cell])
            eager = _agg([r["eager_ff"] for r in cell])
            n_block = round(statistics.mean([r["n_block"] for r in cell]), 2)
            summary["per_cell"][f"r{stratum}_k{k}"] = {
                "plain_ff": plain,
                "eager_ff": eager,
                "mean_n_block": n_block,
            }
            print(
                f"  r{stratum} k={k}: solve={plain['solve_rate']:.0%} "
                f"plain_ff mean={plain['mean']} median={plain['median']} max={plain['max']} | "
                f"eager_ff mean={eager['mean']} median={eager['median']} | n_block={n_block}",
                flush=True,
            )

    # Recipe pick: solvability is judged on the EAGER order (the plain hff order buries the longer
    # relocate-first plans past K_max, so its pool is censored on cluttered problems -- that IS the
    # intended catastrophic naive-order FP, but a feasible skeleton still exists and the eager order
    # surfaces it). Pick the largest k whose eager pool stays feasible (>= 0.95) and whose plain FP
    # rises over k=0 (measurable difficulty increase; plain censoring counts as an increase).
    recipe: dict = {}
    for stratum in strata:
        base = (
            summary["per_cell"]
            .get(f"r{stratum}_k0", {})
            .get("plain_ff", {})
            .get("mean")
        )
        chosen = 0
        for k in sorted(ks):
            if k == 0:
                continue
            cell = summary["per_cell"].get(f"r{stratum}_k{k}", {})
            eager_solv = cell.get("eager_ff", {}).get("solve_rate", 0)
            plain_solv = cell.get("plain_ff", {}).get("solve_rate", 1)
            plain_mean = cell.get("plain_ff", {}).get("mean")
            harder = plain_solv < 1.0 or (plain_mean or 0) > (
                base or 0
            )  # censoring or higher FP
            if eager_solv >= 0.95 and harder:
                chosen = k
        recipe[f"r{stratum}"] = chosen
    summary["recipe_k"] = recipe
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[clutter-sweep] recipe (blocked goals per stratum): {recipe}", flush=True)
    print(f"[clutter-sweep] wrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
