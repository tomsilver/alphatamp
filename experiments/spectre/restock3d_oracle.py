"""Restock3D oracle harness: certify feasibility + calibrate the per-candidate refinement cap.

For each (stratum, problem) the privileged oracle constructs a feasible skeleton and refines it
through the standard refiner, recording the successful call's wall-clock. Per stratum:
``cap_r = max(feasible t_oracle) * 1.2`` (also reports p95*1.5). Worker-parallel across problems
(spawn; each worker builds its own env). Writes
``data/spectre/derived/restock3d_v1/oracle_calibration.json``.

Run from the repo root (venv active)::

    python experiments/spectre/restock3d_oracle.py --strata 0,1,2,3 --problems 8 --budget-s 300 --workers 24
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# restock3d_harness runs the IKFast BLAS shim at import; keep it first.
from restock3d_harness import build_cfg, run_parallel

from alphatamp.approaches.spectre.envs.restock3d import strata as S
from alphatamp.approaches.spectre.envs.restock3d.oracle import (
    OracleResult,
    refine_oracle,
)

_OUT = Path("data/spectre/derived/restock3d_v1/oracle_calibration.json")


def _oracle_worker(job: object) -> OracleResult:
    """Top-level (picklable) worker: build the cfg and run the oracle for one
    problem."""
    stratum, problem_id, budget_s, max_retries = job  # type: ignore[misc]
    cfg = build_cfg(
        stratum,
        k_max=1,  # unused by the oracle (it builds its own skeleton)
        num_sampling_attempts_per_step=10,
        refinement_timeout_s=budget_s,
    )
    return refine_oracle(cfg, problem_id, budget_s=budget_s, max_retries=max_retries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--problems", type=int, default=8)
    parser.add_argument("--budget-s", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    strata = [int(s) for s in args.strata.split(",")]
    jobs: list[object] = []
    for stratum in strata:
        for i in range(args.problems):
            pid = S.problem_id("train", stratum, i)
            jobs.append((stratum, pid, args.budget_s, args.max_retries))

    print(
        f"[oracle] {len(jobs)} jobs ({len(strata)} strata x {args.problems}), "
        f"budget {args.budget_s}s, {args.workers} workers",
        flush=True,
    )
    results = run_parallel(
        _oracle_worker, jobs, workers=args.workers, heartbeat_s=30.0, label="oracle"
    )

    per_stratum: dict[int, dict[str, object]] = {}
    for stratum in strata:
        rows = [
            r for r in results if isinstance(r, OracleResult) and r.stratum == stratum
        ]
        solved = [r for r in rows if r.certified_feasible and r.t_oracle is not None]
        times = sorted(float(r.t_oracle) for r in solved)  # type: ignore[arg-type]
        rec: dict[str, object] = {
            "n": len(rows),
            "n_solved": len(solved),
            "solve_rate": (len(solved) / len(rows)) if rows else 0.0,
            "t_oracle_max": (max(times) if times else None),
            "t_oracle_p95": (
                statistics.quantiles(times, n=20)[-1]
                if len(times) >= 2
                else (times[0] if times else None)
            ),
            "t_oracle_median": (statistics.median(times) if times else None),
            "cap_r_max_x1p2": (round(max(times) * 1.2, 2) if times else None),
            "cap_r_p95_x1p5": (
                round(statistics.quantiles(times, n=20)[-1] * 1.5, 2)
                if len(times) >= 2
                else None
            ),
            "mean_refiner_calls": (
                round(statistics.mean(r.n_refiner_calls for r in solved), 2)
                if solved
                else None
            ),
        }
        per_stratum[stratum] = rec
        print(
            f"  r{stratum}: solved {rec['n_solved']}/{rec['n']} "
            f"({rec['solve_rate']:.0%}), t_max={rec['t_oracle_max']}, "
            f"cap_r(max x1.2)={rec['cap_r_max_x1p2']}, calls~{rec['mean_refiner_calls']}",
            flush=True,
        )

    n_err = sum(1 for r in results if not isinstance(r, OracleResult))
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "config": {
                    "problems_per_stratum": args.problems,
                    "budget_s": args.budget_s,
                    "max_retries": args.max_retries,
                    "num_sampling_attempts_per_step": 10,
                    "cap_rule": "max(feasible t_oracle) * 1.2",
                },
                "per_stratum": {str(k): v for k, v in per_stratum.items()},
                "n_worker_errors": n_err,
            },
            indent=2,
        )
    )
    print(f"[oracle] wrote {_OUT} (worker errors: {n_err})", flush=True)


if __name__ == "__main__":
    main()
