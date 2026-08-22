"""Restock3D **v2** parameter sweep: geometry plan-generation + oracle-plan refinement.

For every (n_tall, n_short) config in a grid (default 1..4 × 1..4 = 16 configs; the 5-object columns
are dropped because 5 objects overflow a section — 5+5 refine fails on the 5th cube), over N problems:

* **Plan generation (geometry heuristic):** attempts for
  :class:`GeometryGuidedRestockPlanGenerator` to first emit the oracle plan (oracle south-to-north pick
  order AND all talls via ``place_tall``). Reported per cell as success / mean±std / 95% CI.
* **Oracle-plan refinement:** run :func:`oracle_v2.refine_skeleton_v2` on the oracle skeleton
  (``attempts_per_step=18`` — the certified budget; a ``--refine-cap-s`` wall-clock cap counts a capped
  refinement as unsolved). Reported per cell as solve-rate / mean±std wall-clock (s, over solved) / 95% CI.

The env is only registered for r0..r3, so arbitrary configs are built by injecting a temporary
``STRATA``/``CLUTTER_PER_STRATUM`` entry (validated: the floor sampler packs 2..8 objects, the abstractor
is config-agnostic). Refinement is the cost (~40-60 s/problem), so tasks run in a process pool using
~80% of the CPUs; each worker is pinned to a single BLAS thread for clean parallelism + honest wall-clock.

Results go to the CLI **and** a stored ``.md`` (+ a machine JSON). Per-problem plan-gen attempt indices
are ``PYTHONHASHSEED``-dependent (quote per-cell aggregates); refine solve/time are seed-deterministic.

    python experiments/spectre/restock3d_v2_heuristic_sweep.py --talls 1-4 --shorts 1-4 --problems 10
"""

from __future__ import annotations

# --- single-thread BLAS per worker (clean N-way parallelism + honest wall-clock) BEFORE numpy import.
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). -----------
import glob
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
pathlib.Path(_B).mkdir(parents=True, exist_ok=True)
for _a, (_sd, _pt) in {
    "liblapack.a": ("lapack", "liblapack.so.3*"),
    "libblas.a": ("blas", "libblas.so.3*"),
}.items():
    _lk = pathlib.Path(_B) / _a
    if not (_lk.exists() or _lk.is_symlink()):
        _cs = sorted(
            glob.glob(f"/usr/lib/x86_64-linux-gnu/{_sd}/{_pt}")
            + glob.glob(f"/usr/lib/x86_64-linux-gnu/{_pt}")
        )
        _r = next((c for c in _cs if os.path.isfile(c)), None)
        if _r:
            _lk.symlink_to(_r)

import argparse
import itertools
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from alphatamp.approaches.spectre.envs.restock3d import generator as G
from alphatamp.approaches.spectre.envs.restock3d import kinematic_env as KE
from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import (
    build_skeleton_v2,
    build_v2_bundle,
    refine_skeleton_v2,
    solve_assignment_v2,
)
from alphatamp.approaches.spectre.envs.restock3d.plan_generator_v2 import (
    GeometryGuidedRestockPlanGenerator,
    pick_distance_from_state,
)

_OUT_JSON = Path("data/spectre/derived/restock3d_v2/heuristic_sweep.json")
_OUT_MD = Path("data/spectre/derived/restock3d_v2/heuristic_sweep_results.md")
_TALL_PREFIX = "block_goal"
_GOAL_PREFIXES = ("cube_goal", _TALL_PREFIX)


# --- Match + stats helpers (mirror restock3d_v2_heuristic_eval; duplicated so this parallel script is
# self-contained under fork/spawn and needs no sibling import). ---------------------------------------
def _pick_order(action_plan) -> list[str]:
    return [op.parameters[1].name for op in action_plan if op.name == "pick"]


def _talls_feasible(action_plan, tall_set: set[str]) -> bool:
    """Every tall block is placed via ``place_tall`` (a ``place_short`` on a tall is
    F3)."""
    return all(
        op.name == "place_tall"
        for op in action_plan
        if op.name in ("place_tall", "place_short")
        and op.parameters[1].name in tall_set
    )


def _first_match_indices(
    gen, x0, s0, goal, oracle_order, tall_set, k_max, timeout
) -> tuple[Optional[int], Optional[int], int]:
    """First (talls-feasible-oracle-order, pick-order-only) 1-based attempts; None if
    not found."""
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    bpg.add_abstract_state_node(s0)
    bpg.add_state_node(x0)
    bpg.add_state_abstractor_edge(x0, s0)
    first_full: Optional[int] = None
    first_pick: Optional[int] = None
    n = 0
    for idx, (_state_plan, action_plan) in enumerate(
        itertools.islice(gen(x0, s0, goal, timeout, bpg), k_max)
    ):
        n = idx + 1
        matches_order = _pick_order(action_plan) == oracle_order
        if first_pick is None and matches_order:
            first_pick = idx + 1
        if (
            first_full is None
            and matches_order
            and _talls_feasible(action_plan, tall_set)
        ):
            first_full = idx + 1
        if first_full is not None:
            break
    return first_full, first_pick, n


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_ci(
    vals: list[float], n_boot: int = 10000, seed: int = 0
) -> tuple[Optional[float], Optional[float]]:
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _v2_operators() -> tuple:
    """The 4 count-independent lifted operators (mirrors ``models_v2`` — no sim/kinder
    needed)."""
    from relational_structs import LiftedAtom, LiftedOperator, Variable

    from alphatamp.approaches.spectre.envs.restock3d import models_v2 as M

    robot = Variable("?robot", M.RobotType)
    target = Variable("?target", M.CubeType)
    pick = LiftedOperator(
        "pick",
        [robot, target],
        preconditions={
            LiftedAtom(M.HandEmpty, [robot]),
            LiftedAtom(M.OnFloor, [target]),
        },
        add_effects={LiftedAtom(M.Holding, [robot, target])},
        delete_effects={
            LiftedAtom(M.HandEmpty, [robot]),
            LiftedAtom(M.OnFloor, [target]),
        },
    )

    def _place(name: str, stored_pred) -> LiftedOperator:
        return LiftedOperator(
            name,
            [robot, target],
            preconditions={LiftedAtom(M.Holding, [robot, target])},
            add_effects={
                LiftedAtom(M.HandEmpty, [robot]),
                LiftedAtom(stored_pred, [target]),
            },
            delete_effects={LiftedAtom(M.Holding, [robot, target])},
        )

    operators = {
        pick,
        _place("place_tall", M.Stored),
        _place("place_short", M.Stored),
        _place("place_buffer", M.OnBuffer),
    }
    types = {M.RobotType, M.CubeType}
    predicates = {M.HandEmpty, M.Holding, M.OnFloor, M.Stored, M.OnBuffer}
    return types, predicates, operators


_OPS_CACHE: Optional[tuple] = None


def _ops() -> tuple:
    global _OPS_CACHE
    if _OPS_CACHE is None:
        _OPS_CACHE = _v2_operators()
    return _OPS_CACHE


def _key(n_tall: int, n_short: int) -> int:
    return 900 + n_tall * 10 + n_short


def _inject(n_tall: int, n_short: int) -> int:
    """Register a temporary STRATA/CLUTTER entry for an arbitrary (n_short, n_tall)
    config."""
    k = _key(n_tall, n_short)
    G.STRATA[k] = (n_short, n_tall, n_tall, n_short)
    G._CLUTTER_PER_STRATUM[k] = 0
    KE.CLUTTER_PER_STRATUM[k] = 0
    return k


def _task(params: tuple) -> dict:
    """One (config, problem): geometry plan-gen attempts + oracle-plan refinement (solve
    + wall)."""
    n_tall, n_short, i, k_max, lam, attempts, cap_s, timeout = params
    k = _inject(n_tall, n_short)  # idempotent; robust under spawn
    types, predicates, operators = _ops()
    bundle = build_v2_bundle(k)
    seed = (n_tall * 10 + n_short) * 100000 + i * 997
    try:
        x0, _ = bundle.sim.reset(seed=seed)
        goal_names = [o.name for o in x0 if o.name.startswith(_GOAL_PREFIXES)]
        tall_set = {n for n in goal_names if n.startswith(_TALL_PREFIX)}
        oracle_order = sorted(
            goal_names, key=lambda n: x0.get_object_pose(n).position[1]
        )

        # --- Plan generation (geometry heuristic) ---
        s0 = bundle.abstractor.state_abstractor(x0)
        goal = bundle.abstractor.goal_deriver(x0)
        geo = GeometryGuidedRestockPlanGenerator(
            types,
            predicates,
            operators,
            seed=seed,
            pick_distance=pick_distance_from_state(x0, goal_names),
            lam=lam,
        )
        geo_attempts, _pick_only, _n = _first_match_indices(
            geo, x0, s0, goal, oracle_order, tall_set, k_max, timeout
        )

        # --- Oracle-plan refinement (18 retries/step, cap_s wall-clock cap) ---
        order = build_skeleton_v2(
            x0, solve_assignment_v2(bundle.section_infos, goal_names)
        )
        t0 = time.perf_counter()
        solved, _final, _frames, note = refine_skeleton_v2(
            bundle, x0, order, seed=seed, attempts_per_step=attempts, max_seconds=cap_s
        )
        wall = time.perf_counter() - t0
    finally:
        bundle.sim.close()
    return {
        "n_tall": n_tall,
        "n_short": n_short,
        "i": i,
        "n_total": n_tall + n_short,
        "geo_attempts": geo_attempts,
        "solved": bool(solved),
        "wall": wall,
        "note": note,
    }


def _agg(vals: list[float]) -> dict:
    """Mean / std / bootstrap-95%-CI over a value list (None-safe)."""
    if not vals:
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    lo, hi = _bootstrap_ci([float(v) for v in vals])
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "ci95": [lo, hi],
    }


def _parse_range(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def _cell_summary(rows: list[dict], n: int) -> dict:
    attempts = [r["geo_attempts"] for r in rows if r["geo_attempts"] is not None]
    solved_walls = [r["wall"] for r in rows if r["solved"]]
    n_solved = sum(1 for r in rows if r["solved"])
    return {
        "plan_gen": {
            "n_found": len(attempts),
            "success": len(attempts) / n,
            "success_ci95": list(_wilson(len(attempts), n)),
            **_agg(attempts),
        },
        "refine": {
            "n_solved": n_solved,
            "solve_rate": n_solved / n,
            "solve_rate_ci95": list(_wilson(n_solved, n)),
            "wall": _agg(solved_walls),
        },
    }


def _fmt_ci(ci) -> str:
    if ci[0] is None:
        return "—"
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def _fmt_mean_std(a: dict) -> str:
    if a["mean"] is None:
        return "—"
    return f"{a['mean']:.1f}±{a['std']:.1f}"


def _render(cells: dict, talls: list[int], shorts: list[int], meta: dict) -> str:
    """Both tables as markdown (row table + 4×4-style grids)."""
    lines: list[str] = []
    lines.append("# Restock3D v2 — geometry plan-gen + oracle-refinement sweep\n")
    lines.append(
        f"- date: {meta['date']}  ·  grid: n_tall {min(talls)}–{max(talls)} × "
        f"n_short {min(shorts)}–{max(shorts)} ({len(talls)*len(shorts)} configs)\n"
        f"- N={meta['n']} problems/config · K_MAX={meta['k_max']} · λ={meta['lam']} · "
        f"attempts/step={meta['attempts']} · refine cap={meta['cap_s']} s\n"
        f"- workers={meta['workers']} (single-thread BLAS) · wall-clock under this parallelism\n"
        f"- calibration anchors (solo, 1 thread): r3 (2t,4s) ≈ 37.9 s solved; 5+5 ≈ 56.7 s failed\n"
        f"- geometry plan-gen success is ~100% by construction; the signal is mean±std attempts.\n"
    )

    lines.append(
        "\n## Table A — Geometry plan generation (attempts to generate the oracle plan)\n"
    )
    lines.append("| n_tall | n_short | n_total | success | mean±std | 95% CI |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for nt in talls:
        for ns in shorts:
            c = cells[(nt, ns)]["plan_gen"]
            lines.append(
                f"| {nt} | {ns} | {nt+ns} | {c['n_found']}/{meta['n']} | "
                f"{_fmt_mean_std(c)} | {_fmt_ci(c['ci95'])} |"
            )

    lines.append("\n## Table B — Oracle-plan refinement (solve rate + wall-clock, s)\n")
    lines.append(
        "| n_tall | n_short | n_total | solve rate | wall mean±std (solved) | 95% CI | n_solved |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for nt in talls:
        for ns in shorts:
            c = cells[(nt, ns)]["refine"]
            lines.append(
                f"| {nt} | {ns} | {nt+ns} | {c['n_solved']}/{meta['n']} | "
                f"{_fmt_mean_std(c['wall'])} | {_fmt_ci(c['wall']['ci95'])} | {c['n_solved']} |"
            )

    def _grid(title: str, fn) -> None:
        lines.append(f"\n### {title} (rows n_tall, cols n_short)\n")
        lines.append("| tall\\short | " + " | ".join(str(ns) for ns in shorts) + " |")
        lines.append("|---:|" + "---:|" * len(shorts))
        for nt in talls:
            lines.append(
                f"| **{nt}** | "
                + " | ".join(fn(cells[(nt, ns)]) for ns in shorts)
                + " |"
            )

    _grid("A · geometry attempts mean±std", lambda c: _fmt_mean_std(c["plan_gen"]))
    _grid("B · refine solve-rate", lambda c: f"{c['refine']['n_solved']}/{meta['n']}")
    _grid(
        "B · refine wall-clock mean±std (s, solved)",
        lambda c: _fmt_mean_std(c["refine"]["wall"]),
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--talls", default="1-4")
    parser.add_argument("--shorts", default="1-4")
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=2000)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--attempts-per-step", type=int, default=18)
    parser.add_argument("--refine-cap-s", type=float, default=90.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--workers", type=int, default=max(1, int(0.8 * (os.cpu_count() or 1)))
    )
    args = parser.parse_args()

    talls = _parse_range(args.talls)
    shorts = _parse_range(args.shorts)
    for nt in talls:  # inject in the parent so forked workers inherit
        for ns in shorts:
            _inject(nt, ns)

    tasks = [
        (
            nt,
            ns,
            i,
            args.k_max,
            args.lam,
            args.attempts_per_step,
            args.refine_cap_s,
            args.timeout,
        )
        for nt in talls
        for ns in shorts
        for i in range(args.problems)
    ]
    total = len(tasks)
    print(
        f"[v2-sweep] {len(talls)}×{len(shorts)} configs × {args.problems} = {total} tasks, "
        f"workers={args.workers}, attempts/step={args.attempts_per_step}, cap={args.refine_cap_s}s",
        flush=True,
    )

    results: list[dict] = []
    t_start = time.perf_counter()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_task, t) for t in tasks]
        for done, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            results.append(r)
            if done % max(1, total // 20) == 0 or done == total:
                el = time.perf_counter() - t_start
                eta = el / done * (total - done)
                print(
                    f"  {done}/{total} done ({el:.0f}s, ETA {eta:.0f}s) "
                    f"last: t{r['n_tall']}s{r['n_short']} #{r['i']} "
                    f"geo={r['geo_attempts']} solved={r['solved']} wall={r['wall']:.1f}s",
                    flush=True,
                )

    cells = {
        (nt, ns): _cell_summary(
            [r for r in results if r["n_tall"] == nt and r["n_short"] == ns],
            args.problems,
        )
        for nt in talls
        for ns in shorts
    }

    date = time.strftime("%Y-%m-%d", time.gmtime())
    meta = {
        "date": date,
        "n": args.problems,
        "k_max": args.k_max,
        "lam": args.lam,
        "attempts": args.attempts_per_step,
        "cap_s": args.refine_cap_s,
        "workers": args.workers,
    }
    md = _render(cells, talls, shorts, meta)
    print("\n" + md, flush=True)

    _OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    _OUT_MD.write_text(md)
    _OUT_JSON.write_text(
        json.dumps(
            {
                "config": meta,
                "cells": {
                    f"t{nt}_s{ns}": cells[(nt, ns)] for nt in talls for ns in shorts
                },
                "rows": results,
            },
            indent=2,
        )
    )
    print(f"[v2-sweep] wrote {_OUT_MD} and {_OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
