"""Restock3D **v2** head-to-head: geometry-informed A* pick-cost vs geometry-blind hff.

For each of ``--problems`` r3 problems we enumerate the abstract skeleton pool (no refinement) under
two planners and record how many plan-generation attempts each takes to first emit *the oracle plan*:

* **hff** — the stock ``RelationalHeuristicSearchAbstractPlanGenerator`` (geometry-blind unit costs).
* **geometry** — :class:`GeometryGuidedRestockPlanGenerator`, whose pick cost
  ``1 + lam*(# nearer OnFloor)`` front-loads the nearest-first (oracle south-to-north) pick order.

**Match metric** (per the eval protocol): a generated skeleton matches iff its **pick order equals the
oracle south-to-north order** AND **both tall blocks are placed via** ``place_tall`` (never
``place_short`` = F3). The 4 cubes' section op is free (orthogonal to feasibility and to the pick
cost). This is ~1/2880 of the ~46k-skeleton r3 space, so hff hits it rarely while geometry front-loads
the oracle pick order and only draws the tall-section lottery within that leading band. We also report
the *pick-order-only* match as a diagnostic (geometry ~attempt 1).

Stats are reported **per planner independently** (no cross-planner intersection): success-rate (Wilson
95% CI) and mean / std / bootstrap-95% CI of attempts over that planner's own found problems. NOTE:
the geometry cost is *constructed* to rank the oracle pick order first, so its low attempt count is
expected by design; the measured quantity is how many attempts / how often geometry-blind hff stumbles
onto the same order. mean-over-found is survivorship-biased low for the weaker planner (read it with
the success-rate).

    python experiments/spectre/restock3d_v2_heuristic_eval.py --strata 3 --problems 10 --k-max 1000
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). ----------
import glob
import os
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
import time
from pathlib import Path
from typing import Optional

import kinder
import numpy as np
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import strata as S
from alphatamp.approaches.spectre.envs.restock3d.models_v2 import (
    create_restock3d_v2_models,
)
from alphatamp.approaches.spectre.envs.restock3d.plan_generator_v2 import (
    GeometryGuidedRestockPlanGenerator,
    pick_distance_from_state,
)

_OUT = Path("data/spectre/derived/restock3d_v2/heuristic_eval.json")
_TALL_PREFIX = "block_goal"
_GOAL_PREFIXES = ("cube_goal", _TALL_PREFIX)


def _pick_order(action_plan) -> list[str]:
    return [op.parameters[1].name for op in action_plan if op.name == "pick"]


def _talls_feasible(action_plan, tall_set: set[str]) -> bool:
    """Every tall block is placed via ``place_tall`` (a ``place_short`` on a tall is
    F3)."""
    for op in action_plan:
        if (
            op.name in ("place_tall", "place_short")
            and op.parameters[1].name in tall_set
        ):
            if op.name != "place_tall":
                return False
    return True


def _first_match_indices(
    gen, x0, s0, goal, oracle_order, tall_set, k_max, timeout
) -> tuple[Optional[int], Optional[int], int]:
    """Enumerate the pool; return (first talls-feasible-oracle-order attempt, first
    pick-order-only attempt, n_enumerated).

    Attempts are 1-based; ``None`` if not found within ``k_max``.
    """
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
        if (
            first_full is not None
        ):  # a full match implies the pick-order match is already set
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


def _summ(name: str, found: list[int], n_total: int) -> dict:
    lo, hi = _wilson(len(found), n_total)
    blo, bhi = _bootstrap_ci([float(v) for v in found])
    return {
        "planner": name,
        "n_total": n_total,
        "n_found": len(found),
        "success_rate": len(found) / n_total if n_total else None,
        "success_rate_ci95": [lo, hi],
        "mean_attempts": float(np.mean(found)) if found else None,
        "std_attempts": float(np.std(found, ddof=1)) if len(found) > 1 else 0.0,
        "attempts_ci95": [blo, bhi],
        "attempts": sorted(found),
    }


def _eval_stratum(
    stratum: int, n_problems: int, k_max: int, lam: float, timeout: float
):
    rows: list[dict] = []
    found: dict[str, list[int]] = {
        "hff": [],
        "geo": [],
        "hff_pickorder": [],
        "geo_pickorder": [],
    }
    for i in range(n_problems):
        pid = S.problem_id("train", stratum, i)
        env = kinder.make(f"spectre/Restock3D-r{stratum}-v0")
        try:
            obs, _ = env.reset(seed=pid)
            models = create_restock3d_v2_models(
                env.observation_space, env.action_space, stratum
            ).models
            x0 = models.observation_to_state(obs)
            s0 = models.state_abstractor(x0)
            goal = models.goal_deriver(x0)
            goal_names = [o.name for o in x0 if o.name.startswith(_GOAL_PREFIXES)]
            tall_set = {n for n in goal_names if n.startswith(_TALL_PREFIX)}
            # Oracle south-to-north order == sort goal objects by y (oracle_v2.build_skeleton_v2 key).
            oracle_order = sorted(
                goal_names, key=lambda n: x0.get_object_pose(n).position[1]
            )
            pick_distance = pick_distance_from_state(x0, goal_names)

            t0 = time.perf_counter()
            hff: RelationalHeuristicSearchAbstractPlanGenerator = (
                RelationalHeuristicSearchAbstractPlanGenerator(
                    models.types,
                    models.predicates,
                    models.operators,
                    heuristic_name="hff",
                    seed=pid,
                )
            )
            hff_full, hff_pick, hff_n = _first_match_indices(
                hff, x0, s0, goal, oracle_order, tall_set, k_max, timeout
            )
            geo = GeometryGuidedRestockPlanGenerator(
                models.types,
                models.predicates,
                models.operators,
                seed=pid,
                pick_distance=pick_distance,
                lam=lam,
            )
            geo_full, geo_pick, geo_n = _first_match_indices(
                geo, x0, s0, goal, oracle_order, tall_set, k_max, timeout
            )
        finally:
            env.close()
        rows.append(
            {
                "problem_id": pid,
                "hff": hff_full,
                "hff_pickorder": hff_pick,
                "hff_enum": hff_n,
                "geo": geo_full,
                "geo_pickorder": geo_pick,
                "geo_enum": geo_n,
            }
        )
        for key, val in (
            ("hff", hff_full),
            ("geo", geo_full),
            ("hff_pickorder", hff_pick),
            ("geo_pickorder", geo_pick),
        ):
            if val is not None:
                found[key].append(val)
        print(
            f"  r{stratum} pid={pid} ({time.perf_counter()-t0:.1f}s): "
            f"hff full={hff_full} pick={hff_pick} (enum {hff_n}) | "
            f"geo full={geo_full} pick={geo_pick} (enum {geo_n})",
            flush=True,
        )

    return {
        "stratum": stratum,
        "rows": rows,
        "summary": {
            "full_match": {
                "hff": _summ("hff", found["hff"], n_problems),
                "geometry": _summ("geometry", found["geo"], n_problems),
            },
            "pick_order_only": {
                "hff": _summ("hff", found["hff_pickorder"], n_problems),
                "geometry": _summ("geometry", found["geo_pickorder"], n_problems),
            },
        },
    }


def _print_summary(res: dict) -> None:
    s = res["summary"]["full_match"]
    print(
        f"\n=== r{res['stratum']} — full match (oracle pick order + talls feasible) ==="
    )
    print(
        f"{'planner':10s} {'success':>10s} {'mean':>8s} {'std':>7s} {'attempts CI95':>18s}"
    )
    for name in ("hff", "geometry"):
        d = s[name]
        sr = f"{d['n_found']}/{d['n_total']}"
        mean = "—" if d["mean_attempts"] is None else f"{d['mean_attempts']:.1f}"
        std = f"{d['std_attempts']:.1f}"
        ci = d["attempts_ci95"]
        cis = "—" if ci[0] is None else f"[{ci[0]:.1f}, {ci[1]:.1f}]"
        print(f"{name:10s} {sr:>10s} {mean:>8s} {std:>7s} {cis:>18s}")
    sp = res["summary"]["pick_order_only"]
    print(
        "  (pick-order-only diagnostic: "
        f"hff {sp['hff']['n_found']}/{sp['hff']['n_total']} "
        f"mean {sp['hff']['mean_attempts']}; "
        f"geometry {sp['geometry']['n_found']}/{sp['geometry']['n_total']} "
        f"mean {sp['geometry']['mean_attempts']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="3")
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=1000)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    register_extra_envs()
    strata = [int(x) for x in args.strata.split(",")]
    results = {}
    for stratum in strata:
        print(
            f"[v2-heuristic-eval] r{stratum}: {args.problems} problems, "
            f"K_max={args.k_max}, lam={args.lam}",
            flush=True,
        )
        res = _eval_stratum(stratum, args.problems, args.k_max, args.lam, args.timeout)
        _print_summary(res)
        results[str(stratum)] = res

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "config": {
                    "problems_per_stratum": args.problems,
                    "k_max": args.k_max,
                    "lam": args.lam,
                    "match_metric": "oracle south-to-north pick order AND both talls via place_tall",
                    "d_of_o": "object y-coordinate (northward reach from the park corridor), from x0",
                    "note": "per-planner stats, no cross-planner intersection; geometry constructed "
                    "to rank the oracle pick order first (low attempts expected by design)",
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n[v2-heuristic-eval] wrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
