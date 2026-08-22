"""Restock3D **v2** K_max estimation: first-feasible index under plain-hff pool order (no refinement).

Mirrors the v1 K_max method (`restock3d_kmax.py`): enumerate the skeleton pool under the plain-hff
generator (the generic `RelationalHeuristicSearchAbstractPlanGenerator` — for restock, `closed_form`
falls through to exactly this) and record the index of the first **abstractly-feasible** member;
`K_max_r = ceil(max plain first-feasible index * 1.2)`. No refinement, no collection pipeline.

Because v2's `place_tall`/`place_short` have identical abstract effects, EVERY goal-reaching skeleton
satisfies the abstract goal, so feasibility is a **geometric proxy** (`is_feasible_v2`) analogous to
v1's `is_feasible_skeleton`, checking:
  * **F3** — no `place_short(tall_block)` (an upright block into the short section collides its ceiling).
  * **reach-over** — every object is picked only after its south reach-blockers are stored (the same
    `_blocks_reach` corridor the v1 reach_blockers table uses), from the initial floor layout.
  * **section-capacity** — no section is assigned more than `--n-cap` objects (the continuous-packing
    analog of v1's single-object-region over-assignment / F2; a **geometric estimate**, see `--n-cap`).

    python experiments/spectre/restock3d_v2_kmax.py --strata 0,1,2,3 --problems 20 --k-max 200 --n-cap 5
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
import math
import statistics
from pathlib import Path
from typing import Optional

import kinder
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import strata as S
from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    _blocks_reach,
)
from alphatamp.approaches.spectre.envs.restock3d.models_v2 import (
    create_restock3d_v2_models,
)

_OUT = Path("data/spectre/derived/restock3d_v2/kmax_estimate.json")
_TALL_PREFIX = "block_goal"


def _compute_reach_blockers(x0, goal_names: list[str]) -> dict[str, set[str]]:
    """Which goal objects block each goal's front-grasp reach, from the initial floor
    layout."""

    def pos(n: str):
        return x0.get_object_pose(n).position

    def tall(n: str) -> bool:
        return n.startswith(_TALL_PREFIX)

    rb: dict[str, set[str]] = {}
    for b in goal_names:
        rb[b] = {
            a
            for a in goal_names
            if a != b and _blocks_reach(pos(a), tall(a), pos(b), tall(b))
        }
    return rb


def is_feasible_v2(
    action_plan,
    reach_blockers: dict[str, set[str]],
    tall_goals: set[str],
    n_cap: Optional[int],
) -> bool:
    """Geometric feasibility proxy for a v2 skeleton (F3 + reach-over + section-
    capacity)."""
    stored: set[str] = set()
    counts = {"tall": 0, "short": 0}
    for op in action_plan:
        name = op.name
        target = op.parameters[1].name if len(op.parameters) > 1 else None
        if name == "pick":
            if any(a not in stored for a in reach_blockers.get(target, ())):
                return False  # reach-over: a south blocker still on the floor
        elif name == "place_tall":
            stored.add(target)
            counts["tall"] += 1
        elif name == "place_short":
            if target in tall_goals:
                return False  # F3: tall block into the short section
            stored.add(target)
            counts["short"] += 1
        # place_buffer is inert (never in the pool on the no-clutter strata)
    if n_cap is not None and (counts["tall"] > n_cap or counts["short"] > n_cap):
        return False  # section over-capacity (continuous-packing analog of v1 F2)
    return True


def _first_feasible_index(pool, reach_blockers, tall_goals, n_cap) -> Optional[int]:
    for idx, (_state_plan, action_plan) in enumerate(pool):
        if is_feasible_v2(action_plan, reach_blockers, tall_goals, n_cap):
            return idx
    return None


def _enumerate_and_measure(
    stratum: int, problem_id: int, k_max: int, n_cap: Optional[int], timeout_s: float
) -> dict:
    """Build v2 models for one problem, draw the plain-hff pool, return first-feasible +
    pool stats."""
    env = kinder.make(
        f"spectre/Restock3D-r{stratum}-v0"
    )  # low-level env shared with v1
    try:
        obs, _ = env.reset(seed=problem_id)
        models = create_restock3d_v2_models(
            env.observation_space, env.action_space, stratum
        ).models
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        goal_names = [
            o.name for o in x0 if o.name.startswith(("cube_goal", _TALL_PREFIX))
        ]
        tall_goals = {n for n in goal_names if n.startswith(_TALL_PREFIX)}
        reach_blockers = _compute_reach_blockers(x0, goal_names)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen = RelationalHeuristicSearchAbstractPlanGenerator(
            models.types,
            models.predicates,
            models.operators,
            heuristic_name="hff",
            seed=problem_id,
        )
        pool = list(itertools.islice(gen(x0, s0, goal, timeout_s, bpg), k_max))
        return {
            "stratum": stratum,
            "problem_id": problem_id,
            "pool": len(pool),
            "ff_noc": _first_feasible_index(pool, reach_blockers, tall_goals, None),
            "ff_cap": _first_feasible_index(pool, reach_blockers, tall_goals, n_cap),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--problems", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=200)
    # Section-capacity cap (max objects assigned to one section band). Geometric estimate: the
    # object-center band is ~0.522 m and a front-place needs ~0.09 m centre spacing (object 0.05 +
    # gripper clearance), so a section holds ~5-6 objects; 5 is the conservative default. This is the
    # continuous-packing analog of v1's single-object regions -- a measured effective capacity is a
    # follow-up refinement.
    parser.add_argument("--n-cap", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    register_extra_envs()
    strata = [int(s) for s in args.strata.split(",")]
    per_stratum: dict[str, dict[str, object]] = {}
    for stratum in strata:
        rows = [
            _enumerate_and_measure(
                stratum,
                S.problem_id("train", stratum, i),
                args.k_max,
                args.n_cap,
                args.timeout,
            )
            for i in range(args.problems)
        ]
        for key, tag in (
            ("ff_noc", "F3+reach"),
            ("ff_cap", f"F3+reach+cap{args.n_cap}"),
        ):
            found = [r[key] for r in rows if r[key] is not None]
            n_cens = sum(1 for r in rows if r[key] is None)
            kmax_r = math.ceil(max(found) * 1.2) if found else None
            per_stratum.setdefault(str(stratum), {})[key] = {
                "tag": tag,
                "ff_max": max(found) if found else None,
                "ff_median": statistics.median(found) if found else None,
                "censored": n_cens,
                "K_max_r": kmax_r,
            }
        pool_med = statistics.median(r["pool"] for r in rows)
        per_stratum[str(stratum)]["pool_median"] = pool_med
        c = per_stratum[str(stratum)]
        print(
            f"  r{stratum} (n={len(rows)}, pool~{pool_med:.0f}): "
            f"F3+reach ff_max={c['ff_noc']['ff_max']} K={c['ff_noc']['K_max_r']} "
            f"(cens {c['ff_noc']['censored']}); "
            f"+cap{args.n_cap} ff_max={c['ff_cap']['ff_max']} K={c['ff_cap']['K_max_r']} "
            f"(cens {c['ff_cap']['censored']})",
            flush=True,
        )

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "config": {
                    "problems_per_stratum": args.problems,
                    "K": args.k_max,
                    "n_cap": args.n_cap,
                    "kmax_rule": "ceil(max(plain-hff first-feasible index) * 1.2)",
                    "feasibility": "geometric proxy: no place_short(tall) [F3] + south-to-north reach + section<=n_cap",
                },
                "per_stratum": per_stratum,
            },
            indent=2,
        )
    )
    print(f"[v2-kmax] wrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
