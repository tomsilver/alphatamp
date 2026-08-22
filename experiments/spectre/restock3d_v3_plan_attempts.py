"""Restock3D-v3 plan-generation difficulty: FP-to-first-feasible for hff vs the geometry prior.

For each stratum, over N accepted problems, enumerate each abstract-plan generator's candidate stream
**in order** and count the **FPs** = how many candidates fail the *analytic* refiner
(``feasibility_v3.classify_skeleton`` — pure geometry, no motion planning) before the first that
passes. This is time-to-first-success in plan-gen attempts, for the two production generators:

* **hff** — the stock geometry-blind ``RelationalHeuristicSearchAbstractPlanGenerator``;
* **geometry** — the v2 ``GeometryGuidedRestockPlanGenerator`` (nearest-first pick-cost prior).

The pool is capped at **K=150** (the deployment budget — the real refiner would not try more), so a
problem whose first feasible candidate is deeper is **censored**; a stratum's **solve%** = fraction not
censored. The refiner is analytic (fast); the only cost is hff's deep enumeration on the crowded strata.
**hff on n=9 is skipped** (``_HFF_SKIP_STRATA``): it is censored on ~every accepted problem and its
search graph balloons to ~5 GB/worker, so it is **memory-bound** — 24 workers swap-thrashed a 59 GB box;
the default caps at ~10. Appends a table to ``data/spectre/derived/restock3d_v3/generator_calibration.md``.

Usage:
    python experiments/spectre/restock3d_v3_plan_attempts.py [--k 150] [--n 200] [--workers 10]
"""

from __future__ import annotations

import argparse
import itertools
import os
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d import generator_v3 as G
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S
from alphatamp.approaches.spectre.envs.restock3d.generator import _Rng

_K = 150
_ENV_CACHE: dict = {}  # per-process: stratum -> (env, env_models)


def _steps_of(action_plan):
    return [(op.name, [p.name for p in op.parameters]) for op in action_plan]


def _dims_pos(x0):
    dims, pos = {}, {}
    for o in x0:
        if o.name.startswith("obj_goal"):
            dims[o.name] = (
                2 * x0.get(o, "half_extent_x"),
                2 * x0.get(o, "half_extent_z"),
            )
            p = x0.get_object_pose(o.name).position
            pos[o.name] = (float(p[0]), float(p[1]))
    return dims, pos


def _cfg(stratum, pid, plan_generator, k):
    return CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant="restock3d_v3",
        model_name="restock3d_v3",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=pid,
        problem_seed_end=pid + 1,
        K_max=k,
        plan_generator=plan_generator,
        abstract_plan_timeout_s=300.0,
        refinement_timeout_s=1.0,
        num_sampling_attempts_per_step=1,
        max_trajectory_steps=1,
    )


def _first_feasible(gen, x0, s0, goal, bpg, dims, pos, k):
    """FP = index of the first analytic-feasible candidate (0-based failures before it); (K, True)
    if censored (none feasible within k)."""
    for i, (_sp, ap) in enumerate(itertools.islice(gen(x0, s0, goal, 300.0, bpg), k)):
        if F.classify_skeleton(_steps_of(ap), dims, pos) is None:
            return i, False
    return k, True


def _measure(args):
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    from alphatamp.approaches.spectre import collect as C
    from alphatamp.approaches.spectre.env_registry import register_extra_envs

    stratum, pid, k, run_hff = args
    if stratum not in _ENV_CACHE:
        import kinder

        register_extra_envs()
        env = kinder.make(S.env_id(stratum))
        obs0, _ = env.reset(seed=S.problem_id("train", stratum, 0))
        em = C._make_env_models(
            _cfg(stratum, pid, "closed_form", k),
            env.observation_space,
            env.action_space,
        )
        _ENV_CACHE[stratum] = (env, em)
    env, em = _ENV_CACHE[stratum]
    obs, _ = env.reset(seed=pid)
    x0 = em.observation_to_state(obs)
    s0 = em.state_abstractor(x0)
    goal = em.goal_deriver(x0)
    dims, pos = _dims_pos(x0)

    def run(plan_gen):
        bpg = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen = C._make_plan_generator(_cfg(stratum, pid, plan_gen, k), em, obs, pid, x0)
        return _first_feasible(gen, x0, s0, goal, bpg, dims, pos, k)

    fp_g, cens_g = run("closed_form")
    # hff on the crowded n=9 stratum is skipped: it is censored on essentially every accepted
    # problem (pilot FP 102-421, all > K=150) and enumerating 150 goal-reaching plans there is
    # prohibitively slow (~minutes/problem + ~5 GB search graph). Assumed-fail instead.
    if run_hff:
        fp_h, cens_h = run("heuristic_search")
    else:
        fp_h, cens_h = None, True
    return stratum, fp_g, cens_g, fp_h, cens_h


def _raw_rho(stratum, n_raw=1500):
    """Raw-draw feasible-rate + rho (mean/median), mirroring the calibration probe."""
    p = S.params(stratum)
    solvable = 0
    rhos = []
    for seed in range(n_raw):
        rng = _Rng(seed * 97 + stratum + 7)
        widths = [G._u(rng, F.WIDTH_MIN, F.WIDTH_MAX) for _ in range(p.n)]
        heights = G._sample_heights(rng, p.n, p.n_forced, p.n_near)
        blocks = [F.Block(f"o{i}", widths[i], heights[i]) for i in range(p.n)]
        nf, _t, rho = F.feasible_ratio(blocks)
        if nf >= 1:
            solvable += 1
            rhos.append(rho)
    return (
        solvable / n_raw,
        (st.mean(rhos) if rhos else 0.0),
        (st.median(rhos) if rhos else 0.0),
    )


def _mean_std(xs):
    return (st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--k", type=int, default=_K, help="abstract-plan pool cap (deployment budget)"
    )
    ap.add_argument("--n", type=int, default=200, help="problems per stratum")
    # MEMORY-bound, not CPU-bound: each worker holds 2 PyBullet sims + hff's search graph, which
    # balloons to ~2.5 GB on the n=9 stratum. 24 workers (~50 GB) swap-thrashed a 59 GB box; ~10 keeps
    # the peak near 25 GB. Raise only if you have the RAM headroom for the heavy strata.
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    def _fp_str(solved_fps):
        if not solved_fps:
            return "—"
        m, s = _mean_std(solved_fps)
        return f"{m:.1f} ± {s:.1f}"

    _HFF_SKIP_STRATA = {
        3
    }  # hff assumed-fail on n=9 (censored on ~all accepted problems; too slow)

    rows = {}
    for st_ in S.STRATA:
        t0 = time.perf_counter()
        run_hff = st_ not in _HFF_SKIP_STRATA
        tasks = [
            (st_, S.problem_id("train", st_, i), args.k, run_hff) for i in range(args.n)
        ]
        g_solved, h_solved, gc, hc = [], [], 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for _s, fg, cgi, fh, chi in ex.map(_measure, tasks, chunksize=4):
                if cgi:
                    gc += 1
                else:
                    g_solved.append(fg)
                if fh is not None:  # None => hff was skipped for this stratum
                    if chi:
                        hc += 1
                    else:
                        h_solved.append(fh)
        fr, rmean, rmed = _raw_rho(st_)
        if run_hff:
            hff_solve, hff_fp = (args.n - hc) / args.n, _fp_str(h_solved)
        else:
            hff_solve, hff_fp = 0.0, "— (assumed fail, not run)"
        rows[st_] = {
            "n": S.params(st_).n,
            "feasible_rate": fr,
            "rho_mean": rmean,
            "rho_med": rmed,
            "geom_solve": (args.n - gc) / args.n,
            "geom_fp": _fp_str(g_solved),
            "hff_solve": hff_solve,
            "hff_fp": hff_fp,
        }
        print(f"  stratum {st_} done in {time.perf_counter()-t0:.0f}s", flush=True)

    lines = [
        "",
        "## Plan-generation attempts — solve-rate + FP to first analytic-feasible skeleton",
        "",
    ]
    lines.append(
        f"_hff (stock, geometry-blind) vs the geometry-informed prior "
        f"(`GeometryGuidedRestockPlanGenerator`). **solve%** = fraction of problems whose first "
        f"**analytically-feasible** skeleton (no motion planning) appears within the **K={args.k}** "
        f"pool cap (= the deployment budget; the real refiner would not try more). **FP** = failed "
        f"candidates before that first feasible one, **averaged over solved problems only** "
        f"(a censored problem has no defined FP). {args.n} problems/stratum. **hff on the n=9 "
        f"stratum is not run** — it is censored on essentially every accepted problem (pilot FP "
        f"102-421, all > K) and prohibitively slow to enumerate; reported as assumed-fail._\n"
    )
    lines.append(
        "| stratum | n | feasible-rate | ρ mean | ρ med | hff solve% | hff FP (solved) | geom solve% | geom FP (solved) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for st_ in S.STRATA:
        r = rows[st_]
        lines.append(
            f"| {st_} | {r['n']} | {100*r['feasible_rate']:.0f}% | {r['rho_mean']:.3f} | "
            f"{r['rho_med']:.3f} | {100*r['hff_solve']:.0f}% | {r['hff_fp']} | "
            f"{100*r['geom_solve']:.0f}% | {r['geom_fp']} |"
        )
    report = "\n".join(lines)
    print(report)
    out = Path("data/spectre/derived/restock3d_v3/generator_calibration.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write("\n" + report + "\n")
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
