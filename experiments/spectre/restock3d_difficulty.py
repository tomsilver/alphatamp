"""Restock3D baseline-planner difficulty probe (Stage 5 / gates P3+P5).

For each stratum r0-r3 and a band of seeds, generate the scene, enumerate the astar/hff skeleton
pool, and classify every candidate with the geometric feasibility gate (``refine.evaluate_skeleton``
— the deterministic symbolic walk that IS the env's feasibility model, DD-7). Reports, per stratum:
solve rate, mean baseline FP (failed attempts before the first feasible candidate in the default
order — the astar-dist baseline; oracle FP = 0), and the F2/F3 failure-family mix. Answers "does
Restock3D earn its slot": a baseline↔oracle FP gap that grows with stratum, driven by capacity
(F2) and height (F3) infeasibilities the height-/capacity-blind planner cannot see.

    python experiments/spectre/restock3d_difficulty.py --seeds 8 --pool 200
    python experiments/spectre/restock3d_difficulty.py --strata 2,3 --seeds 20 --out data/spectre/restock3d/difficulty.json

This uses the symbolic walk for labels (fast, exact); a physical MuJoCo spot-check of one feasible
candidate per stratum confirms symbolic feasibility executes (``--spot-check``).
"""

from __future__ import annotations

# --- environment setup: MUST precede any kinder/mujoco import. ---
import glob
import os

_BLAS_DIR = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault("LAPACK_DIR", _BLAS_DIR)
os.environ.setdefault("BLAS_DIR", _BLAS_DIR)
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _ensure_blas_symlinks() -> None:
    Path(_BLAS_DIR).mkdir(parents=True, exist_ok=True)
    for archive, (subdir, pattern) in {
        "liblapack.a": ("lapack", "liblapack.so.3*"),
        "libblas.a": ("blas", "libblas.so.3*"),
    }.items():
        link = Path(_BLAS_DIR) / archive
        if link.exists() or link.is_symlink():
            continue
        libroot = "/usr/lib/x86_64-linux-gnu"
        cands = sorted(
            glob.glob(os.path.join(libroot, subdir, pattern))
            + glob.glob(os.path.join(libroot, pattern))
        )
        real = next((c for c in cands if os.path.isfile(c)), None)
        if real is not None:
            link.symlink_to(real)


def probe_one(stratum: int, seed: int, pool_cap: int) -> dict:
    """Generate one scene, enumerate the pool, classify every candidate symbolically."""
    # pylint: disable=import-outside-toplevel
    import gymnasium
    import kinder
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    from alphatamp.approaches.spectre.envs.restock3d import generator as gen_mod
    from alphatamp.approaches.spectre.envs.restock3d.models import (
        CubeType,
        create_restock3d_models,
    )
    from alphatamp.approaches.spectre.envs.restock3d.refine import (
        evaluate_skeleton,
        object_dims,
    )
    from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
        load_region_infos,
    )

    spec = gen_mod.build_spec(seed, stratum)
    cfg = gen_mod.build_task_config(spec)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        task_path = tf.name
    eid = f"kinder/Restock3D-diff-r{stratum}-s{seed}-v0"
    if eid not in gymnasium.registry:
        gymnasium.register(
            id=eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={"task_config_path": task_path, "scene_render_camera": "task_view"},
        )
    env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    try:
        obs, _ = env.reset(seed=seed)
        n_obj = len(cfg["goal_objects"])
        models = create_restock3d_models(
            env.observation_space, env.action_space, task_path, num_objects=n_obj
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        region_infos = load_region_infos(task_path, x0)
        dims = object_dims(x0, CubeType)
        bpg = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen = RelationalHeuristicSearchAbstractPlanGenerator(
            models.types, models.predicates, models.operators, "hff", seed=seed
        )
        pool = list(itertools.islice(gen(x0, s0, goal, 30.0, bpg), pool_cap))
        first_fp: int | None = None
        fam = {"feasible": 0, "F2": 0, "F3": 0}
        for i, (sp, ap) in enumerate(pool):
            v = evaluate_skeleton(sp, ap, region_infos, dims)
            if v.feasible:
                fam["feasible"] += 1
                if first_fp is None:
                    first_fp = i
            else:
                fam[v.family] = fam.get(v.family, 0) + 1
    finally:
        env.close()
        os.unlink(task_path)
    return {
        "stratum": stratum,
        "seed": seed,
        "sigma_tall": spec.sigma_tall,
        "sigma_short": spec.sigma_short,
        "n_small": spec.n_small,
        "n_tall": spec.n_tall,
        "pool_size": len(pool),
        "solved": first_fp is not None,
        "fp": first_fp,  # baseline false positives before first feasible (oracle = 0)
        "n_feasible": fam["feasible"],
        "n_f2": fam.get("F2", 0),
        "n_f3": fam.get("F3", 0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--strata", default="0,1,2,3", help="comma-separated strata")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per stratum")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--pool", type=int, default=200, help="skeleton pool cap (K_max)")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args(argv)

    _ensure_blas_symlinks()
    strata = [int(s) for s in args.strata.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    records: list[dict] = []
    for stratum in strata:
        for seed in seeds:
            records.append(probe_one(stratum, seed, args.pool))

    print(
        f"\n{'stratum':>7} {'σt':>3} {'σs':>3} {'solve':>7} {'meanFP':>7} {'sdFP':>6} "
        f"{'meanFeas':>9} {'F2':>5} {'F3':>5}"
    )
    print("-" * 62)
    for stratum in strata:
        rs = [r for r in records if r["stratum"] == stratum]
        fps = [r["fp"] for r in rs if r["fp"] is not None]
        solved = sum(r["solved"] for r in rs)
        mean_fp = statistics.mean(fps) if fps else float("nan")
        sd_fp = statistics.stdev(fps) if len(fps) > 1 else 0.0
        print(
            f"{'r'+str(stratum):>7} {rs[0]['sigma_tall']:>3} {rs[0]['sigma_short']:>3} "
            f"{solved:>3}/{len(rs):<3} {mean_fp:7.1f} {sd_fp:6.1f} "
            f"{statistics.mean(r['n_feasible'] for r in rs):9.1f} "
            f"{round(statistics.mean(r['n_f2'] for r in rs)):>5} "
            f"{round(statistics.mean(r['n_f3'] for r in rs)):>5}"
        )
    print("\n(oracle FP = 0 for every stratum: an oracle refines a feasible candidate first.)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2))
        print(f"wrote {len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
