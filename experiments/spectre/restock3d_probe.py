"""Restock3D baseline-FP probe: how deep in the hff-ranked pool is the first refinable
skeleton?

For each (stratum, problem) it draws the hff skeleton pool and refines candidates **in order**, one at
a time, stopping at the first that fully refines. The index of that first success is the naive
planner-order **false-positive count** (an oracle with geometric knowledge would pick a feasible one
first -> FP 0). Prints per-candidate outcomes and the F2/F3 family mix.

Real-collision refinement of an N-object skeleton is 2N flaky controller rollouts, so this uses a
generous per-step retry count and per-candidate timeout. Small K_max / few problems by default.

Run from the repo root (venv active)::

    python experiments/spectre/restock3d_probe.py --strata 0,1 --problems 2 --k-max 30
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
from collections import Counter

import kinder
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

from alphatamp.approaches.spectre.collect import (
    _make_env_models,
    _make_plan_generator,
    _make_refiner,
    _make_trajectory_sampler,
    _refinement_seed,
)
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import strata as S
from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    failure_metadata,
)


def _cfg(stratum: int, k_max: int) -> CollectionConfig:
    start = S.problem_id("train", stratum, 0)
    return CollectionConfig(
        env_id=f"spectre/Restock3D-r{stratum}-v0",
        env_variant="restock3d_v1",
        model_name="restock3d",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=start,
        problem_seed_end=start + 1,
        K_max=k_max,
        abstract_plan_timeout_s=20.0,
        refinement_timeout_s=180.0,  # a 2N-rollout skeleton needs room
        num_sampling_attempts_per_step=15,  # placements are ~1/6 reliable -> retry generously
        max_trajectory_steps=500,
    )


def _classify(failures) -> str:
    if not failures:
        return "none"
    f = failures[0]
    if f.get("culprits"):
        return "F2"
    if f.get("exhausted") and not f.get("budget_exhausted"):
        return "F3"
    return "other"


def probe(stratum: int, pid: int, k_max: int):
    cfg = _cfg(stratum, k_max)
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=pid)
        env_models = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen = _make_plan_generator(cfg, env_models, obs, pid, x0)
        pool = list(
            itertools.islice(gen(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg), k_max)
        )
        sampler = _make_trajectory_sampler(cfg, env_models)
        fam: Counter = Counter()
        first = None
        for idx, (state_plan, action_plan) in enumerate(pool):
            if hasattr(sampler, "clear"):
                sampler.clear()
            seed = _refinement_seed(cfg.refinement_seed_rule, pid, idx)
            refiner = _make_refiner(cfg, obs, sampler, seed)
            try:
                plan = refiner(
                    x0, state_plan, action_plan, cfg.refinement_timeout_s, bpg
                )
            except BaseException:  # noqa: BLE001
                plan = None
            if plan is not None:
                first = idx
                print(f"    candidate {idx}: SUCCESS  (first refinable)", flush=True)
                break
            f = failure_metadata(
                sampler, action_plan, cfg.num_sampling_attempts_per_step, False
            )
            fam[_classify(f)] += 1
            print(f"    candidate {idx}: fail [{_classify(f)}]", flush=True)
        return first, len(pool), fam
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1")
    parser.add_argument("--problems", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=30)
    args = parser.parse_args()
    print(f"{'stratum':>7} {'problem':>8} {'FP':>5} {'pool':>5}  families", flush=True)
    for stratum in [int(s) for s in args.strata.split(",")]:
        for i in range(args.problems):
            pid = S.problem_id("train", stratum, i)
            first, pool, fam = probe(stratum, pid, args.k_max)
            fp = "none" if first is None else str(first)
            print(
                f"r{stratum:<6} {pid:>8} {fp:>5} {pool:>5}  {dict(fam)}",
                flush=True,
            )
    print(
        "\nFP = # of hff-ranked skeletons that fail refinement before the first that succeeds "
        "(oracle FP = 0). 'none' = no refinable skeleton within K_max."
    )


if __name__ == "__main__":
    main()
