"""Shared harness for the Restock3D calibration runs (oracle timing, K_max, spot-
checks).

Provides the IKFast BLAS/LAPACK shim, a ``CollectionConfig`` builder, a whole-problem
pool-enumeration helper (used to measure first-feasible index under either order), and a
spawn-based worker-parallel driver with heartbeats (the ``shelf3d_collect.py`` pattern).
Parallelism is **whole-problem only** — each worker builds its own PyBullet env;
candidates within a problem share one sim via ``collect._restock_extras`` and must not
be parallelised.
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

import itertools
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Iterable, Optional

import kinder
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from relational_structs import GroundOperator

from alphatamp.approaches.spectre.collect import (
    _make_env_models,
    _make_plan_generator,
    _restock_extras,
)
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import strata as S
from alphatamp.approaches.spectre.envs.restock3d.eager_tables import (
    EagerTables,
    build_tables,
    is_feasible_skeleton,
)

_PlanGen = str  # "closed_form" (plain hff) | "astar_eager"


def build_cfg(
    stratum: int,
    *,
    k_max: int,
    plan_generator: _PlanGen = "closed_form",
    num_sampling_attempts_per_step: int = 10,
    refinement_timeout_s: float = 300.0,
    abstract_plan_timeout_s: float = 30.0,
    max_trajectory_steps: int = 500,
) -> CollectionConfig:
    """A restock3d CollectionConfig for one problem of ``stratum`` (single-problem
    window)."""
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
        plan_generator=plan_generator,  # type: ignore[arg-type]
        abstract_plan_timeout_s=abstract_plan_timeout_s,
        refinement_timeout_s=refinement_timeout_s,
        num_sampling_attempts_per_step=num_sampling_attempts_per_step,
        max_trajectory_steps=max_trajectory_steps,
    )


def enumerate_pool(
    cfg: CollectionConfig, problem_id: int
) -> tuple[list[tuple[list, list[GroundOperator]]], EagerTables]:
    """Build the env + models + generator for one problem and draw the capped skeleton
    pool.

    Returns ``(pool, tables)``; the generator (plain hff vs eager) is selected by
    ``cfg.plan_generator``. No refinement. The env is closed before returning.
    """
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=problem_id)
        env_models = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen = _make_plan_generator(cfg, env_models, obs, problem_id, x0)
        pool = list(
            itertools.islice(
                gen(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg), cfg.K_max
            )
        )
        tables = build_tables(
            _restock_extras["region_infos"],  # type: ignore[arg-type]
            _restock_extras["goal_names"],  # type: ignore[arg-type]
        )
        return pool, tables
    finally:
        env.close()


def first_feasible_index(
    pool: Iterable[tuple[list, list[GroundOperator]]], tables: EagerTables
) -> Optional[int]:
    """Index of the first pool member classified feasible by the tables (no
    refinement)."""
    for idx, (_state_plan, action_plan) in enumerate(pool):
        if is_feasible_skeleton(action_plan, tables):
            return idx
    return None


def n_f3_candidates(
    pool: Iterable[tuple[list, list[GroundOperator]]], tables: EagerTables
) -> int:
    """Count pool members containing a tall→short place (F3 evidence present, V3
    check)."""
    count = 0
    for _state_plan, action_plan in pool:
        if any(
            a.name == "place"
            and not tables.fits(a.parameters[1].name, a.parameters[2].name)
            for a in action_plan
        ):
            count += 1
    return count


def run_parallel(
    worker_fn: Callable[[object], object],
    jobs: list[object],
    *,
    workers: int,
    heartbeat_s: float = 30.0,
    label: str = "run",
) -> list[object]:
    """Run ``worker_fn`` over ``jobs`` on a spawn ProcessPool, draining with heartbeats
    + ETA.

    ``worker_fn`` must be a top-level (importable) function so it pickles under spawn. A
    worker exception is captured as ``("ERROR", job, repr(exc))`` rather than killing
    the run.
    """
    ctx = mp.get_context("spawn")
    results: list[object] = []
    start = time.time()
    total = len(jobs)
    done_n = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(worker_fn, job): job for job in jobs}
        pending = set(futs)
        while pending:
            done, pending = wait(
                pending, timeout=heartbeat_s, return_when=FIRST_COMPLETED
            )
            for fut in done:
                done_n += 1
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    results.append(("ERROR", futs[fut], repr(exc)))
            elapsed = time.time() - start
            rate = done_n / elapsed if elapsed > 0 else 0.0
            eta = (total - done_n) / rate if rate > 0 else float("inf")
            print(
                f"[{label}] {done_n}/{total} done | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
                flush=True,
            )
    return results
