"""Run SPECTRE data collection for one split of one env variant.

Parallelism modes (composable):

1. **Within-job worker pool (recommended for SLURM)**: set ``workers=N`` to fan
   out N problem_ids in parallel via ``ProcessPoolExecutor``. On a cluster
   this is paired with ``#SBATCH --cpus-per-task=N`` so one SLURM job
   saturates its allocated cores::

       python experiments/spectre/spectre_collect.py workers=8 \
           split=train problem_seed_start=0 problem_seed_end=500

2. **Hydra multirun (for heterogeneous / very-many-job scheduling)**: sweep
   ``problem_ids`` (single-element lists) and let the Hydra launcher spawn
   one job per element::

       python experiments/spectre/spectre_collect.py -m \
           'problem_ids=[[0]],[[1]],[[2]]' hydra/launcher=joblib

       python experiments/spectre/spectre_collect.py -m \
           'problem_ids=[[0]],[[1]],[[2]]' hydra/launcher=slurm

3. **Sequential batch (default, debugging)**: ``workers=1`` runs the
   ``[problem_seed_start, problem_seed_end)`` range inline, one id at a
   time.

All modes write to the same on-disk layout::

    <data_root>/raw/<env_variant>/<split>/episodes/ep_<problem_id>.pkl.gz

Existing files are skipped, so re-running after a partial failure costs only
the missing problem_ids.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from alphatamp.approaches.spectre.collect import (
    collect_and_save_result,
    save_config_yaml,
)
from alphatamp.approaches.spectre.config import CollectionConfig


def _build_config(cfg: DictConfig) -> CollectionConfig:
    env = cfg.env
    model_kwargs = cast(dict, OmegaConf.to_container(env.model_kwargs, resolve=True))
    split = cast(Literal["train", "val", "test"], str(cfg.split))
    state_path_depth = cast(Literal["s0_sL_only", "full"], str(cfg.state_path_depth))
    return CollectionConfig(
        env_id=env.env_id,
        env_variant=env.env_variant,
        model_name=env.model_name,
        model_kwargs=model_kwargs,
        split=split,
        num_problems=int(cfg.num_problems),
        problem_seed_start=int(cfg.problem_seed_start),
        problem_seed_end=int(cfg.problem_seed_end),
        K_max=int(cfg.K_max),
        abstract_plan_timeout_s=float(cfg.abstract_plan_timeout_s),
        refinement_timeout_s=float(cfg.refinement_timeout_s),
        num_sampling_attempts_per_step=int(cfg.num_sampling_attempts_per_step),
        max_trajectory_steps=int(cfg.max_trajectory_steps),
        heuristic_name=str(cfg.heuristic_name),
        refinement_seed_rule=str(cfg.refinement_seed_rule),
        collect_instrumentation=bool(cfg.collect_instrumentation),
        state_path_depth=state_path_depth,
    )


def _resolve_problem_ids(cfg: DictConfig, spectre_cfg: CollectionConfig) -> list[int]:
    """Decide which ``problem_id``s this invocation should collect.

    Precedence:

    1. ``cfg.problem_ids`` (explicit list) — used as-is after dedup + sort.
    2. Otherwise the range ``[problem_seed_start, problem_seed_end)``.
    """
    override = cfg.get("problem_ids", None)
    if override is not None:
        resolved = OmegaConf.to_container(override, resolve=True)
        if not isinstance(resolved, list) or not resolved:
            raise ValueError(f"problem_ids must be a non-empty list, got {override!r}")
        return sorted({int(pid) for pid in resolved})
    return list(range(spectre_cfg.problem_seed_start, spectre_cfg.problem_seed_end))


def _run_sequential(
    spectre_cfg: CollectionConfig, data_root: Path, ids: list[int]
) -> None:
    for problem_id in ids:
        pid, path, err = collect_and_save_result(spectre_cfg, data_root, problem_id)
        if err is not None:
            print(f"[problem_id={pid}] FAILED: {err}")
        else:
            print(f"[problem_id={pid}] wrote {path}")


def _run_parallel(
    spectre_cfg: CollectionConfig,
    data_root: Path,
    ids: list[int],
    workers: int,
) -> None:
    # ``spawn`` is required because pyperplan / bilevel_planning keep
    # module-level caches that don't survive a ``fork`` concurrently.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(collect_and_save_result, spectre_cfg, data_root, pid): pid
            for pid in ids
        }
        for fut in as_completed(futures):
            pid, path, err = fut.result()
            if err is not None:
                print(f"[problem_id={pid}] FAILED: {err}")
            else:
                print(f"[problem_id={pid}] wrote {path}")


@hydra.main(
    config_path="conf",
    config_name="spectre_collect",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for SPECTRE data collection."""
    spectre_cfg = _build_config(cfg)
    data_root = Path(cfg.data_root)
    # Write once in the parent so multiple workers don't race on the YAML file.
    save_config_yaml(spectre_cfg, data_root)

    ids = _resolve_problem_ids(cfg, spectre_cfg)
    workers = int(cfg.get("workers", 1))

    print(
        f"SPECTRE collect: env_variant={spectre_cfg.env_variant}"
        f" split={spectre_cfg.split}"
        f" problem_ids={ids[0]}..{ids[-1]} ({len(ids)})"
        f" workers={workers}"
        f" config_hash={spectre_cfg.config_hash}"
    )

    if workers <= 1 or len(ids) <= 1:
        _run_sequential(spectre_cfg, data_root, ids)
    else:
        _run_parallel(spectre_cfg, data_root, ids, workers)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
