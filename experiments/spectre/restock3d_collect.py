"""Collect Restock3D episodes (kinematic-PyBullet, real-collision feasibility).

A thin CLI over :func:`alphatamp.approaches.spectre.collect.collect_and_save`: one
``CollectionConfig`` per stratum (env ``spectre/Restock3D-r{r}-v0``), looping over the
per-stratum problem-id band from :mod:`envs.restock3d.strata`. Feasibility
and the F2/F3 failure evidence come from the real controllers +
:class:`RestockRecordingSampler`; no gate.

Keep this pass small (the plan): real BiRRT per candidate is far cheaper than MuJoCo, but
not free.

Run from the repo root (venv active)::

    python experiments/spectre/restock3d_collect.py --strata 0,1,2,3 --per-stratum 20
    python experiments/spectre/restock3d_collect.py --split test --per-stratum 10
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (cached afterwards). ---
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
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from alphatamp.approaches.spectre.collect import collect_and_save
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.restock3d import strata as S

_ENV_VARIANT = "restock3d_v1"


def _config(
    split: str,
    stratum: int,
    per_stratum: int,
    k_max: int,
    plan_generator: str = "closed_form",
) -> CollectionConfig:
    start = S.problem_id(split, stratum, 0)
    return CollectionConfig(
        env_id=f"spectre/Restock3D-r{stratum}-v0",
        env_variant=_ENV_VARIANT,
        model_name="restock3d",
        model_kwargs={"stratum": stratum},
        split=split,  # type: ignore[arg-type]
        num_problems=per_stratum,
        problem_seed_start=start,
        problem_seed_end=start + max(1, per_stratum),
        K_max=k_max,
        plan_generator=plan_generator,  # type: ignore[arg-type]
        abstract_plan_timeout_s=30.0,
        refinement_timeout_s=20.0,
        num_sampling_attempts_per_step=10,  # config default; per calibration (was 3)
        max_trajectory_steps=500,
    )


def _one(args) -> str:
    cfg, data_root, pid = args
    path = collect_and_save(cfg, Path(data_root), pid)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--plan-generator",
        choices=["closed_form", "heuristic_search", "astar_eager"],
        default="closed_form",
        help="astar_eager surfaces feasibles early (accelerator; see the ADR).",
    )
    parser.add_argument("--data-root", default="data/spectre")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    strata = [int(s) for s in args.strata.split(",")]
    jobs = []
    for stratum in strata:
        cfg = _config(
            args.split, stratum, args.per_stratum, args.k_max, args.plan_generator
        )
        for index in range(args.per_stratum):
            pid = S.problem_id(args.split, stratum, index)
            jobs.append((cfg, args.data_root, pid))

    total = len(jobs)
    print(
        f"[restock3d_collect] {args.split}: {total} episodes over strata {strata} "
        f"(K_max={args.k_max}, workers={args.workers})",
        flush=True,
    )
    start = time.perf_counter()
    done = 0
    if args.workers > 1:
        # spawn: pyperplan / bilevel_planning keep module-level state that fork corrupts.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for fut in as_completed(futs):
                fut.result()
                done += 1
                _heartbeat(done, total, start)
    else:
        for j in jobs:
            _one(j)
            done += 1
            _heartbeat(done, total, start)
    print(
        f"[restock3d_collect] done: {total} episodes in {time.perf_counter()-start:.0f}s"
    )


def _heartbeat(done: int, total: int, start: float) -> None:
    if done % 5 == 0 or done == total:
        elapsed = time.perf_counter() - start
        rate = done / max(elapsed, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        print(
            f"[restock3d] {done}/{total} elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
