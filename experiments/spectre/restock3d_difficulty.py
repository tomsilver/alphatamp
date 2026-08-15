"""Restock3D difficulty probe: baseline (hff-order) false positives vs an oracle, per
stratum.

Runs the real collection loop (:func:`collect_episode`) on a small per-stratum sample and reads
feasibility off the **real refiner** (a skeleton is feasible iff it refined). The naive planner-order
false-positive count is ``first_success_idx`` (how many hff-ranked skeletons the planner tries and
fails before the first that refines); an oracle with geometric knowledge picks a feasible one first
(FP 0). The gap should grow with stratum. Also reports the F2/F3 failure mix (F1 is deferred in v1).

Run from the repo root (venv active)::

    python experiments/spectre/restock3d_difficulty.py --strata 0,1,2,3 --per-stratum 8 --k-max 30
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
import statistics
from collections import Counter

from alphatamp.approaches.spectre.collect import collect_episode
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.restock3d import strata as S


def _classify(failures: list[dict]) -> str:
    """F2 (movable culprit) / F3 (culprit-free, proved) / other, from the deepest
    record."""
    if not failures:
        return "none"
    f = failures[0]
    if f.get("culprits"):
        return "F2"
    if f.get("exhausted") and not f.get("budget_exhausted"):
        return "F3"
    return "other"


def _run_stratum(
    stratum: int,
    per_stratum: int,
    k_max: int,
    split: str,
    plan_generator: str = "closed_form",
):
    cfg = CollectionConfig(
        env_id=f"spectre/Restock3D-r{stratum}-v0",
        env_variant="restock3d_v1",
        model_name="restock3d",
        model_kwargs={"stratum": stratum},
        split=split,  # type: ignore[arg-type]
        num_problems=per_stratum,
        problem_seed_start=S.problem_id(split, stratum, 0),
        problem_seed_end=S.problem_id(split, stratum, 0) + max(1, per_stratum),
        K_max=k_max,
        plan_generator=plan_generator,  # type: ignore[arg-type]
        abstract_plan_timeout_s=30.0,
        refinement_timeout_s=20.0,
        num_sampling_attempts_per_step=10,  # config default; per calibration (was 3)
        max_trajectory_steps=500,
    )
    fps: list[int] = []
    solved = 0
    families: Counter = Counter()
    for index in range(per_stratum):
        pid = S.problem_id(split, stratum, index)
        ep = collect_episode(cfg, problem_id=pid)
        if ep.summary.first_success_idx is not None:
            solved += 1
            fps.append(int(ep.summary.first_success_idx))  # baseline (naive-order) FP
        for o in ep.outcomes:
            if o.outcome == "fail" and o.refiner_metadata:
                families[_classify(o.refiner_metadata.get("failures", []))] += 1
    return fps, solved, per_stratum, families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    # Default reports the plain-hff baseline FP; --eager measures the informed order instead.
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    plan_generator = "astar_eager" if args.eager else "closed_form"

    print(
        f"{'stratum':>7} {'solve':>7} {'FP_mean':>8} {'FP_sd':>7} {'FP_max':>7}  families",
        flush=True,
    )
    for stratum in [int(s) for s in args.strata.split(",")]:
        fps, solved, n, fam = _run_stratum(
            stratum, args.per_stratum, args.k_max, args.split, plan_generator
        )
        mean = statistics.mean(fps) if fps else float("nan")
        sd = statistics.pstdev(fps) if len(fps) > 1 else 0.0
        mx = max(fps) if fps else 0
        print(
            f"r{stratum:<6} {solved}/{n:<5} {mean:8.2f} {sd:7.2f} {mx:7d}  {dict(fam)}",
            flush=True,
        )
    print(
        "\nFP = naive hff-order false positives before first feasible (oracle FP = 0). "
        "Expect FP and the F2/F3 mix to grow with stratum."
    )


if __name__ == "__main__":
    main()
