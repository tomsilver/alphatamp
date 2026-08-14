"""Collect the ``restock3d_v1`` dataset (EpisodeRecords for the SPECTRE pipeline).

For each split x stratum, generate scenes, enumerate the astar/hff skeleton pool, classify every
candidate with the geometric feasibility gate (``refine.evaluate_skeleton`` — the deterministic
symbolic walk, DD-7), and persist one ``EpisodeRecord`` per solvable problem under
``data/spectre/raw/restock3d_v1/<split>/episodes/``. Unsolvable problems (no feasible candidate in
the pool) are dropped and the index advances (reject-resample; the ``num_success >= 1`` keep rule).
Stratum indices are laid into ``compare.stratum_of``-recoverable pid bands (``strata.problem_id``).

The failure evidence per candidate rides in ``OutcomeRecord.refiner_metadata["failures"]`` in the
canonical shape the env-agnostic SPECTRE downstream consumes (culprits for F2, culprit-free
exhausted for F3), so vocab/train/score work with no per-environment change.

    # smoke: a few kept problems per stratum, one split
    python experiments/spectre/restock3d_collect.py --split train --per-stratum 3 --max-index 40
    # full-ish: 400/100/100 across the four strata
    python experiments/spectre/restock3d_collect.py --split train --per-stratum 100
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
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_K_MAX = 200  # skeleton pool cap
_PLAN_TIMEOUT_S = 30.0


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


def collect_problem(split: str, stratum: int, index: int):
    """Build one EpisodeRecord for ``(split, stratum, index)``, or None if
    unsolvable."""
    # pylint: disable=import-outside-toplevel
    import gymnasium
    import kinder
    from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
        AbstractPlanGenerator,
    )
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    from alphatamp.approaches.spectre.envs.restock3d import generator as gen_mod
    from alphatamp.approaches.spectre.envs.restock3d import strata as strata_mod
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
    from alphatamp.approaches.spectre.schema import (
        EpisodeRecord,
        OutcomeRecord,
        ProvenanceBlock,
        SkeletonRecord,
        SummaryBlock,
    )

    pid = strata_mod.problem_id(split, stratum, index)
    spec = gen_mod.build_spec(pid, stratum)
    cfg = gen_mod.build_task_config(spec)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        task_path = tf.name
    eid = f"kinder/Restock3D-collect-{pid}-v0"
    if eid not in gymnasium.registry:
        gymnasium.register(
            id=eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={"task_config_path": task_path, "scene_render_camera": "task_view"},
        )
    env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    t0 = time.perf_counter()
    try:
        obs, _ = env.reset(seed=pid)
        n_obj = len(cfg["goal_objects"])
        models = create_restock3d_models(
            env.observation_space, env.action_space, task_path, num_objects=n_obj
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        region_infos = load_region_infos(task_path, x0)
        dims = object_dims(x0, CubeType)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen: AbstractPlanGenerator = RelationalHeuristicSearchAbstractPlanGenerator(
            models.types, models.predicates, models.operators, "hff", seed=pid
        )
        pool = list(itertools.islice(gen(x0, s0, goal, _PLAN_TIMEOUT_S, bpg), _K_MAX))
    finally:
        env.close()
        os.unlink(task_path)

    skeletons: list = []
    outcomes: list = []
    first_success: int | None = None
    for idx, (state_plan, action_plan) in enumerate(pool):
        verdict = evaluate_skeleton(state_plan, action_plan, region_infos, dims)
        skeletons.append(
            SkeletonRecord(
                skeleton_idx=idx,
                operator_seq=tuple(action_plan),
                final_abstract_state=state_plan[-1],
            )
        )
        outcome = "success" if verdict.feasible else "fail"
        if outcome == "success" and first_success is None:
            first_success = idx
        failures = [] if verdict.feasible else [verdict.failure]
        outcomes.append(
            OutcomeRecord(
                skeleton_idx=idx,
                outcome=outcome,  # type: ignore[arg-type]
                refinement_wall_clock_s=0.0,
                refinement_seed=idx,
                refiner_metadata={"failures": failures},
            )
        )

    if first_success is None:
        return None  # unsolvable — reject-resample

    summary = SummaryBlock(
        num_skeletons=len(skeletons),
        num_success=sum(1 for o in outcomes if o.outcome == "success"),
        num_fail=sum(1 for o in outcomes if o.outcome == "fail"),
        num_error=0,
        first_success_idx=first_success,
        total_wall_clock_s=time.perf_counter() - t0,
        pool_truncated=len(pool) >= _K_MAX,
    )
    provenance = ProvenanceBlock(
        problem_id=pid,
        env_id="restock3d/Restock3D-v0",
        env_variant=strata_mod.ENV_VARIANT,
        split=split,
        config_hash=_config_hash(),
        problem_seed=pid,
        git_sha="restock3d_collect_v1",
        collection_timestamp="",
        package_versions={},
        gen_params={
            "stratum": stratum,
            "sigma_tall": spec.sigma_tall,
            "sigma_short": spec.sigma_short,
            "n_small": spec.n_small,
            "n_tall": spec.n_tall,
        },
    )
    return EpisodeRecord(
        provenance=provenance,
        initial_abstract_state=s0,
        goal_atoms=frozenset(goal.atoms),
        object_registry={o.name: o.type.name for o in s0.objects},
        skeleton_pool=tuple(skeletons),
        outcomes=tuple(outcomes),
        summary=summary,
        scene_geometry=None,  # SPECTRE is abstract-first; PIGINet crops are a follow-on
    )


def _config_hash() -> str:
    import hashlib

    from alphatamp.approaches.spectre.envs.restock3d import strata as strata_mod
    from alphatamp.approaches.spectre.envs.restock3d.models import (
        HandEmpty,
        Holding,
        InRegion,
        OnFloor,
        Stored,
    )

    preds = sorted(p.name for p in (HandEmpty, Holding, OnFloor, InRegion, Stored))
    payload = json.dumps(
        {"env_variant": strata_mod.ENV_VARIANT, "predicates": preds, "v": 1},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def collect_split(split: str, per_stratum: int, max_index: int, out_root: Path) -> None:
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.restock3d import strata as strata_mod
    from alphatamp.approaches.spectre.io import atomic_write_pickle_gz

    episodes_dir = out_root / strata_mod.ENV_VARIANT / split / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for stratum in strata_mod.STRATA:
        kept = 0
        t0 = time.time()
        for index in range(max_index):
            if kept >= per_stratum:
                break
            pid = strata_mod.problem_id(split, stratum, index)
            path = episodes_dir / f"ep_{pid}.pkl.gz"
            if path.exists():
                kept += 1
                continue
            try:
                ep = collect_problem(split, stratum, index)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                print(
                    f"  r{stratum} idx {index}: ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            if ep is None:
                continue
            atomic_write_pickle_gz(ep, path)
            kept += 1
            if kept % 5 == 0 or kept == per_stratum:
                el = time.time() - t0
                print(
                    f"  r{stratum}: kept {kept}/{per_stratum} (idx {index}, {el/60:.1f}m, "
                    f"pool {ep.summary.num_skeletons}, succ {ep.summary.num_success})",
                    flush=True,
                )
        print(f"[{split}] r{stratum}: {kept} episodes", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument(
        "--max-index", type=int, default=2000, help="reject-resample index cap"
    )
    ap.add_argument("--out-root", default=str(REPO / "data" / "spectre" / "raw"))
    args = ap.parse_args(argv)
    _ensure_blas_symlinks()
    collect_split(args.split, args.per_stratum, args.max_index, Path(args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
