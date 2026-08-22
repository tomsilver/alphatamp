"""Collect Shelf3D (dynamic3d / MuJoCo TidyBot) difficulty data under the baseline
planner.

Quantifies how hard vanilla ``kinder/Shelf3D-o{1,2,8}-v0`` is for the SPECTRE baseline
astar planner *before* the env's config is modified into harder variants. For each
problem (= variant x seed) it enumerates the astar skeleton pool, refines candidates one
at a time with a per-plan-attempt time budget, and records for every attempted plan
whether it succeeded / failed and its refinement wall-clock. Those per-attempt records
are all the notebook (``shelf3d_difficulty.py``) needs to compute, per variant: mean/std
FP (failed attempts before first success), mean/std wall-clock-to-first-success, and
solve rate.

Two modes, one code path:
  * ``--pilot`` (non-short-circuit): refine *every* candidate at a generous per-attempt
    budget, no video. Because ``BacktrackingRefiner`` is deterministic given its seed
    and monotone in the timeout, recording each attempt's success wall-clock lets the
    notebook re-derive the metrics under *any smaller* per-attempt budget offline -- so
    one run sweeps every budget. Used to pick the budget for the full collection.
  * full (default, short-circuit): refine in astar order, stop at the first plan that
    refines, and replay that plan into an ``.mp4`` under ``data/spectre/shelf3D-kinder``.

Why this is a standalone script and not ``collect.collect_episode``: that engine is
non-short-circuit-only, discards the refined ``Plan`` (so it cannot make a video), and
its ``EpisodeRecord`` validation expects a scene-geometry layer we would have to
synthesize for 3D. This reuses the same *primitives*
(``RelationalHeuristicSearchAbstractPlanGenerator``,
``ParameterizedControllerTrajectorySampler``, ``BacktrackingRefiner``) wired exactly as
``pure_planning_approach.py`` / ``collect.py``, and emits a lean per-problem JSON.

Three environment facts, resolved in the 2026-08-12 Phase-0 de-risk, are baked in below:
  * The dynamic3d ``Shelf3D`` gym ids are *not* auto-registered (only kinematic3d is),
    so we register them ourselves against ``TidyBot3DEnv`` (which also gives the boxed
    obs space the model factory asserts on).
  * Headless rendering must be ``MUJOCO_GL=egl`` with ``PYOPENGL_PLATFORM`` *unset*
    (``register_all_environments`` would force osmesa, whose PyOpenGL is broken here);
    we therefore skip ``register_all_environments`` entirely.
  * The TidyBot arm IK (``ikfast_kortex``) is compiled on first use and links
    ``liblapack.a`` / ``libblas.a`` by explicit path via ``LAPACK_DIR`` / ``BLAS_DIR``;
    the static archives are not installed and there is no sudo, so we point those dirs
    at symlinks named ``*.a`` that target the installed shared libs (ld links by ELF
    type, not extension; the module loads ``libblas.so.3`` at runtime).

Problems are independent, so they run concurrently over a ``spawn`` process pool.
``spawn`` (not ``fork``) is required: bilevel_planning / pyperplan keep module-level
caches that do not survive a concurrent fork. MuJoCo is heavy, so the default is a
modest worker count.

Usage::

    # pilot: 5 seeds x 15 candidate plans per variant, generous 30s/attempt, no video
    python experiments/spectre/shelf3d_collect.py --pilot \\
        --variants o1,o2,o8 --num-problems 5 --max-abstract-plans 15 \\
        --per-attempt-budget-s 30 --workers 8

    # full: 20 problems/variant, chosen budget, videos of successes
    python experiments/spectre/shelf3d_collect.py \\
        --variants o1,o2,o8 --num-problems 20 --max-abstract-plans 100 \\
        --per-attempt-budget-s 10 --video --workers 8
"""

from __future__ import annotations

# --- environment setup: MUST precede any kinder/mujoco import, and runs in every spawned
# worker (module top re-executes on spawn). kinder is imported lazily inside functions so
# these are always in effect first. ---
import glob
import os

_BLAS_DIR = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("PYOPENGL_PLATFORM", None)  # egl requires this unset (or 'egl')
os.environ.setdefault("LAPACK_DIR", _BLAS_DIR)
os.environ.setdefault("BLAS_DIR", _BLAS_DIR)
os.environ.setdefault("PYTHONHASHSEED", "0")  # inherited by spawned children at startup

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import (  # noqa: E402
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

VARIANTS: dict[str, int] = {"o1": 1, "o2": 2, "o8": 8}
_MODEL_NAME = "tidybot3d_shelf3D"
_HEARTBEAT_S = 30.0
_VIDEO_FPS = 30


@dataclasses.dataclass(frozen=True)
class ShelfConfig:
    """One collection run's knobs (picklable; travels to each spawned worker)."""

    variant: str
    num_objects: int
    per_attempt_budget_s: float
    max_abstract_plans: int
    num_sampling_attempts_per_step: int
    max_trajectory_steps: int
    abstract_plan_timeout_s: float
    heuristic: str
    stop_at_first_success: bool
    record_video: bool
    out_root: str
    tag: str  # "" for the full run, "_pilot" for the pilot (segregates dirs)


# --------------------------------------------------------------------------------------
# One-time environment preparation (BLAS symlinks + gym registration + ikfast warmup).
# --------------------------------------------------------------------------------------
def ensure_blas_symlinks() -> None:
    """Create ``liblapack.a`` / ``libblas.a`` symlinks to the installed *shared* libs.

    ``pybullet_helpers`` compiles ``ikfast_kortex`` on first use and links these as
    explicit file paths located via ``LAPACK_DIR`` / ``BLAS_DIR``. The static ``.a``
    archives ship in ``lib{blas,lapack}-dev``, which are not installed and cannot be
    apt-installed (no sudo). ``ld`` links a file by its ELF type, not its extension, so
    a ``.so`` symlinked as ``.a`` links dynamically; the resulting module needs only
    ``libblas.so.3`` (runtime, installed).
    """
    Path(_BLAS_DIR).mkdir(parents=True, exist_ok=True)
    wanted = {
        "liblapack.a": ("lapack", "liblapack.so.3*"),
        "libblas.a": ("blas", "libblas.so.3*"),
    }
    libroot = "/usr/lib/x86_64-linux-gnu"
    for archive_name, (subdir, pattern) in wanted.items():
        link = Path(_BLAS_DIR) / archive_name
        if link.exists() or link.is_symlink():
            continue
        candidates = sorted(
            glob.glob(os.path.join(libroot, subdir, pattern))
            + glob.glob(os.path.join(libroot, pattern))
        )
        real = next((c for c in candidates if os.path.isfile(c)), None)
        if real is not None:
            link.symlink_to(real)


def register_shelf3d() -> None:
    """Register the dynamic3d Shelf3D gym ids (kinder does not auto-register them)."""
    # pylint: disable=import-outside-toplevel
    import gymnasium
    import kinder

    tasks = Path(kinder.__file__).parent / "envs/dynamic3d/tasks/Shelf3D"
    for num_objects in VARIANTS.values():
        env_id = f"kinder/Shelf3D-o{num_objects}-v0"
        if env_id not in gymnasium.registry:
            gymnasium.register(
                id=env_id,
                entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
                kwargs={
                    "task_config_path": str(tasks / f"Shelf3D-o{num_objects}.json"),
                    "scene_render_camera": "task_view",
                },
            )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _refinement_seed(problem_id: int, skeleton_idx: int) -> int:
    """Stable per-candidate seed (mirrors ``collect._refinement_seed``)."""
    payload = f"v1_blake2b_problem_skeleton:{problem_id}:{skeleton_idx}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFF_FFFFFFFF


# --------------------------------------------------------------------------------------
# Per-problem collection.
# --------------------------------------------------------------------------------------
def _result_dir(cfg: ShelfConfig) -> Path:
    if cfg.tag:  # pilot -> data/spectre/shelf3D-kinder/_pilot/<variant>/
        return Path(cfg.out_root) / cfg.tag / cfg.variant
    return Path(cfg.out_root) / cfg.variant / "results"


def _video_dir(cfg: ShelfConfig) -> Path:
    return Path(cfg.out_root) / cfg.variant


def result_path(cfg: ShelfConfig, problem_id: int) -> Path:
    return _result_dir(cfg) / f"{problem_id:05d}.json"


def _render_plan_video(cfg: ShelfConfig, problem_id: int, states) -> str | None:
    """Render the verified plan's per-step ``states`` to ``<variant>/<pid>.mp4``.

    Renders each stored state directly via ``set_state`` + ``render`` rather than
    replaying the plan's low-level actions through the dynamics. Action-replay
    accumulates MuJoCo numerical drift over a long trajectory -- harmless on o1's single
    pick-place but enough on o2's two that the goal check no longer fires on the
    replayed final state, even though the *planned* trajectory reaches the goal.
    Rendering the states the refiner certified is drift-free and faithful by
    construction. ``plan.states`` is per-step (one per low-level action + 1), so the
    video is smooth. The gym env's ``set_state`` takes a vectorized obs, so each
    ``ObjectCentricState`` is vectorized first.
    """
    # pylint: disable=import-outside-toplevel
    import imageio
    import kinder

    vdir = _video_dir(cfg)
    vdir.mkdir(parents=True, exist_ok=True)
    # allow_state_access=True is required for set_state on the env (the planning sim
    # sets it too); without it set_state raises "State access is not allowed".
    env = kinder.make(
        f"kinder/Shelf3D-{cfg.variant}-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )
    frames = []
    try:
        env.reset(seed=problem_id)
        base = env.unwrapped
        space = base.observation_space
        for state in states:
            # set_state / vectorize are kinder extensions the gym stubs omit.
            base.set_state(space.vectorize(state))  # type: ignore[attr-defined]
            frames.append(env.render())
    finally:
        env.close()
    if not frames:
        return None
    dest = vdir / f"{problem_id:05d}.mp4"
    imageio.mimsave(str(dest), frames, fps=_VIDEO_FPS)  # type: ignore[arg-type]
    return str(dest.relative_to(REPO)) if dest.is_relative_to(REPO) else str(dest)


def collect_problem(cfg: ShelfConfig, problem_id: int) -> dict:
    """Enumerate the pool, refine candidates under the per-attempt budget, record
    outcomes.

    Returns the full per-problem record (also written to disk by the worker).
    """
    # pylint: disable=import-outside-toplevel,line-too-long
    import kinder
    from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
        AbstractPlanGenerator,
    )
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
    from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
    from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
        ParameterizedControllerTrajectorySampler,
    )
    from bilevel_planning.utils import RelationalControllerGenerator
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    register_shelf3d()
    env = kinder.make(f"kinder/Shelf3D-{cfg.variant}-v0")
    try:
        obs, _ = env.reset(seed=problem_id)
        models = create_bilevel_planning_models(
            _MODEL_NAME,
            env.observation_space,
            env.action_space,
            num_objects=cfg.num_objects,
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        gen: AbstractPlanGenerator = RelationalHeuristicSearchAbstractPlanGenerator(
            models.types,
            models.predicates,
            models.operators,
            cfg.heuristic,
            seed=problem_id,
        )
        t_gen = time.perf_counter()
        pool = list(
            itertools.islice(
                gen(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg),
                cfg.max_abstract_plans,
            )
        )
        plan_gen_s = time.perf_counter() - t_gen
        pool_size = len(pool)

        sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(models.skills),
            transition_function=models.transition_fn,
            state_abstractor=models.state_abstractor,
            max_trajectory_steps=cfg.max_trajectory_steps,
        )

        attempts: list[dict] = []
        first_success_idx: int | None = None
        success_plan = None
        for idx, (state_plan, action_plan) in enumerate(pool):
            if hasattr(sampler, "clear"):
                sampler.clear()  # rejections outlive one candidate on some samplers
            refiner = BacktrackingRefiner(
                trajectory_sampler=sampler,
                num_sampling_attempts_per_step=cfg.num_sampling_attempts_per_step,
                seed=_refinement_seed(problem_id, idx),
            )
            start = time.perf_counter()
            attempt_error: str | None = None
            try:
                plan = refiner(
                    x0, state_plan, action_plan, cfg.per_attempt_budget_s, bpg
                )
                outcome = "success" if plan is not None else "fail"
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                plan, outcome = None, "error"
                attempt_error = f"{type(exc).__name__}: {exc}"
            wall = time.perf_counter() - start
            attempts.append(
                {
                    "plan_idx": idx,
                    "outcome": outcome,
                    "wall_clock_s": wall,
                    "timed_out": wall >= cfg.per_attempt_budget_s,
                    "n_ops": len(action_plan),
                    "error": attempt_error,
                }
            )
            if outcome == "success" and first_success_idx is None:
                first_success_idx = idx
                success_plan = plan
                if cfg.stop_at_first_success:
                    break
    finally:
        env.close()

    solved = first_success_idx is not None
    fp = first_success_idx
    if first_success_idx is not None:
        wall_to_first = sum(
            a["wall_clock_s"] for a in attempts[: first_success_idx + 1]
        )
    else:
        wall_to_first = None
    total_wall = sum(a["wall_clock_s"] for a in attempts)

    video_path: str | None = None
    video_ok: bool | None = None
    if solved and cfg.record_video and success_plan is not None:
        # A successful refinement already certifies the plan reaches the goal; re-check the
        # final state's abstraction as a cheap guard that the rendered video is the real one.
        final_abs = models.state_abstractor(success_plan.states[-1])
        video_ok = goal.atoms.issubset(final_abs.atoms)
        video_path = _render_plan_video(cfg, problem_id, success_plan.states)

    return {
        "variant": cfg.variant,
        "env_id": f"kinder/Shelf3D-{cfg.variant}-v0",
        "problem_id": problem_id,
        "seed": problem_id,
        "num_objects": cfg.num_objects,
        "mode": "pilot" if cfg.tag else "full",
        "stop_at_first_success": cfg.stop_at_first_success,
        "per_attempt_budget_s": cfg.per_attempt_budget_s,
        "max_abstract_plans": cfg.max_abstract_plans,
        "num_sampling_attempts_per_step": cfg.num_sampling_attempts_per_step,
        "max_trajectory_steps": cfg.max_trajectory_steps,
        "pool_size": pool_size,
        "pool_truncated": pool_size >= cfg.max_abstract_plans,
        "n_attempts": len(attempts),
        "plan_gen_s": plan_gen_s,
        "attempts": attempts,
        "solved": solved,
        "first_success_idx": first_success_idx,
        "fp": fp,
        "wall_clock_to_first_success_s": wall_to_first,
        "total_wall_clock_s": total_wall,
        "video_path": video_path,
        "video_ok": video_ok,
        "git_sha": _git_sha(),
    }


def _worker(args: tuple[ShelfConfig, int]) -> dict:
    """Pool worker: collect one problem, write its JSON, return a compact verdict.

    Returns a verdict dict (never raises) so one pathological problem cannot strand the
    pool.
    """
    cfg, problem_id = args
    path = result_path(cfg, problem_id)
    start = time.perf_counter()
    if path.exists():
        return {"variant": cfg.variant, "pid": problem_id, "cached": True, "s": 0.0}
    try:
        record = collect_problem(cfg, problem_id)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        return {
            "variant": cfg.variant,
            "pid": problem_id,
            "error": f"{type(exc).__name__}: {exc}",
            "s": time.perf_counter() - start,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)
    return {
        "variant": cfg.variant,
        "pid": problem_id,
        "solved": record["solved"],
        "fp": record["fp"],
        "pool": record["pool_size"],
        "video_ok": record["video_ok"],
        "s": time.perf_counter() - start,
    }


def warmup() -> None:
    """Serially build models + refine one trivial problem, forcing the one-time
    ``ikfast_kortex`` compile before the parallel pool starts (concurrent first-compiles
    would race on the shared build dir)."""
    ensure_blas_symlinks()
    cfg = ShelfConfig(
        variant="o1",
        num_objects=1,
        per_attempt_budget_s=30.0,
        max_abstract_plans=1,
        num_sampling_attempts_per_step=1,
        max_trajectory_steps=500,
        abstract_plan_timeout_s=30.0,
        heuristic="hff",
        stop_at_first_success=True,
        record_video=False,
        out_root="/tmp",
        tag="_warmup",
    )
    t0 = time.perf_counter()
    rec = collect_problem(cfg, 0)
    print(
        f"[warmup] o1 seed 0: solved={rec['solved']} in {time.perf_counter()-t0:.1f}s "
        f"(ikfast ready)",
        flush=True,
    )


# --------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------
def _heartbeat(done: int, total: int, inflight: int, t0: float, tallies: dict) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 and done else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    per = " ".join(
        f"{v}:{d['solved']}/{d['done']}✓" for v, d in sorted(tallies.items())
    )
    print(
        f"[{elapsed/60:6.1f}m] done {done}/{total}  inflight {inflight}  "
        f"ETA {eta/60:.0f}m  | {per}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variants", default="o1,o2,o8", help="comma-separated subset of o1,o2,o8"
    )
    ap.add_argument(
        "--num-problems", type=int, default=20, help="problems (seeds) per variant"
    )
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--per-attempt-budget-s",
        type=float,
        default=10.0,
        help="wall-clock budget for refining ONE candidate plan",
    )
    ap.add_argument("--max-abstract-plans", type=int, default=100)
    ap.add_argument(
        "--num-sampling-attempts-per-step",
        type=int,
        default=20,
        help="BacktrackingRefiner samples/step; kept high so the TIME budget is "
        "the binding constraint",
    )
    ap.add_argument(
        "--max-trajectory-steps",
        type=int,
        default=500,
        help="per-skill horizon (kinder shelf skills need up to ~400)",
    )
    ap.add_argument("--abstract-plan-timeout-s", type=float, default=60.0)
    ap.add_argument("--heuristic", default="hff")
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="non-short-circuit (refine every candidate) + '_pilot' dirs + no "
        "video; run once at a generous budget and re-derive smaller budgets",
    )
    ap.add_argument(
        "--video",
        action="store_true",
        help="record a video of each success "
        "(full mode only; ignored under --pilot)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", default="data/spectre/shelf3D-kinder")
    a = ap.parse_args(argv)

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants {unknown}; choose from {sorted(VARIANTS)}")

    out_root = a.out_root
    if not os.path.isabs(out_root):
        out_root = str(REPO / out_root)
    tag = "_pilot" if a.pilot else ""

    configs = {
        v: ShelfConfig(
            variant=v,
            num_objects=VARIANTS[v],
            per_attempt_budget_s=a.per_attempt_budget_s,
            max_abstract_plans=a.max_abstract_plans,
            num_sampling_attempts_per_step=a.num_sampling_attempts_per_step,
            max_trajectory_steps=a.max_trajectory_steps,
            abstract_plan_timeout_s=a.abstract_plan_timeout_s,
            heuristic=a.heuristic,
            stop_at_first_success=not a.pilot,
            record_video=a.video and not a.pilot,
            out_root=out_root,
            tag=tag,
        )
        for v in variants
    }

    seeds = list(range(a.seed_start, a.seed_start + a.num_problems))
    jobs = [(configs[v], pid) for v in variants for pid in seeds]
    total = len(jobs)
    print(
        f"shelf3d collect [{'PILOT' if a.pilot else 'FULL'}]: {total} problems "
        f"({variants} x {a.num_problems} seeds), "
        f"budget={a.per_attempt_budget_s}s/attempt, K_max={a.max_abstract_plans}, "
        f"samples/step={a.num_sampling_attempts_per_step}, "
        f"horizon={a.max_trajectory_steps}, workers={a.workers}",
        flush=True,
    )
    print(f"out_root={out_root}  video={configs[variants[0]].record_video}", flush=True)

    ensure_blas_symlinks()
    warmup()

    tallies = {v: {"solved": 0, "done": 0} for v in variants}
    errors: list[str] = []
    t0 = time.time()
    last_beat = t0
    done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
        futures: dict[Future, tuple[str, int]] = {
            pool.submit(_worker, job): (job[0].variant, job[1]) for job in jobs
        }
        pending = set(futures)
        while pending:
            finished, pending = wait(
                pending, timeout=_HEARTBEAT_S, return_when=FIRST_COMPLETED
            )
            for fut in finished:
                variant, pid = futures[fut]
                done += 1
                tallies[variant]["done"] += 1
                try:
                    res = fut.result()
                except BaseException as exc:  # pylint: disable=broad-exception-caught
                    errors.append(f"{variant}/{pid}: {type(exc).__name__}: {exc}")
                    continue
                if res.get("error"):
                    errors.append(f"{variant}/{pid}: {res['error']}")
                elif res.get("solved"):
                    tallies[variant]["solved"] += 1
            if time.time() - last_beat >= _HEARTBEAT_S:
                _heartbeat(done, total, len(pending), t0, tallies)
                last_beat = time.time()

    print(f"\n=== shelf3d collect done ({(time.time()-t0)/60:.1f} min) ===", flush=True)
    print(f"{'variant':<8}{'solved':<9}{'total':<7}", flush=True)
    for v in variants:
        print(f"{v:<8}{tallies[v]['solved']:<9}{tallies[v]['done']:<7}", flush=True)
    if errors:
        print(f"\n{len(errors)} error(s):", flush=True)
        for e in errors[:10]:
            print(f"  {e}", flush=True)
    print(f"\nresults under {out_root}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
