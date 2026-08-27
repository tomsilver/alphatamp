"""Restock3D-**v3** per-stratum demo videos: real-MP-verified successful plans.

For each stratum (block counts 6/7/8/9 = strata 0/1/2/3) render ``--n-per-stratum`` (default 5)
videos. Each video is an **analytic-success** plan taken from the collected ``restock3d_v3``
train dataset (the labels there are pure-geometry ``feasibility_v3.classify_skeleton`` — NOT
motion planning), **re-verified with the real refiner** (kinder ``BacktrackingRefiner`` + the
v3 ``RestockRecordingSampler`` with the arm-insertion F3 cutoffs), and rendered from the real
low-level state trajectory the refiner produced. An analytic **false positive** (real refine
returns ``None``, or the goal is not actually reached) is skipped and logged.

This deliberately does **not** use the oracle refiner (``oracle_v2``/``oracle_v3`` are
v2-only/stale for v3). It drives everything off the already-collected dataset — the stored
seed recreates the scene, the stored per-skeleton success labels pick the candidate plan — and
recovers the ``(state_plan, action_plan)`` the refiner needs by regenerating the *deterministic*
geometry-guided pool (same generator + seed the collection used) and matching by operator
sequence.

Output layout (one subfolder per stratum, block count in each filename)::

    src/alphatamp/approaches/spectre/envs/restock3d/demos/v3_strata/
      r0/ demo_r0_n6_pid<seed>.mp4 ... r3/ demo_r3_n9_pid<seed>.mp4
      SUMMARY.md

Run (repo root, venv active)::

    # smoke on the cheapest stratum (1 video):
    python experiments/spectre/restock3d_v3_demos.py --strata 0 --n-per-stratum 1
    # full run in the background (~30-60 min; stratum 3 dominates):
    bash experiments/spectre/spectre_run.sh restock3d_v3_demos \
        python experiments/spectre/restock3d_v3_demos.py --strata 0,1,2,3 --n-per-stratum 5
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). ----------
import glob
import os
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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
import gzip
import itertools
import pickle
import time
from typing import Optional

import imageio.v2 as iio
import kinder
import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from pybullet_helpers.camera import capture_image

from alphatamp.approaches.spectre import collect as C
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S

_RAW = pathlib.Path("data/spectre/raw/restock3d_v3/train/episodes")
_OUT = pathlib.Path("src/alphatamp/approaches/spectre/envs/restock3d/demos/v3_strata")
_N_BY_STRATUM = {0: 6, 1: 7, 2: 8, 3: 9}  # block count per stratum (documentation only)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _render(sim) -> np.ndarray:
    return capture_image(
        sim.physics_client_id,
        image_width=640,
        image_height=480,
        **sim.config.get_camera_kwargs(),
    )


def _steps_of(operator_seq) -> tuple:
    """Canonical (op_name, [arg_names]) tuple for a ground-operator sequence."""
    return tuple(
        (op.name, tuple(p.name for p in op.parameters)) for op in operator_seq
    )


def _load_episode(pid: int):
    """Load one train EpisodeRecord by problem id, or None if its file is absent."""
    path = _RAW / f"ep_{pid:05d}.pkl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _stratum_episode_pids(stratum: int) -> list[int]:
    """Train problem ids for a stratum, in seed order, restricted to files present."""
    pids = []
    for path in sorted(_RAW.glob("ep_*.pkl.gz")):
        pid = int(path.name[len("ep_") :].split(".")[0])
        if S.stratum_of(pid) == stratum:
            pids.append(pid)
    return sorted(pids)


# --------------------------------------------------------------------------------------------
# Per-stratum rendering
# --------------------------------------------------------------------------------------------
def render_stratum(
    stratum: int,
    n_want: int,
    attempts_per_step: int,
    frame_skip: int,
    fps: int,
    log,
) -> list[dict]:
    """Produce up to ``n_want`` real-MP-verified demo videos for one stratum.

    Returns a list of per-video summary dicts (pid, skeleton_idx, n_false_positives, frames).
    """
    k_max, r_cap = S.budget(stratum)
    cfg = CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant="restock3d_v3",
        model_name="restock3d_v3",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=0,  # unused by the helpers below (pid passed explicitly)
        problem_seed_end=1,
        K_max=k_max,
        abstract_plan_timeout_s=120.0,
        refinement_timeout_s=r_cap,
        num_sampling_attempts_per_step=attempts_per_step,
        max_trajectory_steps=500,
        plan_generator="closed_form",
        refiner_mode="real",
    )

    out_dir = _OUT / f"r{stratum}"
    out_dir.mkdir(parents=True, exist_ok=True)

    made: list[dict] = []
    for pid in _stratum_episode_pids(stratum):
        if len(made) >= n_want:
            break
        out = out_dir / f"demo_r{stratum}_n{_N_BY_STRATUM[stratum]}_pid{pid}.mp4"
        if out.exists():  # resume: reuse an already-rendered video
            made.append(
                {
                    "pid": pid,
                    "skeleton_idx": "-",
                    "n_false_positives": "-",
                    "frames": "-",
                    "path": str(out),
                }
            )
            log(f"  r{stratum} pid={pid}: reuse existing {out.name} [{len(made)}/{n_want}]")
            continue
        ep = _load_episode(pid)
        if ep is None or ep.summary.num_success == 0:
            continue
        assert ep.provenance.gen_params["stratum"] == stratum, (
            f"stratum banding mismatch: pid {pid} decoded {stratum} "
            f"but stored {ep.provenance.gen_params['stratum']}"
        )

        result = _try_problem(
            cfg, ep, pid, k_max, r_cap, attempts_per_step, frame_skip, log
        )
        if result is None:
            continue
        frames, used_idx, n_fp = result
        iio.mimsave(out, frames, fps=fps, macro_block_size=16)
        made.append(
            {
                "pid": pid,
                "skeleton_idx": used_idx,
                "n_false_positives": n_fp,
                "frames": len(frames),
                "path": str(out),
            }
        )
        log(
            f"  r{stratum} pid={pid}: WROTE {out.name} "
            f"(skeleton {used_idx}, {n_fp} analytic FP skipped, {len(frames)} frames) "
            f"[{len(made)}/{n_want}]"
        )

    if len(made) < n_want:
        log(f"  r{stratum}: only {len(made)}/{n_want} videos (ran out of solvable episodes)")
    return made


def _try_problem(
    cfg,
    ep,
    pid: int,
    k_max: int,
    r_cap: float,
    attempts_per_step: int,
    frame_skip: int,
    log,
) -> Optional[tuple[list, int, int]]:
    """Recreate the scene, regenerate the pool, and real-refine the dataset's analytic-success
    skeletons in index order. Return (frames, used_skeleton_idx, n_false_positives) for the
    first one whose real refinement genuinely reaches the goal, else None.

    Builds env + models FRESH per problem, mirroring ``restock3d_v3_gates.gate_g1`` (the proven
    real-refiner path): x0 comes from the gym env, and the refiner's ``set_state(x0)`` establishes
    the pid scene on the models' internal sim (``_restock_extras['sim']``), which is also what we
    render. Both PyBullet clients are closed on the way out.
    """
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    sim = None
    try:
        obs, _ = env.reset(seed=pid)
        em = C._make_env_models(cfg, env.observation_space, env.action_space)
        sim = C._restock_extras["sim"]
        x0 = em.observation_to_state(obs)
        s0 = em.state_abstractor(x0)
        goal = em.goal_deriver(x0)

        # Sanity: the reconstructed block dims must match the stored scene geometry.
        _assert_dims_match(x0, ep, pid)

        # Regenerate the deterministic pool -> map operator-sequence -> (state_plan, action_plan).
        bpg = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen = C._make_plan_generator(cfg, em, obs, pid, x0)
        pool = list(
            itertools.islice(gen(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg), k_max)
        )
        pool_by_steps = {_steps_of(ap): (sp, ap) for sp, ap in pool}

        # Candidate plans = the dataset's analytic-success skeletons, in index order.
        success_idxs = sorted(o.skeleton_idx for o in ep.outcomes if o.outcome == "success")
        candidates: list[tuple[int, object, object]] = []
        for idx in success_idxs:
            key = _steps_of(ep.skeleton_pool[idx].operator_seq)
            if key in pool_by_steps:
                sp, ap = pool_by_steps[key]
                candidates.append((idx, sp, ap))
        if not candidates:
            # Fallback (regeneration order diverged): live-filter the pool with the same analytic
            # classifier the dataset labels encode, and take the feasible ones.
            dims, pos = C._restock3d_analytic_inputs(x0)
            for i, (sp, ap) in enumerate(pool):
                if F.classify_skeleton(list(_steps_of(ap)), dims, pos) is None:
                    candidates.append((i, sp, ap))
            if candidates:
                log(
                    f"  pid={pid}: dataset-index match empty; using "
                    f"{len(candidates)} live-feasible"
                )

        # Real-refine each candidate until one genuinely succeeds; render it.
        sampler = C._make_trajectory_sampler(cfg, em)
        n_fp = 0
        for idx, state_plan, action_plan in candidates:
            seed = C._refinement_seed(cfg.refinement_seed_rule, pid, idx)
            refiner = C._make_refiner(cfg, obs, sampler, seed)
            if hasattr(sampler, "clear"):
                sampler.clear()
            t0 = time.perf_counter()
            try:
                plan = refiner(x0, state_plan, action_plan, r_cap, bpg)
            except BaseException as exc:  # noqa: BLE001 — a refiner crash is a false positive
                plan = None
                log(
                    f"  pid={pid} skeleton {idx}: refiner raised "
                    f"{type(exc).__name__}: {exc}"
                )
            dt = time.perf_counter() - t0

            # Verify: a non-None plan means real MP realized the whole skeleton; double-check the
            # stored goal atoms actually hold in the final abstract state (false-positive guard).
            ok = plan is not None
            if ok:
                final_atoms = em.state_abstractor(plan.states[-1]).atoms
                ok = set(ep.goal_atoms).issubset(final_atoms)
            if not ok:
                n_fp += 1
                log(
                    f"  pid={pid} skeleton {idx}: analytic FALSE POSITIVE "
                    f"(real refine {'None' if plan is None else 'goal-miss'} in {dt:.0f}s)"
                )
                continue

            log(f"  pid={pid} skeleton {idx}: real-MP SUCCESS in {dt:.0f}s; rendering")
            frames = _render_plan(sim, plan, frame_skip)
            return frames, idx, n_fp

        return None
    finally:
        env.close()
        if sim is not None and hasattr(sim, "close"):
            sim.close()


def _render_plan(sim, plan, frame_skip: int) -> list:
    """Render the verified real-MP trajectory: set each recorded state and capture a frame."""
    frames = []
    for i, st in enumerate(plan.states):
        if i % frame_skip == 0:
            sim.set_state(st)
            frames.append(_render(sim))
    # ensure the final placed configuration is shown, then hold it
    sim.set_state(plan.states[-1])
    final = _render(sim)
    frames.extend([final] * 15)
    return frames


def _assert_dims_match(x0, ep, pid: int) -> None:
    """Cross-check reconstructed (width,height) per goal block against stored scene geometry."""
    stored = {}
    for o in ep.scene_geometry.objects:
        if o.name.startswith("obj_goal"):
            bx = o.boundary
            w = max(px for px, _ in bx) - min(px for px, _ in bx)
            stored[o.name] = (w, getattr(o, "height", None))
    for o in x0:
        if o.name.startswith("obj_goal"):
            w = 2 * x0.get(o, "half_extent_x")
            h = 2 * x0.get(o, "half_extent_z")
            sw, sh = stored.get(o.name, (None, None))
            if sw is not None:
                assert abs(w - sw) < 5e-3, (
                    f"pid {pid} {o.name}: width {w:.3f} != stored {sw:.3f} "
                    f"(scene reconstruction drift)"
                )
                if sh is not None:
                    assert abs(h - sh) < 5e-3, (
                        f"pid {pid} {o.name}: height {h:.3f} != stored {sh:.3f}"
                    )


# --------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", default="0,1,2,3")
    ap.add_argument("--n-per-stratum", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=18, help="sampling attempts per step")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    strata = [int(s) for s in args.strata.split(",")]
    t_start = time.time()

    def log(m: str) -> None:
        print(f"[{time.time() - t_start:6.0f}s] {m}", flush=True)

    _OUT.mkdir(parents=True, exist_ok=True)
    log(
        f"restock3d_v3 demos: strata={strata} n_per_stratum={args.n_per_stratum} "
        f"attempts={args.attempts} -> {_OUT}"
    )

    summary: dict[int, list[dict]] = {}
    for si, stratum in enumerate(strata):
        log(
            f"=== stratum {stratum} (n={_N_BY_STRATUM[stratum]}) "
            f"[{si + 1}/{len(strata)}] ==="
        )
        made = render_stratum(
            stratum,
            args.n_per_stratum,
            args.attempts,
            args.frame_skip,
            args.fps,
            log,
        )
        summary[stratum] = made
        done = sum(len(v) for v in summary.values())
        target = args.n_per_stratum * len(strata)
        if done:
            eta = (time.time() - t_start) / done * (target - done)
            log(f"  progress {done}/{target} videos; rough ETA {eta / 60:.0f} min")

    # One summary file per stratum (so parallel single-stratum runs never clobber a shared
    # file). Assemble a combined SUMMARY.md from these afterwards.
    for stratum in strata:
        lines = [
            f"# Restock3D-v3 demos — stratum {stratum} (n={_N_BY_STRATUM[stratum]})",
            "",
            "Real-MP-verified successful plans (kinder `BacktrackingRefiner` low-level "
            "trajectory). Analytic-success skeletons that fail real refinement are skipped "
            "as false positives.",
            "",
            "| pid | skeleton_idx | analytic FP skipped | frames | file |",
            "|---|---|---|---|---|",
        ]
        for r in summary.get(stratum, []):
            lines.append(
                f"| {r['pid']} | {r['skeleton_idx']} | {r['n_false_positives']} | "
                f"{r['frames']} | r{stratum}/{pathlib.Path(r['path']).name} |"
            )
        (_OUT / f"SUMMARY_r{stratum}.md").write_text("\n".join(lines) + "\n")
    log(
        f"DONE: {sum(len(v) for v in summary.values())} videos in "
        f"{(time.time() - t_start) / 60:.1f} min. Wrote per-stratum SUMMARY_r*.md"
    )


if __name__ == "__main__":
    main()
