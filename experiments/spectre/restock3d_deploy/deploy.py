"""Live SPECTRE-adaptive planning on a hand-specified Restock3D-v3 scene, for TidyBot.

This is the proof-of-concept deployment driver. Unlike the evaluation pipeline -- where
the candidate pool and every per-candidate refinement outcome are collected offline and
the ranker only *reads* them -- here everything runs LIVE:

  scene.yaml -> build the sim + v3 models -> draw the candidate skeleton pool ->
    loop: rank the pool with the trained SPECTRE checkpoint conditioned on the failures
          so far -> real-refine the top-ranked skeleton (kinder BacktrackingRefiner, real
          motion planning) -> on success stop; on failure record it and re-rank.

The winning refined plan is exported (``robot_export``) as an absolute base+joint
trajectory (directly replayable on the TidyBot -- the sim robot IS a tidybot-kinova) plus
per-operator semantic waypoints. Progress is printed after every attempt.

Everything the run needs is bundled in this folder: the checkpoint (``checkpoint/``), the
input scenes (``scenes/``), and the outputs (``outputs/``). With the repo venv active, a
bare run uses the bundled ``scenes/demo6`` example and writes to ``outputs/demo6``::

    python deploy.py                          # from inside this folder
    python deploy.py --scene scenes/myscene --render

Budgets (K_max / per-candidate refinement cap) default to the restock3d_v3_real
collection values for the scene's object count, clamped to the nearest trained stratum;
override with ``--k-max`` / ``--refinement-timeout``. Checkpoint, vocab, scene and output
directory all default to this folder and are overridable with ``--ckpt`` / ``--vocab`` /
``--scene`` / ``--out``.
"""

from __future__ import annotations

import os
import sys

# Pin the hash seed and re-exec: the candidate pool's enumeration order is
# PYTHONHASHSEED-dependent, so pinning it makes the pool (and the plan) deterministic and
# reproducible run to run. A scene is solvable only if its pool holds a feasible skeleton;
# seed 0 keeps the bundled demo6 solvable on the first attempt.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached). Mirrors --
# ``restock3d_v3_demos.py``; required because live refinement runs real motion planning.
import glob
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
import itertools
import time
from pathlib import Path

# deploy_scene and robot_export are sibling modules in this folder. deploy.py runs as a
# script, so its own directory is on sys.path and they import directly.
import deploy_scene as DS
import numpy as np
import robot_export
import torch
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractGoal

from alphatamp.approaches.spectre import collect as C
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.dataset import (
    atom_emission,
    build_example,
    collate,
    pointset_emission,
)
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.envs.restock3d.scene_geometry import (
    build_scene_geometry,
)
from alphatamp.approaches.spectre.inference import load_checkpoint
from alphatamp.approaches.spectre.schema import (
    EpisodeRecord,
    OutcomeRecord,
    ProvenanceBlock,
    SkeletonRecord,
    SummaryBlock,
)
from alphatamp.approaches.spectre.vocab import Vocab

_HERE = Path(__file__).resolve().parent
_TRIED = -1e9
_PID = 0  # a fixed synthetic problem id for the deterministic refinement seed


def _ops_str(action_plan) -> str:
    return " -> ".join(
        f"{op.name}({', '.join(p.name for p in op.parameters)})" for op in action_plan
    )


def _attribution(refiner_metadata: dict) -> str:
    """A short human-readable summary of the harvested failure dict.

    A culprit-bearing record is F2 crowding; a culprit-free failure on a ``place_*`` step
    is F3 (the block is too tall for the section, or the arm hits the ceiling board). The
    ``[proven]`` tag means the step exhausted its samples within the budget, so it is a
    real dead end (the signal the adaptive re-ranker uses to avoid repeating it).
    """
    failures = refiner_metadata.get("failures") if refiner_metadata else None
    if not failures:
        return "no attribution (budget/other)"
    f = failures[0]
    culprits = f.get("culprits") or []
    schema = str(f.get("schema"))
    if culprits:
        fam = f"F2 crowding (blocked by {culprits})"
    elif schema.startswith("place"):
        fam = "F3 (block too tall for the section / ceiling collision)"
    else:
        fam = "abstract-state deviation"
    proven = " [proven]" if f.get("exhausted") and not f.get("budget_exhausted") else ""
    return f"{fam} at step {f.get('step_index')} ({schema}){proven}"


def _refine_capture(cfg, sampler, bpg, x0, state_plan, action_plan, seed) -> tuple:
    """One real refinement that ALSO returns the plan on success.

    Mirrors ``collect._real_refine_candidate`` (which discards the plan) so the failure-
    metadata harvest is identical, but keeps the ``Plan`` object we need to
    export/execute.
    """
    refiner = C._make_refiner(cfg, None, sampler, seed)
    if hasattr(sampler, "clear"):
        sampler.clear()
    refiner_metadata: dict[str, object] = {}
    error_info = None
    stuck = None
    plan = None
    start = time.perf_counter()
    try:
        plan = refiner(x0, state_plan, action_plan, cfg.refinement_timeout_s, bpg)
        outcome = "success" if plan is not None else "fail"
    except (
        BaseException
    ) as exc:  # noqa: BLE001 -- a refiner crash is a failed candidate
        outcome = "error"
        error_info = {"cls": type(exc).__name__, "msg": str(exc)}
    fm_fn = C._failure_metadata_fn(cfg.model_name) if outcome == "fail" else None
    if fm_fn is not None:
        failures = fm_fn(
            sampler,
            action_plan,
            cfg.num_sampling_attempts_per_step,
            budget_exhausted=(time.perf_counter() - start >= cfg.refinement_timeout_s),
        )
        if failures:
            refiner_metadata["failures"] = failures
            stuck = int(failures[0]["step_index"])  # type: ignore[call-overload]
    return (
        plan,
        outcome,
        time.perf_counter() - start,
        refiner_metadata,
        stuck,
        error_info,
    )


def _placeholder_outcome(idx: int, cfg) -> OutcomeRecord:
    return OutcomeRecord(
        skeleton_idx=idx,
        outcome="fail",
        refinement_wall_clock_s=0.0,
        refinement_seed=C._refinement_seed(cfg.refinement_seed_rule, _PID, idx),
        refiner_metadata={},
    )


def _summary(outcomes, k_cap) -> SummaryBlock:
    return SummaryBlock(
        num_skeletons=len(outcomes),
        num_success=sum(1 for o in outcomes if o.outcome == "success"),
        num_fail=sum(1 for o in outcomes if o.outcome == "fail"),
        num_error=sum(1 for o in outcomes if o.outcome == "error"),
        first_success_idx=next(
            (i for i, o in enumerate(outcomes) if o.outcome == "success"), None
        ),
        total_wall_clock_s=sum(o.refinement_wall_clock_s for o in outcomes),
        pool_truncated=len(outcomes) >= k_cap,
    )


def _render_plan(sim, plan, out_path: Path, frame_skip: int = 2, fps: int = 20) -> None:
    import imageio.v2 as iio
    from pybullet_helpers.camera import capture_image

    def _frame() -> np.ndarray:
        return capture_image(
            sim.physics_client_id,
            image_width=640,
            image_height=480,
            **sim.config.get_camera_kwargs(),
        )

    frames = []
    for i, st in enumerate(plan.states):
        if i % frame_skip == 0:
            sim.set_state(st)
            frames.append(_frame())
    sim.set_state(plan.states[-1])
    frames.extend([_frame()] * 15)
    iio.mimsave(out_path, frames, fps=fps, macro_block_size=16)  # type: ignore[arg-type]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene",
        default=str(_HERE / "scenes" / "demo6"),
        help="scene dir or scene.yaml (default: the bundled scenes/demo6 example)",
    )
    ap.add_argument(
        "--ckpt",
        default=str(_HERE / "checkpoint" / "best.pt"),
        help="SPECTRE checkpoint (default: the bundled checkpoint/best.pt)",
    )
    ap.add_argument(
        "--vocab",
        default=str(_HERE / "checkpoint" / "train_vocab.json"),
        help="vocab json (default: the bundled checkpoint/train_vocab.json)",
    )
    ap.add_argument(
        "--k-max", type=int, default=None, help="override pool cap (else by n)"
    )
    ap.add_argument(
        "--refinement-timeout",
        type=float,
        default=None,
        help="override per-candidate refinement cap in seconds (else by n)",
    )
    ap.add_argument("--samples-per-step", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--render", action="store_true", help="also write an mp4 of the plan"
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output dir (default: <this folder>/outputs/<scene name>)",
    )
    args = ap.parse_args()

    t0 = time.time()

    def log(m: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    scene_path = Path(args.scene)
    scene_name = scene_path.name if scene_path.is_dir() else scene_path.parent.name
    out_dir = Path(args.out) if args.out else (_HERE / "outputs" / scene_name)

    # --- 1. scene ---------------------------------------------------------------
    scene = DS.load_scene(scene_path)
    log(f"scene: {scene.n} objects (stratum {scene.stratum})")
    for w in DS.validate_scene(scene):
        log(f"  WARNING: {w}")

    # --- 2. env + models (hand scene baked into the sim at construction) ---------
    bundle = DS.build_deploy_models(scene)
    em = bundle.models
    sim = bundle.sim
    x0 = DS.make_x0(sim)
    s0 = em.state_abstractor(x0)
    goal = em.goal_deriver(x0)
    assert isinstance(goal, RelationalAbstractGoal)
    scene_geometry = build_scene_geometry(x0)

    # --- 3. budgets -------------------------------------------------------------
    k_def, r_def, s_def = DS.budget_for_n(scene.n)
    k_max = args.k_max if args.k_max is not None else k_def
    r_cap = args.refinement_timeout if args.refinement_timeout is not None else r_def
    samples = args.samples_per_step if args.samples_per_step is not None else s_def
    cfg = CollectionConfig(
        env_id=f"spectre/Restock3Dv3-r{scene.stratum}-v0",
        env_variant="restock3d_v3_real",
        model_name="restock3d_v3",
        model_kwargs={"stratum": scene.stratum},
        split="test",
        num_problems=1,
        problem_seed_start=0,
        problem_seed_end=1,
        K_max=k_max,
        abstract_plan_timeout_s=120.0,
        refinement_timeout_s=r_cap,
        num_sampling_attempts_per_step=samples,
        max_trajectory_steps=500,
        plan_generator="closed_form",
        refiner_mode="real",
    )
    log(f"budgets: K_max={k_max} r_cap={r_cap}s samples/step={samples}")

    # --- 4. candidate pool (geometry-guided generator, capped at K_max) ----------
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    bpg.add_abstract_state_node(s0)
    bpg.add_state_node(x0)
    bpg.add_state_abstractor_edge(x0, s0)
    gen = C._make_plan_generator(cfg, em, None, _PID, x0)
    pool = list(
        itertools.islice(gen(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg), k_max)
    )
    log(f"pool: {len(pool)} candidate skeletons")
    if not pool:
        log("EMPTY POOL -- the generator found no goal-reaching skeleton. Aborting.")
        return

    # --- 5. mutable EpisodeRecord scaffold (placeholders; real outcomes live) ----
    skeleton_records = [
        SkeletonRecord(
            skeleton_idx=i,
            operator_seq=tuple(action_plan),
            final_abstract_state=state_plan[-1],
        )
        for i, (state_plan, action_plan) in enumerate(pool)
    ]
    outcomes = [_placeholder_outcome(i, cfg) for i in range(len(pool))]
    provenance = ProvenanceBlock(
        problem_id=_PID,
        env_id=cfg.env_id,
        env_variant=cfg.env_variant,
        split=cfg.split,
        config_hash=cfg.config_hash,
        problem_seed=_PID,
        git_sha=cfg.git_sha,
        collection_timestamp="",
        package_versions=dict(cfg.package_versions),
        gen_params={"stratum": scene.stratum, "split": cfg.split},
    )
    object_registry = C._collect_all_objects(s0, pool, goal.atoms)

    def _episode() -> EpisodeRecord:
        return EpisodeRecord(
            provenance=provenance,
            initial_abstract_state=s0,
            goal_atoms=frozenset(goal.atoms),
            object_registry=object_registry,
            skeleton_pool=tuple(skeleton_records),
            outcomes=tuple(outcomes),
            summary=_summary(outcomes, k_max),
            scene_geometry=scene_geometry,
        )

    # --- 6. checkpoint + tensorizer emission (all read off the checkpoint) -------
    vocab = Vocab.from_json(Path(args.vocab))
    model, deploy = load_checkpoint(args.ckpt, vocab, device=args.device)
    spec = spec_for(cfg.env_variant)
    scene_3d = getattr(model.cfg, "point_dim", 2) == 3
    ps_feats, ps_pca, ps_k = pointset_emission(model.cfg, scene_3d)
    emit_init, emit_goal = atom_emission(model.cfg)
    sampler = C._make_trajectory_sampler(cfg, em)
    log(
        f"model: {args.ckpt} (scene_3d={scene_3d}, atoms={emit_init}/{emit_goal}, "
        f"repeat_feats={deploy.get('repeat_feats')})"
    )

    # --- 7. LIVE rank -> refine -> re-rank ---------------------------------------
    tried: list[int] = []
    winner = None  # (pick, plan, action_plan)
    # Pure-geometry inputs for the analytic pre-filter below (constant per scene).
    analytic_dims, analytic_pos = C._restock3d_analytic_inputs(x0)
    while len(tried) < len(pool):
        example, records = build_example(
            _episode(),
            vocab,
            rng=None,
            evidence=True,
            context_f=frozenset(tried),
            augment_tags=False,
            spec=spec,
            scene_3d=scene_3d,
            pointset_feats=ps_feats,
            use_pca_feats=ps_pca,
            edgeconv_k=ps_k,
            emit_init_atoms=emit_init,
            emit_goal_atoms=emit_goal,
            **deploy,
        )
        batch = collate(
            [example],
            max_arity=vocab.max_operator_arity,
            records=[records],
            max_pred_arity=vocab.max_predicate_arity,
        ).to(args.device)
        with torch.no_grad():
            logits, _ = model(batch)
        raw = logits[0].detach().cpu().numpy().astype(float)
        if tried:
            raw[tried] = _TRIED
        pick = int(np.argmax(raw))
        tried.append(pick)

        state_plan, action_plan = pool[pick]
        log(f"attempt {len(tried)}: skeleton #{pick} | {_ops_str(action_plan)}")

        seed = C._refinement_seed(cfg.refinement_seed_rule, _PID, pick)

        # Analytic pre-filter: classify the skeleton with the same pure-geometry
        # classifier the collection's hybrid prune uses. A provably infeasible
        # candidate is recorded as a failure (with the first-violation dict, so the
        # adaptive re-ranker conditions on it exactly as on a real one) without
        # paying real refinement -- a ranked-order misfire costs milliseconds
        # instead of the full per-candidate budget.
        fm = C._restock3d_classify(
            action_plan, analytic_dims, analytic_pos, cfg.num_sampling_attempts_per_step
        )
        if fm is not None:
            outcomes[pick] = OutcomeRecord(
                skeleton_idx=pick,
                outcome="fail",
                refinement_wall_clock_s=0.0,
                refinement_seed=seed,
                stuck_step_index=fm.get("step_index"),
                error_info=None,
                refiner_metadata={"failures": [fm]},
                label_source="analytic",
            )
            log(f"  -> skipped, analytically infeasible: {_attribution({'failures': [fm]})}")
            continue

        plan, outcome, wall, meta, stuck, err = _refine_capture(
            cfg, sampler, bpg, x0, state_plan, action_plan, seed
        )
        ok = outcome == "success" and plan is not None
        if ok:
            final_atoms = em.state_abstractor(plan.states[-1]).atoms
            ok = set(goal.atoms).issubset(final_atoms)
        outcomes[pick] = OutcomeRecord(
            skeleton_idx=pick,
            outcome=("success" if ok else outcome),  # type: ignore[arg-type]
            refinement_wall_clock_s=wall,
            refinement_seed=seed,
            stuck_step_index=stuck,
            error_info=err,
            refiner_metadata=meta,
            label_source="real",
        )
        if ok:
            log(f"  -> SUCCESS in {wall:.1f}s (after {len(tried)} attempts)")
            winner = (pick, plan, action_plan)
            break
        if outcome == "error":
            log(f"  -> ERROR in {wall:.1f}s: {err}")
        else:
            log(f"  -> fail in {wall:.1f}s: {_attribution(meta)}")

    # --- 8. export ---------------------------------------------------------------
    if winner is None:
        log(
            f"NO PLAN FOUND after {len(tried)} attempts. Ranked order tried: {tried}. "
            f"Consider raising --k-max / --refinement-timeout or easing the scene."
        )
        return
    pick, plan, action_plan = winner
    meta = {
        "scene": str(scene_path),
        "checkpoint": args.ckpt,
        "winning_skeleton_idx": pick,
        "attempts_to_first_success": len(tried),
        "attempt_order": tried,
        "K_max": k_max,
        "r_cap_s": r_cap,
    }
    paths = robot_export.export_plan(plan, action_plan, sim, out_dir, meta=meta)
    log(f"exported: {paths['level_a']}")
    log(f"          {paths['level_b_json']}")
    log(f"          {paths['level_b_npz']}")
    if args.render:
        mp4 = Path(out_dir) / "plan.mp4"
        _render_plan(sim, plan, mp4)
        log(f"          {mp4}")
    log(f"DONE in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
