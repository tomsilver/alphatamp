"""Render SPECTRE-adaptive refinement demo videos on ``dd2d_v4gen_shapeonly_sz07``.

For a few problems per difficulty stratum, this writes one mp4 **per refinement attempt**
the SPECTRE-adaptive rollout makes -- in the order it tried candidates, up to and
including the first success. It makes the ranker's behaviour legible: try a short/naive
plan, fail, then try staging plans until one refines.

Orchestration over existing pieces (nothing is re-derived):

* ``compare.load_adaptive_trace(cache, "spectre3_adaptive", pid, seed).order`` IS the
  realized attempt sequence of pool-candidate indices, already truncated at the first
  success. Index ``i`` maps 1:1 to ``episode.skeleton_pool[i]`` (schema invariant I2).
* ``reconstruct_scene(episode.scene_geometry)`` rebuilds a live ``DrawerScene``;
  ``Skeleton.from_action_tuples(...)`` rebuilds the candidate skeleton.
* ``DD2DRefiner(**refiner_params).refine(sk, scene, seed=outcome.refinement_seed)``
  reproduces the stored label bit-for-bit and returns the animatable ``bound_plan``.
* ``render_episode(scene, bound_plan, feasible, failure_action, out)`` writes the mp4.

Output tree (default under ``envs/dd2d/out_dd2d/sz07_adaptive_demos``)::

    s0/problem_6000000/Attempt_001.mp4       # unblocked target -> 1 attempt (success)
    s2/problem_6500000/Attempt_001.mp4       # fail: retrieve, unstaged
                       Attempt_002.mp4       # fail: stage 3 -> retrieve
                       Attempt_003.mp4       # success
    manifest.json

Usage::

    python experiments/spectre/sz07_adaptive_demos.py            # 5 problems/stratum
    python experiments/spectre/sz07_adaptive_demos.py --resume   # skip existing mp4s
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

from alphatamp.approaches.spectre import compare, eda
from alphatamp.approaches.spectre.envs.dd2d.dd2d.refine import DD2DRefiner
from alphatamp.approaches.spectre.envs.dd2d.dd2d.render import render_episode
from alphatamp.approaches.spectre.envs.dd2d.skeleton import Skeleton
from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import reconstruct_scene

REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = (
    REPO / "src/alphatamp/approaches/spectre/envs/dd2d/out_dd2d/sz07_adaptive_demos"
)
# refiner knobs for the sz07 collection; stored provenance overrides these per episode
_REFINER_FALLBACK = {
    "budget": None,
    "retry_cap": 10,
    "samples_per_step": 15,
    "time_budget": 20.0,
}


def _skeleton_of(rec) -> Skeleton:
    """A live :class:`Skeleton` from a stored ``SkeletonRecord`` (op + arg names)."""
    return Skeleton.from_action_tuples(
        [[op.name, *(p.name for p in op.parameters)] for op in rec.operator_seq]
    )


def _plan_label(rec) -> str:
    """One-line human plan, e.g. ``stage[item_2,item_1] -> retrieve item_10``."""
    staged = [
        p.name
        for op in rec.operator_seq
        if op.name == "place-buffer"
        for p in op.parameters
    ]
    tgt = next(
        (
            p.name
            for op in rec.operator_seq
            if op.name == "retrieve"
            for p in op.parameters
        ),
        "?",
    )
    head = f"stage[{','.join(staged)}] -> " if staged else ""
    return f"{head}retrieve {tgt}"


def _attempts_for(episode, trace) -> list[int]:
    """Candidate indices SPECTRE-adaptive tried, in order (fails then the success).

    ``trace.order`` already ends at the first success; this validates that, returns it.
    A censored rollout (never solved -- not expected on sz07) is returned as-is, warned.
    """
    order = list(trace.order)
    if not order:
        return order
    if episode.outcomes[order[-1]].outcome != "success":
        print(
            f"  !! pid {episode.provenance.problem_id}: rollout censored "
            f"(last outcome={episode.outcomes[order[-1]].outcome}); rendering anyway"
        )
    return order


def _pick_pids(
    ep_by_pid: Mapping[int, object], n_per_stratum: int
) -> dict[int, list[int]]:
    """First ``n_per_stratum`` sorted pids per stratum (gaps -> not band+0..4)."""
    by_stratum: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for pid in sorted(ep_by_pid):
        by_stratum[compare.stratum_of(pid)].append(pid)
    return {s: pids[:n_per_stratum] for s, pids in by_stratum.items()}


def main(argv: list[str] | None = None) -> None:
    """Render the per-attempt demo videos and write a manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-variant", default="dd2d_v4gen_shapeonly_sz07")
    ap.add_argument("--out-root", default=str(_DEFAULT_OUT))
    ap.add_argument("--n-per-stratum", type=int, default=5)
    ap.add_argument(
        "--spectre-seed", type=int, default=0, help="adaptive checkpoint seed"
    )
    ap.add_argument("--format", default="mp4", choices=("mp4", "gif"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="skip mp4s already on disk")
    args = ap.parse_args(argv)

    variant = args.env_variant
    out_root = Path(args.out_root)
    test_dir = REPO / "data" / "spectre" / "raw" / variant / "test"
    cache_dir = REPO / "data" / "spectre" / "derived" / variant / "compare_cache"

    ep_by_pid = {
        int(e.provenance.problem_id): e
        for e in eda.load_split_episodes(test_dir).episodes
    }
    picks = _pick_pids(ep_by_pid, args.n_per_stratum)
    total_problems = sum(len(v) for v in picks.values())
    print(
        f"env={variant}  problems={total_problems} "
        f"({args.n_per_stratum}/stratum)  seed={args.spectre_seed}  -> {out_root}"
    )

    manifest: list[dict] = []
    t0 = time.time()
    done = 0
    for stratum in sorted(picks):
        for pid in picks[stratum]:
            episode = ep_by_pid[pid]
            assert episode.scene_geometry is not None  # sz07 episodes carry geometry
            trace = compare.load_adaptive_trace(
                cache_dir, "spectre3_adaptive", pid, seed=args.spectre_seed
            )
            if trace is None:
                print(f"  !! pid {pid}: no adaptive trace cached; skipping")
                continue
            attempts = _attempts_for(episode, trace)
            refiner_params = (episode.provenance.gen_params or {}).get(
                "refiner_params", _REFINER_FALLBACK
            )
            pdir = out_root / f"s{stratum}" / f"problem_{pid}"
            pdir.mkdir(parents=True, exist_ok=True)

            attempt_recs = []
            for rank, idx in enumerate(attempts):
                rec = episode.skeleton_pool[idx]
                outcome = episode.outcomes[idx]
                vid = pdir / f"Attempt_{rank + 1:03d}.{args.format}"
                if not (args.resume and vid.exists()):
                    scene = reconstruct_scene(episode.scene_geometry)
                    res = DD2DRefiner(**refiner_params).refine(
                        _skeleton_of(rec), scene, seed=outcome.refinement_seed
                    )
                    if res.feasible != (outcome.outcome == "success"):
                        print(
                            f"  !! pid {pid} attempt {rank + 1}: re-run "
                            f"feasible={res.feasible} != stored {outcome.outcome}"
                        )
                    render_episode(
                        scene,
                        res.bound_plan,
                        res.feasible,
                        res.failure_action,
                        str(vid),
                        fmt=args.format,
                        fps=args.fps,
                    )
                attempt_recs.append(
                    {
                        "attempt": rank + 1,
                        "candidate_idx": idx,
                        "outcome": outcome.outcome,
                        "plan": _plan_label(rec),
                        "video": str(vid.relative_to(out_root)),
                    }
                )

            manifest.append(
                {
                    "problem_id": pid,
                    "stratum": stratum,
                    "fp": trace.fp,
                    "n_attempts": len(attempts),
                    "attempts": attempt_recs,
                }
            )
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total_problems - done)
            print(
                f"[{done}/{total_problems}] s{stratum} pid {pid}: "
                f"{len(attempts)} attempt(s) (fp={trace.fp:g}) | "
                f"{elapsed:.0f}s elapsed, ETA ~{eta:.0f}s"
            )

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    n_videos = sum(m["n_attempts"] for m in manifest)
    print(f"done: {n_videos} videos across {total_problems} problems -> {out_root}")


if __name__ == "__main__":
    main()
