"""Hand-rule stack P4 gate (proposal §9/§10.4): does the ZERO-parameter proof-demotion
cut rollout-FP over its own static base?

The hand-rule stack = a base ordering (default planner order) + the §5 proof-demotion
filter, no learned parameters. For each test problem we harvest a `blocked-at-contents`
proof from every *failed* attempt whose removal still leaves the target blocked (a
post-mortem grasp check), demote the removal-monotone subsets, and compare rollout-FP
with vs without proof demotion. P4 passes when the paired FP reduction on strata >= 2 has
a bootstrap CI excluding zero.

The grasp check is computed by reconstructing the obstacle set from the *stored*
``scene_geometry`` (``spectre_geometry.target_blocked_after_removing``) — the same poses
the refiner labeled — **not** by regenerating the scene from its seed. Regeneration has to
infer the generation parameters and its rejection sampling diverges under any mismatch,
producing a geometrically-different scene whose proofs contradict the collected labels;
that is what made an earlier version of this gate report a spurious *negative* ΔFP (a sound
proof-demotion can only help). See ``docs/decisions.md`` 2026-07-19.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _one_episode(path_str: str):
    """Rollout base vs base+proof-demotion over one episode, reconstructing grasp proofs
    from the stored geometry.

    Returns (base_fp, handrule_fp, stratum) or None.
    """
    from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (
        target_blocked_after_removing,
    )
    from alphatamp.approaches.spectre.io import load_episode
    from alphatamp.approaches.spectre.proof_demotion import ProofState, demote

    ep = load_episode(Path(path_str))
    if ep.scene_geometry is None or ep.summary.num_success < 1:
        return None

    def staged(sk):
        return frozenset(
            op.parameters[0].name for op in sk.operator_seq if op.name == "place-buffer"
        )

    subsets = [staged(s) for s in ep.skeleton_pool]
    feasible = np.array([o.outcome == "success" for o in ep.outcomes])
    stratum = min(
        (len(subsets[i]) for i in range(len(subsets)) if feasible[i]), default=-1
    )

    blocked = {}  # subset -> blocked-after-removing (memoized per unique subset)

    def is_blocked(fs) -> bool:
        if fs not in blocked:
            blocked[fs] = target_blocked_after_removing(ep.scene_geometry, fs)
        return blocked[fs]

    base_order = list(range(len(subsets)))  # default planner order

    def rollout(use_proofs: bool) -> int:
        state = ProofState(subsets=subsets)
        remaining = list(base_order)
        fp = 0
        while remaining:
            if use_proofs:
                remaining = demote(remaining, state.dead)
            idx = remaining.pop(0)
            if feasible[idx]:
                return fp
            fp += 1
            if use_proofs and is_blocked(subsets[idx]):
                state.observe_failure(idx, blocked=True, pack_impossible=False)
        return fp

    return (rollout(False), rollout(True), int(stratum))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hand-rule stack P4 gate")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--min-stratum", type=int, default=2)
    args = ap.parse_args(argv)
    from alphatamp.approaches.spectre.io import list_episodes

    paths = [
        str(p) for p in list_episodes(Path(args.data_root) / "raw" / args.env / "test")
    ]
    print(
        f"# hand-rule P4: {len(paths)} test episodes, workers={args.workers}",
        flush=True,
    )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_one_episode, paths, chunksize=2)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 40 == 0:
                print(
                    f"  ... {i + 1}/{len(paths)} done ({len(rows)} usable)", flush=True
                )

    arr = np.array(rows, dtype=float)  # (n, 3): base_fp, handrule_fp, stratum
    for lo, tag in [(0, "ALL"), (args.min_stratum, f"strata>={args.min_stratum}")]:
        m = arr[:, 2] >= lo
        base, hand = arr[m, 0], arr[m, 1]
        diff = base - hand  # positive = hand-rule cuts FP
        rng = np.random.default_rng(0)
        boot = np.array(
            [rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)]
        )
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        print(
            f"# {tag}: n={m.sum()} base_FP={base.mean():.2f} handrule_FP={hand.mean():.2f} "
            f"ΔFP={diff.mean():.2f} CI=({lo_ci:.2f},{hi_ci:.2f}) "
            f"P4={'PASS' if lo_ci > 0 else 'no'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
