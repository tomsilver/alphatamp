#!/usr/bin/env python
"""Standalone B6 DP-on-counts result for a SINGLE horizon ``h`` (CLI).

Runs one lookahead depth end-to-end on a split, prints a live per-episode ETA,
saves results incrementally, and pair-compares against B4 (``h=1``). Built for
running deep horizons in the background across parallel terminals **without
touching the analysis notebook** — e.g. one terminal for ``--h 6`` and another
for ``--h 7``.

Deep horizons need top-m pruning: the exact search costs ``O(K^{h+1})`` per
decision (h=4 exact ≈ 10 min on RT2D-n3), so ``h ≥ 5`` exact is intractable. Pass
``--m`` (e.g. 8) to prune the lookahead expansion to the top-m candidates at each
internal node — the root choice and ``h=1`` are never pruned, so the policy still
considers the full pool for what it actually attempts. Watch the printed ETA and
Ctrl-C if it is too slow (partial results are saved every ``--progress-every``
episodes).

Examples:
    # two terminals, in parallel:
    python experiments/spectre/dp_horizon_sweep.py --h 6 --m 8
    python experiments/spectre/dp_horizon_sweep.py --h 7 --m 8

    # exact (no pruning), only sensible for small h:
    python experiments/spectre/dp_horizon_sweep.py --h 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon

from alphatamp.approaches.spectre import dp_on_counts
from alphatamp.approaches.spectre.eda import (
    SkeletonKey,
    _build_dp_model,
    _fit_adaptive,
    _fit_refine_costs,
    _trainable_episodes,
    load_split_episodes,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="B6 DP-on-counts result for a single horizon h.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--h", type=int, required=True, help="lookahead depth (>= 1)")
    ap.add_argument(
        "--m",
        type=int,
        default=None,
        help="top-m lookahead pruning width; omit for exact. Use ~6-12 for h>=5.",
    )
    ap.add_argument("--env", default="routedtransport2d_n3_v1", help="env variant")
    ap.add_argument("--split", default="test", help="eval split (train/val/test)")
    ap.add_argument(
        "--budget", type=int, default=30, help="attempt budget (= pool cap)"
    )
    ap.add_argument(
        "--objective", default="attempts", choices=["attempts", "time"], help="cost"
    )
    ap.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "spectre"),
        help="root holding raw/<env>/<split>/episodes",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output .npz path (default: <data-root>/derived/dp_on_counts/...)",
    )
    ap.add_argument(
        "--progress-every", type=int, default=5, help="print + checkpoint every N eps"
    )
    return ap.parse_args()


def _rollout(
    model: dp_on_counts.DPModel,
    ep: EpisodeRecord,
    keys: Sequence[SkeletonKey],
    depth: int,
    budget: int,
    m: int | None,
) -> tuple[float, float, bool]:
    """Time-to-first-success for one episode under the depth-h B6 policy."""
    remaining = set(range(len(keys)))
    failed: tuple[SkeletonKey, ...] = ()
    steps = 0
    wall = 0.0
    att = budget + 1
    censored = True
    while remaining and steps < budget:
        chosen = dp_on_counts.select(model, remaining, failed, depth, m=m)
        steps += 1
        outcome = ep.outcomes[chosen]
        wall += outcome.refinement_wall_clock_s
        if outcome.outcome == "success":
            att = steps
            censored = False
            break
        failed = failed + (keys[chosen],)
        remaining.remove(chosen)
    return att, wall, censored


def main() -> None:
    """Run a single-horizon B6 sweep and write its result files."""
    args = _parse_args()
    if args.h < 1:
        raise SystemExit("--h must be >= 1")
    m_tag = "exact" if args.m is None else f"m{args.m}"
    if args.h >= 5 and args.m is None:
        print(
            f"WARNING: h={args.h} with exact search (no --m) is likely intractable "
            f"(~O(K^{args.h + 1}) per decision). Consider --m 8. Watch the ETA.",
            flush=True,
        )

    data_root = Path(args.data_root)
    raw = data_root / "raw" / args.env
    print(f"loading train + {args.split} from {raw} ...", flush=True)
    train = load_split_episodes(raw / "train")
    evals = load_split_episodes(raw / args.split)

    stats = _fit_adaptive(train)
    refine_costs = _fit_refine_costs(train) if args.objective == "time" else None
    score_cache: dict = {}
    q_cache: dict = {}
    delta_cache: dict = {}

    trainable = _trainable_episodes(evals)
    n = len(trainable)
    print(
        f"env={args.env} split={args.split} trainable={n} K_max={evals.k_max} "
        f"h={args.h} {m_tag} budget={args.budget} objective={args.objective}",
        flush=True,
    )

    out = (
        Path(args.out)
        if args.out
        else data_root
        / "derived"
        / "dp_on_counts"
        / f"{args.env}_{args.split}_h{args.h}_{m_tag}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    attempts = np.full(n, np.nan)
    walls = np.full(n, np.nan)
    censored = np.zeros(n, dtype=bool)
    attempts_h1 = np.full(n, np.nan)  # B4 reference (h=1), per problem
    problem_ids = np.zeros(n, dtype=np.int64)

    def checkpoint(done: int, complete: bool) -> None:
        np.savez(
            out,
            h=args.h,
            m=-1 if args.m is None else args.m,
            attempts=attempts,
            attempts_h1=attempts_h1,
            walls=walls,
            censored=censored,
            problem_ids=problem_ids,
            n_done=done,
            complete=complete,
        )

    t0 = time.perf_counter()
    for i, ep_idx in enumerate(trainable):
        ep = evals.episodes[ep_idx]
        keys = evals.skeleton_keys[ep_idx]
        model = _build_dp_model(
            stats, keys, args.objective, refine_costs, score_cache, q_cache, delta_cache
        )
        problem_ids[i] = ep.provenance.problem_id
        # h=1 reference (B4) — cheap, no expansion.
        attempts_h1[i] = _rollout(model, ep, keys, 1, args.budget, None)[0]
        att, wall, cens = _rollout(model, ep, keys, args.h, args.budget, args.m)
        attempts[i], walls[i], censored[i] = att, wall, cens
        done = i + 1
        if done % args.progress_every == 0 or done <= 2 or done == n:
            el = time.perf_counter() - t0
            eta = el / done * (n - done)
            print(
                f"h={args.h} {m_tag} ep={done}/{n} "
                f"mean_so_far={np.nanmean(attempts[:done]):.3f} "
                f"elapsed={el:.0f}s rate={el / done:.2f}s/ep ETA={eta:.0f}s",
                flush=True,
            )
            checkpoint(done, complete=False)
    checkpoint(n, complete=True)

    a, a1 = attempts, attempts_h1
    d = a1 - a  # positive ⇒ this horizon used fewer attempts than B4
    summary = {
        "env": args.env,
        "split": args.split,
        "h": args.h,
        "m": args.m,
        "budget": args.budget,
        "objective": args.objective,
        "n": int(n),
        "mean_attempts": float(a.mean()),
        "sd_attempts": float(a.std()),
        "mean_wall_clock_s": float(walls.mean()),
        "censoring_rate": float(censored.mean()),
        "h1_mean_attempts": float(a1.mean()),
        "vs_h1_delta_mean": float(d.mean()),
        "vs_h1_win_tie_loss": [
            int((d > 0).sum()),
            int((d == 0).sum()),
            int((d < 0).sum()),
        ],
        "wall_clock_total_s": round(time.perf_counter() - t0, 1),
    }
    summary["vs_h1_wilcoxon_p"] = (
        float(wilcoxon(a1, a).pvalue) if np.any(d != 0) else None
    )

    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== RESULT ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nsaved: {out}\n       {out.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
