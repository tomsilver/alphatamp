"""Gate G0 sweep (v2.2.1 §10.2): does DD2D exercise subset-coupled feasibility?

For each buffer-tightness λ, generate DD2D scenes, label their candidate pools with the
§8.4 certificate (so negatives are trustworthy), and score two cheap probes — slack
ordering and a pairwise-features GBDT — at predicting per-candidate feasibility, alongside
the oracle solve rate. Both overall and **within-length** (size-conditional) AUROC are
reported; the within-length AUROC is the thesis-relevant signal because DD2D feasibility
is length/count-dominated, so an overall AUROC is inflated by |S|. λ* is the loosest λ
where the cheap stats degrade *within-length* (GBDT within-length AUROC < --degrade-thresh)
yet the oracle still solves (solve rate ≥ --oracle-thresh). If no such λ exists, G0 fails
and the honest next step is benchmark work, not model work (pre-registered off-ramp).

Scene generation + labeling (the expensive part) runs worker-side in a process pool;
workers return only lightweight feature rows. Example::

    python experiments/spectre/spectre_g0.py --lams 0.8,0.65,0.5,0.4 \
        --n-train 60 --n-val 30 --workers 12
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from alphatamp.approaches.spectre.g0 import (
    FEATURE_NAMES,
    LabeledCandidates,
    buffer_slack,
    choose_lambda_star,
    evaluate_g0_point,
    feature_vector,
)

# Disjoint seed bands per (λ, split) so train/val never share an instance.
_BAND = 1_000_000


@dataclass
class _SceneResult:
    rows: list  # list[np.ndarray] feature vectors (confident candidates only)
    ys: list  # parallel 0/1 labels
    slacks: list
    sizes: list  # |S| per confident candidate
    solved: bool  # >= 1 feasible candidate
    n_marginal: int
    n_total: int
    ok: bool


def _one_scene(args) -> _SceneResult:
    """Generate + label one scene worker-side; return lightweight feature rows."""
    lam, seed = args
    from alphatamp.approaches.spectre.envs.dd2d.dd2d import label as L
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.enumerate import (
        enumerate_candidates,
    )
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import (
        generate_dd2d_problem,
    )

    try:
        prob = generate_dd2d_problem(
            lam=lam,
            seed=seed,
            margin=1.0,
            split="train",
            n_items=11,
            crowd=5,
            require_subset=True,
            min_subset=2,
            certify=True,
            time_budget=4.0,
        )
    except Exception:
        return _SceneResult([], [], [], [], False, 0, 0, ok=False)
    scene = prob.scene
    cands = enumerate_candidates(scene)
    L.label_all(scene, cands, seed=seed, use_certificate=True)
    rows, ys, slacks, sizes = [], [], [], []
    solved = False
    n_marginal = 0
    for c in cands:
        label = c.meta.get("label")
        if label == "marginal":
            n_marginal += 1
            continue
        feasible = label == "feasible"
        solved = solved or feasible
        rows.append(feature_vector(scene, c.subset))
        ys.append(1 if feasible else 0)
        slacks.append(buffer_slack(scene, c.subset))
        sizes.append(len(c.subset))
    return _SceneResult(
        rows, ys, slacks, sizes, solved, n_marginal, len(cands), ok=True
    )


def _aggregate(results: list[_SceneResult]) -> LabeledCandidates:
    rows, ys, slacks, sizes = [], [], [], []
    n_scenes = n_solved = n_marginal = n_total = 0
    for r in results:
        if not r.ok:
            continue
        n_scenes += 1
        n_solved += int(r.solved)
        n_marginal += r.n_marginal
        n_total += r.n_total
        rows.extend(r.rows)
        ys.extend(r.ys)
        slacks.extend(r.slacks)
        sizes.extend(r.sizes)
    X = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.zeros((0, len(FEATURE_NAMES)))
    )
    return LabeledCandidates(
        X=X,
        y=np.asarray(ys, dtype=np.int64),
        slack=np.asarray(slacks, dtype=np.float64),
        sizes=np.asarray(sizes, dtype=np.int64),
        n_scenes=n_scenes,
        n_oracle_solved=n_solved,
        n_marginal=n_marginal,
        n_total_candidates=n_total,
    )


def _run_split(lam: float, band_lo: int, n: int, workers: int) -> LabeledCandidates:
    tasks = [(lam, band_lo + i) for i in range(n * 4)]  # overshoot; some gens fail
    results: list[_SceneResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_one_scene, tasks, chunksize=1):
            if r.ok:
                results.append(r)
            if len(results) >= n:
                break
    return _aggregate(results)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DD2D Gate G0 λ-sweep")
    # Default sweep stays inside DD2D's designed operating range (0.7-0.95, the loose /
    # naturalistic regime). Tighter λ is off-design (3-subsets stop packing → stratum-3
    # ungenerable) and λ* is constrained to --op-lo..--op-hi regardless.
    ap.add_argument("--lams", default="0.7,0.8,0.9,0.95")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--degrade-thresh", type=float, default=0.65)
    ap.add_argument("--oracle-thresh", type=float, default=0.5)
    ap.add_argument("--op-lo", type=float, default=0.7)
    ap.add_argument("--op-hi", type=float, default=0.95)
    args = ap.parse_args(argv)

    lams = [float(x) for x in args.lams.split(",")]
    print(
        f"# G0 sweep λ={lams} n_train={args.n_train} n_val={args.n_val} "
        f"workers={args.workers}",
        flush=True,
    )
    hdr = (
        f"{'λ':>6} {'scenes':>7} {'oracle':>7} {'feas%':>6} {'marg%':>6} "
        f"{'slack_all':>9} {'GBDT_all':>9} {'slack_wl':>9} {'GBDT_wl':>9}"
    )
    print(hdr + "   (all=overall, wl=within-length)", flush=True)
    points = []
    for li, lam in enumerate(lams):
        train = _run_split(lam, li * _BAND, args.n_train, args.workers)
        val = _run_split(lam, li * _BAND + _BAND // 2, args.n_val, args.workers)
        p = evaluate_g0_point(lam, train, val)
        points.append(p)
        print(
            f"{p.lam:6.2f} {p.n_scenes:7d} {p.oracle_solve_rate:7.2f} "
            f"{100*p.feasible_frac:6.1f} {100*p.marginal_frac:6.1f} "
            f"{p.slack_auroc:9.3f} {p.gbdt_auroc:9.3f} "
            f"{p.slack_within_auroc:9.3f} {p.gbdt_within_auroc:9.3f}",
            flush=True,
        )
        if p.top_features:
            tf = ", ".join(f"{n}={v:.3f}" for n, v in p.top_features)
            print(f"       GBDT top perm-importance: {tf}", flush=True)
    lam_star = choose_lambda_star(
        points, args.degrade_thresh, args.oracle_thresh, (args.op_lo, args.op_hi)
    )
    print("", flush=True)
    if lam_star is None:
        print(
            "# G0 OFF-RAMP: no λ with cheap-stats-degraded AND oracle-solves. "
            "DD2D as configured does not clearly support the subset-coupling claim.",
            flush=True,
        )
    else:
        print(
            f"# G0 PASS: λ* = {lam_star} (cheap stats degrade, oracle solves) — "
            "subset-coupled feasibility binds here.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
