"""Elimination ladder — the anti-shortcut acceptance test (proposal §10.3, Step 9).

"η²(length) < 1" is far too weak: a model that learns *only* area slack passes it while
remaining subset-blind (area is the new length). Acceptance is a **nested variance
decomposition** of the v2.2-static scores — length → +slack → +pairwise proximity →
**residual (true subset identity)** — plus the operational bar: *v2.2-static beats the
slack ordering by a paired margin, CI excluding zero, on strata ≥ 2*. This module is
label/model-agnostic: it takes per-skeleton scores + features + a per-problem grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LadderRungs:
    """Fraction of score variance explained as features are added, and the residual."""

    r2_length: float
    r2_length_slack: float
    r2_length_slack_proximity: float
    residual: float  # 1 − R²(length+slack+proximity): the subset-identity residual

    def as_row(self) -> str:
        return (
            f"length={self.r2_length:.3f} +slack={self.r2_length_slack:.3f} "
            f"+proximity={self.r2_length_slack_proximity:.3f} "
            f"residual={self.residual:.3f}"
        )


def _r2(y: np.ndarray, x: np.ndarray) -> float:
    """R² of an OLS fit of ``y`` on columns ``x`` (with intercept)."""
    if x.ndim == 1:
        x = x[:, None]
    a = np.hstack([np.ones((len(y), 1)), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def variance_ladder(
    scores: np.ndarray, length: np.ndarray, slack: np.ndarray, proximity: np.ndarray
) -> LadderRungs:
    """Nested R² of ``scores`` on length → +slack → +proximity; residual is what a rich
    representation captures beyond those cheap statistics."""
    r1 = _r2(scores, length)
    r2 = _r2(scores, np.column_stack([length, slack]))
    r3 = _r2(scores, np.column_stack([length, slack, proximity]))
    return LadderRungs(r1, r2, r3, 1.0 - r3)


def _rollout_fp(order: np.ndarray, feasible: np.ndarray) -> int:
    """Infeasible skeletons tried before the first feasible one, following ``order``."""
    fp = 0
    for idx in order:
        if feasible[idx]:
            return fp
        fp += 1
    return fp  # pool exhausted (no feasible) — should not happen on solvable problems


def beats_slack_paired(
    v2_scores: list[np.ndarray],
    slack_per_skel: list[np.ndarray],
    feasible: list[np.ndarray],
    strata: np.ndarray,
    min_stratum: int = 2,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired rollout-FP: rank each problem's pool by v2 scores vs by slack, take the
    per-problem FP difference (slack − v2, so positive = v2 better) on problems with
    stratum ≥ ``min_stratum``, and bootstrap the mean difference. Returns the mean and a
    95% CI; the ladder passes when the CI excludes zero (v2 strictly better)."""
    diffs = []
    for i in range(len(v2_scores)):
        if strata[i] < min_stratum or not feasible[i].any():
            continue
        v2_order = np.argsort(-v2_scores[i])
        slack_order = np.argsort(-slack_per_skel[i])
        d = _rollout_fp(slack_order, feasible[i]) - _rollout_fp(v2_order, feasible[i])
        diffs.append(float(d))
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return {
            "n": 0,
            "mean_diff": float("nan"),
            "ci": (float("nan"), float("nan")),
            "passes": False,
        }
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(diffs),
        "mean_diff": float(diffs.mean()),
        "ci": (float(lo), float(hi)),
        "passes": bool(lo > 0.0),  # v2 strictly beats slack, CI excludes zero
    }
