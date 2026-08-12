"""Gate G0 — does the DD2D benchmark test its own thesis? (v2.2.1 §10.2).

G0 asks, *before any model code*, whether there is a buffer-tightness λ where **cheap
statistics degrade but the oracle still solves** — the regime where subset-coupled
feasibility binds (feasibility depends on *which* subset packs, not on a low-order
statistic like area slack). If no such λ exists, DD2D as configured cannot support the
subset-coupling claim and the honest next step is benchmark work, not model work
(pre-registered off-ramp).

This module provides the cheap probes and the per-λ analysis:

- **slack ordering** — rank a candidate subset ``S`` by buffer slack (buffer area − Σ of
  the δ/2-deflated member areas); the strongest one-scalar shortcut.
- **pairwise-features GBDT probe** — per-candidate hand features (per-object area /
  perimeter / circularity / caliper aggregates, |S|, Σ area, buffer slack ratio, a
  pairwise NFP-complementarity proxy) → P(feasible), a gradient-boosted-tree classifier.

Both are scored by AUROC at predicting per-candidate feasibility on held-out scenes;
"the oracle still solves" is the fraction of scenes with ≥ 1 feasible candidate. The
data are generated per λ (the sweep is driven by ``experiments/spectre/spectre_g0.py``);
candidates are labeled with the §8.4 certificate (``use_certificate=True``) so negatives
are trustworthy, and marginals are excluded from the AUROC (reported separately).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Feature names, fixed order (so the GBDT design matrix is stable across λ).
FEATURE_NAMES = (
    "n_items",  # |S|
    "sum_area",
    "max_area",
    "min_area",
    "mean_area",
    "slack",  # buffer_area − Σ deflated member areas
    "slack_ratio",  # slack / buffer_area
    "mean_perim",
    "max_perim",
    "mean_circularity",  # 4π·area / perim²  (1 = disk)
    "min_circularity",
    "mean_caliper",  # max bbox side
    "max_caliper",
    "mean_aspect",  # max_side / min_side
    # Σ_{i<j} area_i·area_j / buffer_area²  (packing pressure)
    "pair_area_complementarity",
)


def _shape_features(poly) -> tuple[float, float, float, float, float]:
    """(area, perimeter, circularity, caliper=max bbox side, aspect=max/min side)."""
    area = float(poly.area)
    perim = float(poly.length)
    x0, y0, x1, y1 = poly.bounds
    w, h = x1 - x0, y1 - y0
    caliper = max(w, h)
    aspect = (max(w, h) / min(w, h)) if min(w, h) > 1e-9 else 1.0
    circ = (4.0 * math.pi * area / (perim * perim)) if perim > 1e-9 else 0.0
    return area, perim, circ, caliper, aspect


def buffer_slack(scene, subset, half_delta: Optional[float] = None) -> float:
    """Buffer area − Σ of the δ/2-deflated member areas (the slack-ordering scalar)."""
    if half_delta is None:
        half_delta = scene.margin / 2.0
    buffer_area = float(scene.buffer.area)
    used = 0.0
    for name in subset:
        d = scene.items[name].shape.polygon.buffer(-half_delta)
        used += float(d.area) if (not d.is_empty and d.geom_type == "Polygon") else 0.0
    return buffer_area - used


def candidate_features(scene, subset) -> dict[str, float]:
    """Low-order hand features for one candidate subset (the cheap-statistics probe)."""
    subset = list(subset)
    buffer_area = float(scene.buffer.area)
    areas, perims, circs, calipers, aspects = [], [], [], [], []
    for name in subset:
        a, p, c, cal, asp = _shape_features(scene.items[name].shape.polygon)
        areas.append(a)
        perims.append(p)
        circs.append(c)
        calipers.append(cal)
        aspects.append(asp)
    slack = buffer_slack(scene, subset)
    n = len(subset)
    pair = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pair += areas[i] * areas[j]
    pair = pair / (buffer_area * buffer_area) if buffer_area > 0 else 0.0
    return {
        "n_items": float(n),
        "sum_area": float(sum(areas)),
        "max_area": float(max(areas)) if areas else 0.0,
        "min_area": float(min(areas)) if areas else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "slack": slack,
        "slack_ratio": slack / buffer_area if buffer_area > 0 else 0.0,
        "mean_perim": float(np.mean(perims)) if perims else 0.0,
        "max_perim": float(max(perims)) if perims else 0.0,
        "mean_circularity": float(np.mean(circs)) if circs else 0.0,
        "min_circularity": float(min(circs)) if circs else 0.0,
        "mean_caliper": float(np.mean(calipers)) if calipers else 0.0,
        "max_caliper": float(max(calipers)) if calipers else 0.0,
        "mean_aspect": float(np.mean(aspects)) if aspects else 0.0,
        "pair_area_complementarity": pair,
    }


def feature_vector(scene, subset) -> np.ndarray:
    f = candidate_features(scene, subset)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)


@dataclass
class LabeledCandidates:
    """Per-candidate feature matrix + labels for one split at one λ.

    ``X``/``y``/``slack`` cover only the **confidently-labeled** candidates (feasible ∪
    infeasible); ``y`` is 1 for feasible, 0 for infeasible. Marginal candidates are
    counted (``n_marginal``) but excluded from the AUROC. ``n_oracle_solved`` is the
    number of scenes with ≥ 1 feasible candidate (the oracle solve numerator)."""

    X: np.ndarray  # confidently-labeled rows only (feasible ∪ infeasible)
    y: np.ndarray  # 1 feasible / 0 infeasible
    slack: np.ndarray  # slack scalar for the confident rows
    sizes: np.ndarray  # |S| per confident row (for the within-length AUROC)
    n_scenes: int
    n_oracle_solved: int  # scenes with >= 1 feasible candidate
    n_marginal: int
    n_total_candidates: int

    @property
    def oracle_solve_rate(self) -> float:
        return self.n_oracle_solved / self.n_scenes if self.n_scenes else 0.0

    @property
    def marginal_frac(self) -> float:
        return (
            self.n_marginal / self.n_total_candidates
            if self.n_total_candidates
            else 0.0
        )


def collect_labeled_candidates(scenes_and_candidates) -> LabeledCandidates:
    """Build a ``LabeledCandidates`` from an iterable of
    ``(scene, labeled_candidates)``.

    Each ``labeled_candidates`` is a list of DD2D ``Candidate`` with ``meta["label"]``
    in {feasible, infeasible, marginal} (from
    ``label_all(..., use_certificate=True)``)."""
    rows, ys, slacks, sizes = [], [], [], []
    n_scenes = n_solved = n_marginal = n_total = 0
    for scene, cands in scenes_and_candidates:
        n_scenes += 1
        solved = False
        for c in cands:
            n_total += 1
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
        if solved:
            n_solved += 1
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


@dataclass
class G0Point:
    """G0 metrics at one λ.

    The *overall* AUROCs are length-inflated (the snapshot showed DD2D feasibility is
    length/count-dominated, and the features include ``n_items``/``sum_area``), so the
    thesis-relevant signal is the **within-length** AUROC — the size-conditional AUROC,
    controlling for |S|. Cheap stats "degrade" for G0 when even the GBDT cannot beat
    chance *within* a subset size, i.e. it fails to capture the subset-identity
    residual a rich representation would target."""

    lam: float
    n_scenes: int
    oracle_solve_rate: float
    feasible_frac: float  # feasible / confidently-labeled
    marginal_frac: float
    slack_auroc: float  # overall AUROC of the slack scalar (length-inflated)
    gbdt_auroc: float  # overall AUROC of the GBDT probe (length-inflated)
    slack_within_auroc: float  # size-conditional AUROC of slack
    gbdt_within_auroc: float  # size-conditional AUROC of the GBDT — the key signal
    n_conf: int  # confidently-labeled candidates evaluated
    top_features: tuple[tuple[str, float], ...] = ()  # GBDT permutation importances

    def cheap_degraded(self, thresh: float = 0.65) -> bool:
        """Cheap statistics fail to capture the *within-length* subset-identity
        residual."""
        return self.gbdt_within_auroc < thresh


def _safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")  # undefined without both classes
    return float(roc_auc_score(y, score))


def within_length_auroc(y: np.ndarray, score: np.ndarray, sizes: np.ndarray) -> float:
    """Size-conditional AUROC: P(feasible outranks infeasible | same |S|), pooled over
    sizes (Mann–Whitney concordance summed within each size group). Controls for the
    length/count axis so the residual is genuine within-length discrimination."""
    concordant = 0.0
    total = 0
    for s in np.unique(sizes):
        m = sizes == s
        pos = score[m & (y == 1)]
        neg = score[m & (y == 0)]
        if pos.size == 0 or neg.size == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        concordant += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        total += pos.size * neg.size
    return concordant / total if total else float("nan")


def evaluate_g0_point(
    lam: float, train: LabeledCandidates, val: LabeledCandidates
) -> G0Point:
    """Fit the GBDT on ``train``, score both probes on ``val`` (overall +
    within-length)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    slack_auroc = _safe_auroc(val.y, val.slack)
    slack_within = within_length_auroc(val.y, val.slack, val.sizes)
    gbdt_auroc = gbdt_within = float("nan")
    top_features: tuple[tuple[str, float], ...] = ()
    if len(np.unique(train.y)) >= 2 and val.X.shape[0] > 0:
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05, random_state=0
        )
        clf.fit(train.X, train.y)
        proba = clf.predict_proba(val.X)[:, 1]
        gbdt_auroc = _safe_auroc(val.y, proba)
        gbdt_within = within_length_auroc(val.y, proba, val.sizes)
        if len(np.unique(val.y)) >= 2:
            imp = permutation_importance(
                clf, val.X, val.y, n_repeats=5, random_state=0, scoring="roc_auc"
            )
            order = np.argsort(imp.importances_mean)[::-1][:5]
            top_features = tuple(
                (FEATURE_NAMES[i], float(imp.importances_mean[i])) for i in order
            )
    feasible_frac = float(val.y.mean()) if val.y.size else 0.0
    return G0Point(
        lam=lam,
        n_scenes=val.n_scenes,
        oracle_solve_rate=val.oracle_solve_rate,
        feasible_frac=feasible_frac,
        marginal_frac=val.marginal_frac,
        slack_auroc=slack_auroc,
        gbdt_auroc=gbdt_auroc,
        slack_within_auroc=slack_within,
        gbdt_within_auroc=gbdt_within,
        n_conf=int(val.y.size),
        top_features=top_features,
    )


def choose_lambda_star(
    points: list[G0Point],
    degrade_thresh: float = 0.65,
    oracle_thresh: float = 0.5,
    operating_range: tuple[float, float] = (0.7, 0.95),
) -> Optional[float]:
    """λ* = the λ that best exhibits "subset-coupled feasibility binds" — cheap stats
    degrade *within-length* (GBDT within-length AUROC < ``degrade_thresh``) yet the
    oracle still solves (solve rate ≥ ``oracle_thresh``) — chosen to **maximize the
    oracle−GBDT_wl gap**. λ* is **constrained to DD2D's designed operating range**
    (``operating_range``, default 0.7–0.95, the naturalistic/loose regime); tighter λ is
    off-design (3-subsets stop packing, so stratum-3 becomes ungenerable) and must not
    be selected even if it maximizes the gap. ``None`` triggers the G0 off-ramp."""
    lo, hi = operating_range
    candidates = [
        p
        for p in points
        if lo <= p.lam <= hi
        and p.cheap_degraded(degrade_thresh)
        and p.oracle_solve_rate >= oracle_thresh
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.oracle_solve_rate - p.gbdt_within_auroc).lam
