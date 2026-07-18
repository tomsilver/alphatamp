"""Load cached DD2D per-method test scores and derive rollout false-positives.

The heavy method evaluations (fresh PIGINet inference, SPECTRE rollouts over 3
seeds) are precomputed once by ``experiments/spectre/precompute_dd2d_cache.py``,
which writes one JSON per (method[, seed], problem) under a cache directory. This
module is the **pure, dependency-light reader**: it turns those cached raw scores
into per-problem rollout-FP records for the comparison notebook. No torch / no
piginet import, so it stays fully CI-checked and fast to load.

Cache layout (``<cache_dir>/``):

- ``astar/<pid>.json``                    → ``{problem_id, stratum, scores, labels}``
- ``piginet_v3/<pid>.json``               → ``{problem_id, stratum, scores, labels}``
- ``spectre_static/seed_<s>/<pid>.json``  → ``{problem_id, stratum, scores, labels}``
- ``spectre_adaptive/seed_<s>/<pid>.json``→ ``{problem_id, stratum, fp}``

For the static methods ``scores[j]`` / ``labels[j]`` align with pool index =
``plan_idx`` = ``skeleton_idx``; FP is derived identically for all three via
:func:`rollout_fp`. SPECTRE-adaptive is an online rollout, so its per-problem FP
is cached directly. SPECTRE FPs are averaged over the training seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

# Display method name -> cache sub-directory. astar / PIGINet are deterministic
# (single run); SPECTRE has a per-seed sub-directory layer.
STATIC_METHODS: dict[str, str] = {
    "astar-dist": "astar",
    "PIGINet_v3": "piginet_v3",
}
SPECTRE_STATIC_METHOD = "SPECTRE-static"
SPECTRE_ADAPTIVE_METHOD = "SPECTRE-adaptive"
SPECTRE_STATIC_DIR = "spectre_static"
SPECTRE_ADAPTIVE_DIR = "spectre_adaptive"

# Presentation order used by the notebook.
METHOD_ORDER: list[str] = [
    "astar-dist",
    "PIGINet_v3",
    SPECTRE_ADAPTIVE_METHOD,
    SPECTRE_STATIC_METHOD,
]


def stratum_of(seed: int) -> int:
    """Min-feasible-subset stratum (0..3) from a DD2D test seed band."""
    return min(3, (int(seed) - 1_000_000) // 250_000)


def rollout_fp(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """Rollout false-positives for a static ranking (higher score first).

    Number of infeasible skeletons ranked strictly above the best-scoring
    feasible one, with half credit for exact score ties. Returns ``None`` if the
    pool has no feasible skeleton. Mirrors
    ``piginet.eval._rollout_fp_group`` exactly, so all methods share accounting.
    """
    pos = [s for s, lbl in zip(scores, labels) if lbl > 0.5]
    if not pos:
        return None
    top = max(pos)
    strict = sum(1 for s, lbl in zip(scores, labels) if lbl < 0.5 and s > top)
    ties = sum(1 for s, lbl in zip(scores, labels) if lbl < 0.5 and s == top)
    return float(strict) + 0.5 * float(ties)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_dir(path: Path, cache_dir: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing cache directory {path}. Build the cache first with:\n"
            f"  python experiments/spectre/precompute_dd2d_cache.py\n"
            f"(expected under {cache_dir})"
        )
    return path


def _static_fp_by_pid(method_dir: Path) -> dict[int, tuple[int, float]]:
    """Map problem_id -> (stratum, FP) from a static-score cache directory."""
    out: dict[int, tuple[int, float]] = {}
    for path in sorted(method_dir.glob("*.json")):
        rec = _load_json(path)
        pid = int(rec["problem_id"])
        fp = rollout_fp(rec["scores"], rec["labels"])
        if fp is not None:
            out[pid] = (int(rec["stratum"]), fp)
    return out


def _spectre_seed_mean(parent: Path, is_adaptive: bool) -> dict[int, tuple[int, float]]:
    """Average per-problem FP over ``seed_*`` sub-dirs.

    Adaptive dirs store ``fp`` directly; static dirs store raw ``scores`` /
    ``labels`` and FP is derived via :func:`rollout_fp`.
    """
    per_pid_fps: dict[int, list[float]] = {}
    per_pid_stratum: dict[int, int] = {}
    seed_dirs = sorted(parent.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* sub-directories under {parent}")
    for seed_dir in seed_dirs:
        for path in sorted(seed_dir.glob("*.json")):
            rec = _load_json(path)
            pid = int(rec["problem_id"])
            if is_adaptive:
                fp: float | None = float(rec["fp"])
            else:
                fp = rollout_fp(rec["scores"], rec["labels"])
            if fp is None:
                continue
            per_pid_fps.setdefault(pid, []).append(fp)
            per_pid_stratum[pid] = int(rec["stratum"])
    return {
        pid: (per_pid_stratum[pid], sum(fps) / len(fps))
        for pid, fps in per_pid_fps.items()
    }


def load_fp_records(cache_dir: Path | str) -> list[dict]:
    """Read the cache and return per-(method, problem) rollout-FP records.

    Each record is ``{"problem_id": int, "stratum": int, "method": str,
    "fp": float}``. SPECTRE methods are averaged over the cached seeds. The
    notebook wraps this in a ``pandas.DataFrame``; keeping this dependency-free
    lets the loader stay CI-checked without adding pandas to the package.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Cache directory {cache_dir} does not exist. Build it with:\n"
            f"  python experiments/spectre/precompute_dd2d_cache.py"
        )

    records: list[dict] = []
    for method, subdir in STATIC_METHODS.items():
        by_pid = _static_fp_by_pid(_require_dir(cache_dir / subdir, cache_dir))
        for pid, (stratum, fp) in by_pid.items():
            records.append(
                {"problem_id": pid, "stratum": stratum, "method": method, "fp": fp}
            )

    for method, subdir, adaptive in (
        (SPECTRE_ADAPTIVE_METHOD, SPECTRE_ADAPTIVE_DIR, True),
        (SPECTRE_STATIC_METHOD, SPECTRE_STATIC_DIR, False),
    ):
        by_pid = _spectre_seed_mean(
            _require_dir(cache_dir / subdir, cache_dir), is_adaptive=adaptive
        )
        for pid, (stratum, fp) in by_pid.items():
            records.append(
                {"problem_id": pid, "stratum": stratum, "method": method, "fp": fp}
            )

    return records


def load_named_fp_records(
    cache_dir: Path | str, subdir: str, method_name: str
) -> list[dict]:
    """Seed-averaged FP records for an adaptive-style intervention cache.

    Reads ``<cache_dir>/<subdir>/seed_*/<pid>.json`` (each carrying a precomputed
    ``fp``, like ``spectre_adaptive``) and returns
    ``{problem_id, stratum, method, fp}`` per problem, averaged over seeds. Used
    for the T1 length-only-context arm (``subdir="spectre_lenctx"``).
    """
    cache_dir = Path(cache_dir)
    by_pid = _spectre_seed_mean(
        _require_dir(cache_dir / subdir, cache_dir), is_adaptive=True
    )
    return [
        {"problem_id": pid, "stratum": stratum, "method": method_name, "fp": fp}
        for pid, (stratum, fp) in by_pid.items()
    ]


# ---------------------------------------------------------------------------
# T0 — length-dependence of a ranking (does length alone explain the scores?)
#
# H1 (docs/spectre_piginet_hypotheses_and_tests_v2.md) predicts SPECTRE-static's
# one-shot ranking is essentially a plan-length ranking with no same-size-subset
# knowledge. These pure, numpy-only helpers quantify that per episode so the
# marimo notebook can display it without re-running any model. "length" is a
# skeleton's operator count (``2·(blockers staged)+1`` in the DD2D drawer domain).
# ---------------------------------------------------------------------------


def _rankdata(values: ArrayLike) -> np.ndarray:
    """Average ranks (1-based) with ties averaged, matching ``scipy.rankdata``."""
    arr = np.asarray(values, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(arr.size, dtype=np.intp)
    inv[sorter] = np.arange(arr.size)
    arr_sorted = arr[sorter]
    obs = np.concatenate(([True], arr_sorted[1:] != arr_sorted[:-1]))
    dense = obs.cumsum()[inv]
    # boundaries[k] = # elements with dense rank < k+1
    boundaries = np.concatenate((np.nonzero(obs)[0], [arr.size]))
    return 0.5 * (boundaries[dense] + boundaries[dense - 1] + 1)


def _spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation (tie-corrected); NaN if either side is constant."""
    xr = _rankdata(x)
    yr = _rankdata(y)
    if xr.std() == 0.0 or yr.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _eta2(scores: np.ndarray, lengths: np.ndarray) -> float:
    """Fraction of score variance explained by length group (categorical R²).

    ``1 - SS_within(length) / SS_total``. This is the tightest "length explains
    the scores" measure (it credits any length-only structure, monotone or not),
    unlike a linear R² which assumes score is linear in length. NaN if the scores
    are constant (no variance to explain).
    """
    grand = float(scores.mean())
    ss_tot = float(((scores - grand) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    ss_within = 0.0
    for group in np.unique(lengths):
        vals = scores[lengths == group]
        ss_within += float(((vals - vals.mean()) ** 2).sum())
    return 1.0 - ss_within / ss_tot


def length_fit(scores: Sequence[float], lengths: Sequence[float]) -> dict:
    """Per-episode length-dependence of a ranking.

    Returns ``{eta2, within_frac, spearman, n_len, n}`` where ``eta2`` is the
    categorical-length R² (see :func:`_eta2`), ``within_frac = 1 - eta2`` is the
    share of score variance that is *within* length groups (i.e. genuine
    same-length / subset discrimination), and ``spearman`` is the rank
    correlation between score and length (sign = long-plan preference: positive
    means higher-scored, tried-earlier candidates are the longer ones).
    """
    sc = np.asarray(scores, dtype=float)
    ln = np.asarray(lengths, dtype=float)
    if sc.size != ln.size:
        raise ValueError(f"scores/lengths length mismatch: {sc.size} vs {ln.size}")
    eta2 = _eta2(sc, ln)
    within = float("nan") if np.isnan(eta2) else 1.0 - eta2
    return {
        "eta2": eta2,
        "within_frac": within,
        "spearman": _spearman(sc, ln),
        "n_len": int(np.unique(ln).size),
        "n": int(sc.size),
    }


def mean_position_by_length(
    scores: Sequence[float], lengths: Sequence[float]
) -> dict[int, float]:
    """Mean within-episode percentile attempt-position of each length tier.

    The ranking tries candidates in descending score order; position 0.0 = tried first,
    1.0 = tried last. Composition-robust (percentiles are within-episode), so averaging
    across episodes with different length mixes is meaningful. Reveals the *shape* of
    the learned length curve (monotone short-first, flat, etc.).
    """
    sc = np.asarray(scores, dtype=float)
    ln = np.asarray(lengths, dtype=float)
    n = sc.size
    order = np.argsort(-sc, kind="stable")  # highest score (tried first) -> position 0
    pos = np.empty(n, dtype=float)
    pos[order] = np.arange(n, dtype=float)
    pct = pos / (n - 1) if n > 1 else np.zeros(n, dtype=float)
    return {int(g): float(pct[ln == g].mean()) for g in np.unique(ln)}


def length_ladder(order: Sequence[int], lengths: Sequence[float]) -> dict:
    """Length trajectory of a *realized* attempt order (adaptive rollout).

    ``order`` is the sequence of pool indices actually attempted (until first
    success). Returns ``{spearman, slope, n_steps, first_len, last_len}`` where
    ``spearman``/``slope`` correlate attempt position (0,1,2,…) with the tried
    plan's length. Positive ⇒ the method climbs to longer plans as it fails (a
    length-escalation ladder).
    """
    ln = np.asarray(lengths, dtype=float)
    tried = np.array([ln[i] for i in order], dtype=float)
    steps = tried.size
    positions = np.arange(steps, dtype=float)
    slope = float("nan")
    if steps >= 2 and positions.std() > 0 and tried.std() > 0:
        slope = float(np.polyfit(positions, tried, 1)[0])
    return {
        "spearman": _spearman(positions, tried) if steps >= 2 else float("nan"),
        "slope": slope,
        "n_steps": int(steps),
        "first_len": float(tried[0]) if steps else float("nan"),
        "last_len": float(tried[-1]) if steps else float("nan"),
    }


def _safe_nanmean(values: Sequence[float]) -> float:
    """``np.nanmean`` that returns NaN (no warning) when every value is NaN."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or bool(np.all(np.isnan(arr))):
        return float("nan")
    return float(np.nanmean(arr))


_FIT_KEYS = ("eta2", "within_frac", "spearman")


def _length_fit_by_pid(
    method_dir: Path, lengths_by_pid: Mapping[int, Sequence[float]]
) -> dict[int, tuple[int, dict]]:
    """Map problem_id -> (stratum, length_fit) from a static-score cache dir."""
    out: dict[int, tuple[int, dict]] = {}
    for path in sorted(method_dir.glob("*.json")):
        rec = _load_json(path)
        if "scores" not in rec:
            continue
        pid = int(rec["problem_id"])
        out[pid] = (int(rec["stratum"]), length_fit(rec["scores"], lengths_by_pid[pid]))
    return out


def _spectre_static_length_fit_mean(
    parent: Path, lengths_by_pid: Mapping[int, Sequence[float]]
) -> dict[int, tuple[int, dict]]:
    """Seed-average the length-fit stats over ``seed_*`` sub-dirs."""
    per_pid: dict[int, list[dict]] = {}
    per_stratum: dict[int, int] = {}
    for seed_dir in sorted(parent.glob("seed_*")):
        for path in sorted(seed_dir.glob("*.json")):
            rec = _load_json(path)
            if "scores" not in rec:
                continue
            pid = int(rec["problem_id"])
            per_pid.setdefault(pid, []).append(
                length_fit(rec["scores"], lengths_by_pid[pid])
            )
            per_stratum[pid] = int(rec["stratum"])
    return {
        pid: (
            per_stratum[pid],
            {k: _safe_nanmean([f[k] for f in fits]) for k in _FIT_KEYS},
        )
        for pid, fits in per_pid.items()
    }


def load_length_fit_records(
    cache_dir: Path | str, lengths_by_pid: Mapping[int, Sequence[float]]
) -> list[dict]:
    """Per-(method, problem) length-fit records for the static-score methods.

    Covers ``astar-dist``, ``PIGINet_v3`` and ``SPECTRE-static`` (seed-averaged);
    each record is ``{problem_id, stratum, method, eta2, within_frac, spearman}``.
    SPECTRE-adaptive has no static per-skeleton scores — its one-shot ranking is
    provably identical to SPECTRE-static (same checkpoint, empty context → c₀), so
    the notebook mirrors static's row for it and uses
    :func:`load_adaptive_ladder_records` for the realized-order view.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Cache directory {cache_dir} does not exist. Build it with:\n"
            f"  python experiments/spectre/precompute_dd2d_cache.py"
        )
    records: list[dict] = []
    for method, subdir in STATIC_METHODS.items():
        by_pid = _length_fit_by_pid(
            _require_dir(cache_dir / subdir, cache_dir), lengths_by_pid
        )
        for pid, (stratum, fit) in by_pid.items():
            records.append(
                {"problem_id": pid, "stratum": stratum, "method": method, **fit}
            )
    by_pid = _spectre_static_length_fit_mean(
        _require_dir(cache_dir / SPECTRE_STATIC_DIR, cache_dir), lengths_by_pid
    )
    for pid, (stratum, fit) in by_pid.items():
        records.append(
            {
                "problem_id": pid,
                "stratum": stratum,
                "method": SPECTRE_STATIC_METHOD,
                **fit,
            }
        )
    return records


def _accumulate_positions(
    method_dir: Path,
    lengths_by_pid: Mapping[int, Sequence[float]],
    acc: dict[int, list[float]],
) -> None:
    """Append each episode's per-length mean percentile-position into ``acc``."""
    for path in sorted(method_dir.glob("*.json")):
        rec = _load_json(path)
        if "scores" not in rec:
            continue
        pid = int(rec["problem_id"])
        for length, p in mean_position_by_length(
            rec["scores"], lengths_by_pid[pid]
        ).items():
            acc.setdefault(length, []).append(p)


def load_position_by_length_records(
    cache_dir: Path | str, lengths_by_pid: Mapping[int, Sequence[float]]
) -> list[dict]:
    """Per-(method, length) mean percentile attempt-position (the length curve).

    Static-score methods only (astar-dist, PIGINet_v3, SPECTRE-static seed-pooled).
    Each record is ``{method, length, mean_pos, n_eps}`` with ``mean_pos`` in
    ``[0, 1]`` (0 = that length tier is tried first). Plots the *shape* of each
    method's learned length preference.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Cache directory {cache_dir} does not exist. Build it with:\n"
            f"  python experiments/spectre/precompute_dd2d_cache.py"
        )
    records: list[dict] = []
    for method, subdir in STATIC_METHODS.items():
        acc: dict[int, list[float]] = {}
        _accumulate_positions(
            _require_dir(cache_dir / subdir, cache_dir), lengths_by_pid, acc
        )
        for length, positions in sorted(acc.items()):
            records.append(
                {
                    "method": method,
                    "length": int(length),
                    "mean_pos": float(np.mean(positions)),
                    "n_eps": len(positions),
                }
            )
    acc = {}
    parent = _require_dir(cache_dir / SPECTRE_STATIC_DIR, cache_dir)
    for seed_dir in sorted(parent.glob("seed_*")):
        _accumulate_positions(seed_dir, lengths_by_pid, acc)
    for length, positions in sorted(acc.items()):
        records.append(
            {
                "method": SPECTRE_STATIC_METHOD,
                "length": int(length),
                "mean_pos": float(np.mean(positions)),
                "n_eps": len(positions),
            }
        )
    return records


_LADDER_KEYS = ("spearman", "slope", "n_steps", "first_len", "last_len")


def load_adaptive_ladder_records(
    cache_dir: Path | str, lengths_by_pid: Mapping[int, Sequence[float]]
) -> list[dict]:
    """Per-problem realized-order length-ladder for SPECTRE-adaptive.

    Reads the ``order`` field written by ``precompute_dd2d_cache.py`` (the
    sequence of attempted pool indices) and seed-averages :func:`length_ladder`.
    Returns ``[]`` if no cached trace carries ``order`` (pre-trace cache), so the
    notebook can degrade gracefully. Each record is
    ``{problem_id, stratum, method, spearman, slope, n_steps, first_len, last_len}``.
    """
    parent = _require_dir(Path(cache_dir) / SPECTRE_ADAPTIVE_DIR, Path(cache_dir))
    per_pid: dict[int, list[dict]] = {}
    per_stratum: dict[int, int] = {}
    for seed_dir in sorted(parent.glob("seed_*")):
        for path in sorted(seed_dir.glob("*.json")):
            rec = _load_json(path)
            order = rec.get("order")
            if order is None:
                continue
            pid = int(rec["problem_id"])
            per_pid.setdefault(pid, []).append(
                length_ladder(order, lengths_by_pid[pid])
            )
            per_stratum[pid] = int(rec["stratum"])
    return [
        {
            "problem_id": pid,
            "stratum": per_stratum[pid],
            "method": SPECTRE_ADAPTIVE_METHOD,
            **{k: _safe_nanmean([lad[k] for lad in lads]) for k in _LADDER_KEYS},
        }
        for pid, lads in per_pid.items()
    ]
