"""Load cached DD2D per-method test scores and derive rollout false-positives.

The heavy method evaluations (fresh PIGINet inference, SPECTRE rollouts) are
precomputed once by ``experiments/spectre/precompute_dd2d_cache.py``, which writes
one JSON per (method[, seed], problem) under a cache directory. This module is the
**pure, dependency-light reader**: it turns those cached raw scores into
per-problem rollout-FP records for the comparison notebook. No torch / no piginet
import, so it stays fully CI-checked and fast to load.

Cache layout (``<cache_dir>/``):

- ``astar/<pid>.json``                     → ``{problem_id, stratum, scores, labels}``
- ``piginet/<pid>.json``                   → ``{problem_id, stratum, scores, labels}``
- ``spectre_static/seed_<s>/<pid>.json``   → ``{problem_id, stratum, scores, labels}``
- ``spectre_adaptive/seed_<s>/<pid>.json`` → ``{problem_id, stratum, fp, order[,
  step_scores, step_dead]}``
- ``spectre2_static/seed_<s>/<pid>.json``  → ``{problem_id, stratum, scores, labels}``
- ``spectre2_adaptive/seed_<s>/<pid>.json``→ ``{problem_id, stratum, fp, order[,
  step_scores, step_dead]}``
- ``spectre3_static``/``spectre3_adaptive`` → same two shapes, for v3
- ``spectre3_abl_<arm>/seed_<s>/<pid>.json`` → adaptive shape; one dir per ablation
  arm, read by name via :func:`load_named_fp_records_per_seed` rather than through
  :data:`SPECTRE_FAMILIES` (an ablation is one method's components switched off, not
  a method in the comparison)

``step_scores``/``step_dead`` are optional (older caches lack them) and only the
single-problem :func:`load_adaptive_trace` accessor reads them; the aggregate
loaders below ignore the extra keys entirely.

For the static methods ``scores[j]`` / ``labels[j]`` align with pool index =
``plan_idx`` = ``skeleton_idx``; FP is derived identically for all via
:func:`rollout_fp`. A ``*-adaptive`` method is an online rollout, so its
per-problem FP is cached directly. SPECTRE FPs are averaged over the cached seeds
(v1/v2 currently ship a single seed each — a 1-seed dev figure).

Two collections can be shown side by side via :func:`merge_collections`: DD2D
re-collections share their test problem-id set, so a per-problem join is well
defined even though the underlying scenes differ slightly (``decisions.md``
2026-07-26). Use it only for methods that have no row on the newer collection.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

# Display method name -> cache sub-directory. astar / PIGINet are deterministic
# (single run); the SPECTRE families have a per-seed sub-directory layer.
STATIC_METHODS: dict[str, str] = {
    "astar-dist": "astar",
    "PIGINet": "piginet",
}
# SPECTRE v1 (abstract-only re-ranker).
SPECTRE_STATIC_METHOD = "SPECTRE-static"
SPECTRE_ADAPTIVE_METHOD = "SPECTRE-adaptive"
SPECTRE_STATIC_DIR = "spectre_static"
SPECTRE_ADAPTIVE_DIR = "spectre_adaptive"
# SPECTRE v2.2 (geometry + typed-evidence re-ranker; observed proof-demotion).
SPECTREV2_STATIC_METHOD = "SPECTREv2-static"
SPECTREV2_ADAPTIVE_METHOD = "SPECTREv2-adaptive"
SPECTREV2_STATIC_DIR = "spectre2_static"
SPECTREV2_ADAPTIVE_DIR = "spectre2_adaptive"
# SPECTRE v3 (FailureRecord + observed coverage/waste; strict proof-demotion).
SPECTREV3_STATIC_METHOD = "SPECTREv3-static"
SPECTREV3_ADAPTIVE_METHOD = "SPECTREv3-adaptive"
SPECTREV3_STATIC_DIR = "spectre3_static"
SPECTREV3_ADAPTIVE_DIR = "spectre3_adaptive"

# Seeded SPECTRE families: (static_method, static_dir, adaptive_method, adaptive_dir).
# Both static and adaptive of a family are two deployment modes of one checkpoint.
SPECTRE_FAMILIES: list[tuple[str, str, str, str]] = [
    (
        SPECTRE_STATIC_METHOD,
        SPECTRE_STATIC_DIR,
        SPECTRE_ADAPTIVE_METHOD,
        SPECTRE_ADAPTIVE_DIR,
    ),
    (
        SPECTREV2_STATIC_METHOD,
        SPECTREV2_STATIC_DIR,
        SPECTREV2_ADAPTIVE_METHOD,
        SPECTREV2_ADAPTIVE_DIR,
    ),
    (
        SPECTREV3_STATIC_METHOD,
        SPECTREV3_STATIC_DIR,
        SPECTREV3_ADAPTIVE_METHOD,
        SPECTREV3_ADAPTIVE_DIR,
    ),
]

# Sequence methods: a method that PRODUCES its own ordered attempt sequence instead of
# ranking the shared candidate pool. VLMPlan is zero-shot, so it has no pool to rank; its
# attempts may include plans the 200-candidate pool does not contain, which is why its FP
# is precomputed at cache-build time (only the builder knows the off-pool labels) and
# read back verbatim here. Same record shape as an adaptive trace (`fp` + `order`), so
# the existing seed-averaging reader handles it unchanged, and ``order`` carries ``-1``
# for an off-pool attempt. Absent dirs are skipped: the notebook loads with no VLM cache.
#
# One subdir per *arm* (model): a cache dir is one method row, so two models must never
# share one. Both arms are Qwen3-VL Instruct, differing only in size, so the pair is a
# clean scale comparison — see ``decisions.md`` 2026-07-25.
SEQUENCE_METHODS: dict[str, str] = {
    "VLMPlan-8B": "vlmplan_qwen8b",
    "VLMPlan-32B": "vlmplan_qwen32b",
}

# Presentation order used by the notebook.
METHOD_ORDER: list[str] = [
    "astar-dist",
    "PIGINet",
    SPECTRE_ADAPTIVE_METHOD,
    SPECTRE_STATIC_METHOD,
    SPECTREV2_ADAPTIVE_METHOD,
    SPECTREV2_STATIC_METHOD,
    SPECTREV3_ADAPTIVE_METHOD,
    SPECTREV3_STATIC_METHOD,
    *SEQUENCE_METHODS,
]


# DD2D's collector encodes the stratum in the seed: each split gets a disjoint
# ``SPLIT_BAND``-wide band (``collect._split_bands``: train [0,1M), test [1M,2M),
# val [2M,3M)), and each band is divided into one equal sub-band per stratum
# (``collect._stratum_bands`` over ``STRATA = (0,1,2,3)``). The record itself has no
# stratum field, so this arithmetic is the only way to recover it.
SPLIT_BAND = 1_000_000
STRATUM_BAND = SPLIT_BAND // 4


def stratum_of(seed: int) -> int:
    """Min-feasible-subset stratum (0..3) from a DD2D problem seed, any split.

    Taking the seed modulo the split band makes this split-agnostic. On the **test**
    split (seeds in [1M, 2M)) it is identical to the earlier test-only formula
    ``(seed - 1_000_000) // 250_000``, so every published number is unchanged; on train
    and val that formula returned negative or saturated strata.
    """
    return min(3, (int(seed) % SPLIT_BAND) // STRATUM_BAND)


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


def _spectre_per_seed(
    parent: Path, is_adaptive: bool
) -> list[tuple[int, int, int, float]]:
    """Per-``(seed, problem)`` FP: ``(seed, problem_id, stratum, fp)``.

    The seed-collapsing sibling :func:`_spectre_seed_mean` averages *within* a problem
    across seeds, which is right for a per-problem plot but destroys the between-seed
    spread -- so a std computed downstream from its output is the across-*problem*
    spread of a seed-mean, not seed noise. Every v3 gate is accepted on "no stratum
    regresses beyond seed noise", so that quantity has to be recoverable.
    """
    out: list[tuple[int, int, int, float]] = []
    seed_dirs = sorted(parent.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* sub-directories under {parent}")
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):  # pragma: no cover - defensive
            continue
        for path in sorted(seed_dir.glob("*.json")):
            rec = _load_json(path)
            fp = (
                float(rec["fp"])
                if is_adaptive
                else rollout_fp(rec["scores"], rec["labels"])
            )
            if fp is None:
                continue
            out.append((seed, int(rec["problem_id"]), int(rec["stratum"]), fp))
    return out


def load_fp_records_per_seed(cache_dir: Path | str) -> list[dict]:
    """Like :func:`load_fp_records` but keeps the seed axis.

    Records are ``{"seed", "problem_id", "stratum", "method", "fp"}``. A static baseline
    is emitted with ``seed=None`` when its cache is a flat directory of ``<pid>.json``,
    so a caller can tell "one deterministic run" from "one seed of several".

    **The static layout is detected, not assumed.** ``astar`` is a planner order and has
    no seed axis at all; PIGINet had none either until it gained a ``--seed`` flag
    (2026-07-28), so its dd2d_v2/v3 caches are flat and its dd2d_v4 cache is per-seed.
    Reading a ``seed_*`` layer when one exists, and ``seed=None`` when it does not, is
    what keeps a genuinely single-run row from being reported as a one-seed sample --
    ``build_table`` renders the two differently on purpose.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Cache directory {cache_dir} does not exist. Build it with:\n"
            f"  python experiments/spectre/precompute_dd2d_cache.py"
        )

    records: list[dict] = []
    missing: list[str] = []
    for method, subdir in STATIC_METHODS.items():
        # Unlike `load_fp_records`, a missing static baseline is skipped rather than
        # fatal: a newly-onboarded collection has SPECTRE checkpoints before it has a
        # retrained PIGINet (which trains on the native JSON with its own CLIP cache),
        # and
        # the yardstick row does not depend on it. Absent arms are *reported*, never
        # silently dropped -- a comparison table quietly missing a baseline is worse than
        # one that says so.
        parent = cache_dir / subdir
        if not parent.is_dir():
            missing.append(method)
            continue
        if sorted(parent.glob("seed_*")):
            rows = list(_spectre_per_seed(parent, is_adaptive=False))
        else:
            rows = [
                (None, pid, stratum, fp)  # type: ignore[misc]
                for pid, (stratum, fp) in _static_fp_by_pid(parent).items()
            ]
        for seed, pid, stratum, fp in rows:
            records.append(
                {
                    "seed": seed,
                    "problem_id": pid,
                    "stratum": stratum,
                    "method": method,
                    "fp": fp,
                }
            )
    if missing:
        warnings.warn(
            f"compare cache {cache_dir} has no rows for {', '.join(missing)}; "
            "the table will omit them",
            stacklevel=2,
        )

    seeded: list[tuple[str, str, bool]] = []
    for stat_m, stat_d, adap_m, adap_d in SPECTRE_FAMILIES:
        seeded.append((adap_m, adap_d, True))
        seeded.append((stat_m, stat_d, False))
    seeded.extend((m, d, True) for m, d in SEQUENCE_METHODS.items())

    for method, subdir, adaptive in seeded:
        parent = cache_dir / subdir
        if not parent.is_dir():  # a family may be absent (e.g. a v1-only cache)
            continue
        for seed, pid, stratum, fp in _spectre_per_seed(parent, is_adaptive=adaptive):
            records.append(
                {
                    "seed": seed,
                    "problem_id": pid,
                    "stratum": stratum,
                    "method": method,
                    "fp": fp,
                }
            )
    return records


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
        parent = _require_dir(cache_dir / subdir, cache_dir)
        # Same layout detection as `load_fp_records_per_seed`: a seeded static cache
        # (PIGINet on dd2d_v4 and later) is averaged over its seeds exactly as a SPECTRE
        # family is, so this function keeps returning one row per (method, problem).
        by_pid = (
            _spectre_seed_mean(parent, is_adaptive=False)
            if sorted(parent.glob("seed_*"))
            else _static_fp_by_pid(parent)
        )
        for pid, (stratum, fp) in by_pid.items():
            records.append(
                {"problem_id": pid, "stratum": stratum, "method": method, "fp": fp}
            )

    for stat_m, stat_d, adap_m, adap_d in SPECTRE_FAMILIES:
        for method, subdir, adaptive in (
            (adap_m, adap_d, True),
            (stat_m, stat_d, False),
        ):
            parent = cache_dir / subdir
            if not parent.is_dir():  # a family may be absent (e.g. v1-only cache)
                continue
            by_pid = _spectre_seed_mean(parent, is_adaptive=adaptive)
            for pid, (stratum, fp) in by_pid.items():
                records.append(
                    {"problem_id": pid, "stratum": stratum, "method": method, "fp": fp}
                )

    for method, subdir in SEQUENCE_METHODS.items():
        parent = cache_dir / subdir
        if not parent.is_dir():  # optional arm; the notebook works without it
            continue
        by_pid = _spectre_seed_mean(parent, is_adaptive=True)
        for pid, (stratum, fp) in by_pid.items():
            records.append(
                {"problem_id": pid, "stratum": stratum, "method": method, "fp": fp}
            )

    return records


def load_named_fp_records_per_seed(
    cache_dir: Path | str, subdir: str, method_name: str
) -> list[dict]:
    """Seed-preserving twin of :func:`load_named_fp_records`.

    Ablation arms live in their own cache sub-directories rather than in
    :data:`SPECTRE_FAMILIES` -- they are not methods in the comparison, they are one
    method's components switched off -- but a table over them still needs the seed axis
    for the same reason the main one does. Returns ``{seed, problem_id, stratum, method,
    fp}``; missing dirs raise, because an ablation cell silently vanishing would turn a
    2x2 into a 2x1 without saying so.
    """
    cache_dir = Path(cache_dir)
    parent = _require_dir(cache_dir / subdir, cache_dir)
    return [
        {
            "seed": seed,
            "problem_id": pid,
            "stratum": stratum,
            "method": method_name,
            "fp": fp,
        }
        for seed, pid, stratum, fp in _spectre_per_seed(parent, is_adaptive=True)
    ]


def merge_collections(
    primary: list[dict],
    legacy: list[dict],
    legacy_methods: Sequence[str],
    primary_name: str = "primary",
    legacy_name: str = "legacy",
) -> list[dict]:
    """Tag records with their collection and graft the named methods from ``legacy``.

    A method retrained on the newer collection must be read from *that* collection, so
    only ``legacy_methods`` -- the ones with no newer equivalent -- are taken from
    ``legacy``, and a name appearing in both resolves to ``primary``. The ``collection``
    key is added to every record rather than only the grafted ones, so a downstream table
    cannot show a mixed row without the provenance being available to display beside it.
    """
    want = set(legacy_methods)
    out = [{**r, "collection": primary_name} for r in primary]
    have = {r["method"] for r in out}
    out += [
        {**r, "collection": legacy_name}
        for r in legacy
        if r["method"] in want and r["method"] not in have
    ]
    return out


def select_seed(
    records: list[dict], prefer: int = 0
) -> tuple[list[dict], dict[str, object]]:
    """Reduce to one seed per method: ``prefer`` if cached, else the best seed.

    Returns ``(records, chosen)`` where ``chosen`` maps method -> the seed used, so a
    caller can display it. Deterministic single runs (``seed=None``) pass through
    untouched.

    "Best" = lowest mean FP. That is a *selection over seeds*, so it flatters a method
    that has several; it exists only as a fallback for methods whose seed 0 was never
    cached, and the chosen seed is reported rather than assumed. Prefer caching seed 0.
    """
    by_method: dict[str, set] = {}
    for r in records:
        by_method.setdefault(r["method"], set()).add(r["seed"])

    chosen: dict[str, object] = {}
    for method, seeds in by_method.items():
        if seeds == {None}:
            chosen[method] = None
        elif prefer in seeds:
            chosen[method] = prefer
        else:
            means = {
                s: _mean(
                    [
                        r["fp"]
                        for r in records
                        if r["method"] == method and r["seed"] == s
                    ]
                )
                for s in seeds
            }
            # `means.get`, not `lambda s: means[s]`: the lambda closes over the loop
            # variable, so it would read whichever `means` existed when it was *called*.
            chosen[method] = min(means, key=means.__getitem__)
    return [r for r in records if r["seed"] == chosen[r["method"]]], chosen


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
# Single-problem accessors for the notebook's planner inspector. The aggregate
# loaders above sweep whole directories; these read one problem so the inspector
# can re-render on a dropdown change without touching the rest of the cache.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveTrace:
    """One adaptive method's cached rollout on one problem.

    ``step_scores`` / ``step_dead`` are ``None`` for caches written before per-step
    scores were persisted; callers must degrade gracefully (the inspector hides its
    adaptive-score columns) rather than assume they are present.

    Within ``step_scores``, a JSON ``null`` — the model's own ``avail_mask`` on the
    candidates already in the failure context — is read back as ``nan``, i.e. "not
    available at that step". At step ``t`` those are exactly ``order[:t]``.
    """

    problem_id: int
    stratum: int
    fp: float
    order: list[int]
    step_scores: list[list[float]] | None
    step_dead: list[list[int]] | None


def load_static_scores(
    cache_dir: Path | str, subdir: str, problem_id: int
) -> dict | None:
    """Read one static-score record — ``{problem_id, stratum, scores, labels}``.

    Returns ``None`` when the problem is absent from that method's cache (PIGINet,
    for instance, is missing one test problem), so callers can skip rather than raise.
    """
    path = Path(cache_dir) / subdir / f"{int(problem_id)}.json"
    return _load_json(path) if path.is_file() else None


def load_adaptive_trace(
    cache_dir: Path | str, subdir: str, problem_id: int, seed: int = 0
) -> AdaptiveTrace | None:
    """Read one adaptive rollout trace from ``<cache_dir>/<subdir>/seed_<seed>/``.

    Returns ``None`` when the problem (or the whole family) is not cached.
    """
    path = Path(cache_dir) / subdir / f"seed_{int(seed)}" / f"{int(problem_id)}.json"
    if not path.is_file():
        return None
    rec = _load_json(path)
    scores = rec.get("step_scores")
    dead = rec.get("step_dead")
    return AdaptiveTrace(
        problem_id=int(rec["problem_id"]),
        stratum=int(rec["stratum"]),
        fp=float(rec["fp"]),
        order=[int(i) for i in rec["order"]],
        step_scores=(
            None
            if scores is None
            else [
                [float("nan") if x is None else float(x) for x in row] for row in scores
            ]
        ),
        step_dead=(None if dead is None else [[int(i) for i in row] for row in dead]),
    )


_VLMPLAN_DIAGNOSTIC_KEYS = (
    "fp",
    "censored",
    "n_attempts",
    "n_offpool",
    "n_fill_used",
    "n_live_refines",
    "first_success_source",
    "spearman_vs_published",
    # Generation-side quality, copied onto the row by the scorer.
    "n_proposed",
    "n_rounds",
    "n_truncated",
    "n_blocks",
    "n_malformed",
    "n_duplicate",
    "n_invalid",
    "stalled",
    "hit_max_rounds",
    # Run identity, so a table can say which model produced the row.
    "model",
    "run",
)


def load_vlmplan_diagnostics(
    cache_dir: Path | str, subdir: str, seed: int = 0
) -> list[dict]:
    """Per-problem generation/scoring diagnostics for a sequence-method arm.

    Returns ``[]`` when the arm is not cached, so the notebook can render an empty
    section rather than raise. Each row carries the keys the VLMPlan section reports:

    - ``n_offpool`` — attempts on plans the 200-candidate pool does not contain. A high
      rate is a finding about pool coverage, not a defect.
    - ``first_success_source`` — ``"vlm"`` if the model itself found the feasible plan,
      ``"fill"`` if the published-order fallback found it after the model ran dry, or
      ``None`` if the budget was exhausted. The headline "does it work at all" question.
    - ``spearman_vs_published`` — the pre-registered **trivial-mimicry null**: near 1
      means the model reproduced the planner's size-ascending order, so its FP says
      little
      about geometric reasoning regardless of where it lands.
    - ``n_rounds`` / ``stalled`` — where zero-shot proposal capacity ran out.
    - ``n_truncated`` — rounds whose completion hit the output cap, so their last plan
      block was cut mid-line and dropped. Nonzero means the run under-reports what the
      model could produce; it is a config problem, not a model result.

    ``subdir`` is required: there is one dir per model arm, and defaulting it would make
    it easy to read the wrong arm's rows.
    """
    parent = Path(cache_dir) / subdir / f"seed_{int(seed)}"
    if not parent.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(parent.glob("*.json")):
        rec = _load_json(path)
        row: dict = {
            "problem_id": int(rec["problem_id"]),
            "stratum": int(rec["stratum"]),
        }
        for key in _VLMPLAN_DIAGNOSTIC_KEYS:
            row[key] = rec.get(key)
        loop = rec.get("loop") or {}
        row["plans_per_round"] = loop.get("plans_per_round")
        attempts = rec.get("attempts") or []
        # Rounds actually used, recovered from the attempts' round tags (the fill entries
        # carry None). Reported as the exhaustion-depth distribution.
        used_rounds = [a["round"] for a in attempts if a.get("round") is not None]
        row["n_rounds_used"] = (max(used_rounds) + 1) if used_rounds else 0
        row["n_vlm_attempts"] = sum(1 for a in attempts if a.get("source") == "vlm")
        rows.append(row)
    return rows


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


def _pearson(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson (linear) correlation; NaN if either side is constant.

    Affine-invariant, so a per-episode Pearson r is unchanged by within-episode
    z-scoring — the per-problem-normalized length correlation the T0 table reports.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.std() == 0.0 or ya.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


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

    Returns ``{pearson, r2, eta2, within_frac, spearman, n_len, n}``:

    - ``pearson`` — linear correlation between score and length (sign = long-plan
      preference: positive ⇒ higher-scored, tried-earlier candidates are longer).
    - ``r2 = pearson²`` — the **linear** length-R² (fraction of score variance a
      straight-line fit on length explains). This is the headline T0 metric; unlike
      ``eta2`` it does not credit non-monotone length structure, so pair it with the
      learned-length-curve plot which does. NaN if scores are constant in length.
    - ``eta2`` — categorical-length R² (see :func:`_eta2`), retained as a secondary.
    - ``within_frac = 1 - eta2`` — share of variance that is within-length signal.
    - ``spearman`` — rank correlation between score and length.
    """
    sc = np.asarray(scores, dtype=float)
    ln = np.asarray(lengths, dtype=float)
    if sc.size != ln.size:
        raise ValueError(f"scores/lengths length mismatch: {sc.size} vs {ln.size}")
    eta2 = _eta2(sc, ln)
    within = float("nan") if np.isnan(eta2) else 1.0 - eta2
    pearson = _pearson(sc, ln)
    r2 = float("nan") if np.isnan(pearson) else pearson * pearson
    return {
        "pearson": pearson,
        "r2": r2,
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


_FIT_KEYS = ("pearson", "r2", "eta2", "within_frac", "spearman")


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

    Covers ``astar-dist``, ``PIGINet`` and the ``*-static`` rows of every SPECTRE
    family (seed-averaged); each record is ``{problem_id, stratum, method, pearson,
    r2, eta2, within_frac, spearman}``. A ``*-adaptive`` method has no static
    per-skeleton scores — its one-shot ranking is provably identical to its
    ``*-static`` twin (same checkpoint, empty context → c₀), so the notebook mirrors
    the static row for it and uses :func:`load_adaptive_ladder_records` for the
    realized-order view.
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
    for stat_m, stat_d, _adap_m, _adap_d in SPECTRE_FAMILIES:
        parent = cache_dir / stat_d
        if not parent.is_dir():  # a family may be absent (e.g. v1-only cache)
            continue
        by_pid = _spectre_static_length_fit_mean(parent, lengths_by_pid)
        for pid, (stratum, fit) in by_pid.items():
            records.append(
                {"problem_id": pid, "stratum": stratum, "method": stat_m, **fit}
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

    Static-score methods only (astar-dist, PIGINet, and each SPECTRE family's
    ``*-static`` seed-pooled). Each record is ``{method, length, mean_pos, n_eps}``
    with ``mean_pos`` in ``[0, 1]`` (0 = that length tier is tried first). Plots the
    *shape* of each method's learned length preference.
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
    for stat_m, stat_d, _adap_m, _adap_d in SPECTRE_FAMILIES:
        parent = cache_dir / stat_d
        if not parent.is_dir():  # a family may be absent (e.g. v1-only cache)
            continue
        acc = {}
        for seed_dir in sorted(parent.glob("seed_*")):
            _accumulate_positions(seed_dir, lengths_by_pid, acc)
        for length, positions in sorted(acc.items()):
            records.append(
                {
                    "method": stat_m,
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
    """Per-problem realized-order length-ladder for the ``*-adaptive`` methods.

    Reads the ``order`` field written by ``precompute_dd2d_cache.py`` (the
    sequence of attempted pool indices) and seed-averages :func:`length_ladder`,
    for every SPECTRE family's adaptive cache present on disk. A family whose
    adaptive dir is missing, or whose records carry no ``order`` (pre-trace cache),
    contributes nothing — so the notebook degrades gracefully. Each record is
    ``{problem_id, stratum, method, spearman, slope, n_steps, first_len, last_len}``.
    """
    cache_dir = Path(cache_dir)
    records: list[dict] = []
    for _stat_m, _stat_d, adap_m, adap_d in SPECTRE_FAMILIES:
        parent = cache_dir / adap_d
        if not parent.is_dir():
            continue
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
        records.extend(
            {
                "problem_id": pid,
                "stratum": per_stratum[pid],
                "method": adap_m,
                **{k: _safe_nanmean([lad[k] for lad in lads]) for k in _LADDER_KEYS},
            }
            for pid, lads in per_pid.items()
        )
    return records


# ---------------------------------------------------------------------------
# Per-stratum summary table. Lives here rather than in the script that first grew
# it (`experiments/spectre/spectre_v3_table.py`) so the marimo notebook can import
# it -- a notebook has no reliable path to a sibling script -- and so the one
# implementation stays CI-checked.
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    """Sample standard deviation; ``nan`` below two observations.

    ``nan`` rather than ``0.0`` on a single seed on purpose -- a zero would read as
    "this method is perfectly stable" when the truth is "nobody measured".
    """
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "--"
    return f"{mean:.2f}" if math.isnan(std) else f"{mean:.2f} ± {std:.2f}"


def build_table(records: list[dict]) -> tuple[list[str], list[list[str]], list[dict]]:
    """Return ``(header, rows, tidy)`` for the per-stratum FP table.

    The ``±`` is the spread ACROSS SEEDS of the per-stratum mean, not across problems:
    :func:`load_fp_records` averages a problem's FP over seeds before returning it, so a
    std taken downstream of *that* is the across-problem spread of a seed-mean. Feed this
    :func:`load_fp_records_per_seed`. With one seed cached the column reads ``--`` and
    populates itself once more seeds exist.
    """
    strata = sorted({r["stratum"] for r in records})
    by_method_seed: dict[tuple[str, object, object], list[float]] = defaultdict(list)
    for r in records:
        by_method_seed[(r["method"], r["seed"], r["stratum"])].append(r["fp"])
        by_method_seed[(r["method"], r["seed"], "ALL")].append(r["fp"])

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in records)]
    methods += sorted({r["method"] for r in records} - set(methods))

    header = ["method", "seeds", "ALL"] + [f"s{s}" for s in strata]
    rows: list[list[str]] = []
    tidy: list[dict] = []
    for method in methods:
        seeds = sorted(
            {r["seed"] for r in records if r["method"] == method},
            key=lambda s: (s is None, s),
        )
        row = [method, "-" if seeds == [None] else str(len(seeds))]
        for stratum in ["ALL"] + list(strata):
            # per seed: mean over that seed's problems; then spread across seeds
            per_seed = [
                _mean(by_method_seed[(method, s, stratum)])
                for s in seeds
                if by_method_seed[(method, s, stratum)]
            ]
            mean, std = _mean(per_seed), _std(per_seed)
            row.append(_fmt(mean, std))
            tidy.append(
                {
                    "method": method,
                    "stratum": stratum,
                    "n_seeds": len(per_seed),
                    "mean_fp": mean,
                    "std_fp_across_seeds": std,
                }
            )
        rows.append(row)
    return header, rows, tidy


def render_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Render :func:`build_table`'s output as a GitHub-flavoured markdown table."""
    widths = [
        max(len(header[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(header))
    ]
    out = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out += [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(out)
