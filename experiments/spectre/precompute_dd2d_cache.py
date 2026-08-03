"""Precompute + cache per-problem test scores for the DD2D method comparison.

Runs each method once on the DD2D test split and writes the *raw per-skeleton
scores* per problem so ``experiments/spectre/compare_methods.py`` can load
them and derive rollout-FPs (and other metrics) without re-running any inference.

Cache layout under ``data/spectre/derived/<env_variant>/compare_cache/`` (one JSON
per problem, keyed by the integer seed = ``problem_id``; resumable — existing files
are skipped unless ``--force``). ``--env-variant`` selects the collection (default
``dd2d_v2``); it repoints the test split, vocab, checkpoints, PIGINet artifacts, and
the cache dir together:

    astar/<pid>.json                     {problem_id, stratum, scores, labels}
    piginet/<pid>.json                   {problem_id, stratum, scores, labels}
        (PIGINet trained with BCE = the paper baseline; see piginet/train.py)
    spectre_static/seed_<s>/<pid>.json   {problem_id, stratum, scores, labels}
    spectre_adaptive/seed_<s>/<pid>.json {problem_id, stratum, fp, order,
                                          step_scores, step_dead}
        (``order`` = the realized attempt sequence of pool indices, until first
        success; consumed by the notebook's realized-order + length-ladder views.
        ``step_scores[t]`` = the raw (K,) logits the step-t pick was made from —
        before the tried-mask and before any demotion offset — so the planner
        inspector can show what the adaptive ranker thought at each step without
        running inference. ``step_dead[t]`` = the provably-dead indices in force at
        step t; always empty for v1, which has no proof-demotion.)
    spectre2_static/seed_<s>/<pid>.json  {problem_id, stratum, scores, labels}
        (SPECTRE v2.2 empty-context logits)
    spectre2_adaptive/seed_<s>/<pid>.json{problem_id, stratum, fp, order,
                                          step_scores, step_dead}
        (SPECTRE v2.2 deployed_rollout, observed proof-demotion)
    spectre3_static/seed_<s>/<pid>.json  {problem_id, stratum, scores, labels}
    spectre3_adaptive/seed_<s>/<pid>.json{problem_id, stratum, fp, order,
                                          step_scores, step_dead}
        (SPECTRE v3 deployed ranker: observed coverage/waste + record tokens + the
        record state delta. **No proof-demotion** -- cut from the method 2026-07-30, so
        nothing outside the network touches the ordering. ``step_dead`` is still recorded
        and now reads as "what a proof would have demoted".)
    abl_<arm>_adaptive/seed_<s>/<pid>.json
        (one dir per v3 ablation arm -- see ``_V3_ARMS``; adaptive shape only)
    abl_with_demotion_adaptive/…, abl_floor_with_demotion_adaptive/…
        (the same checkpoints re-scored with the proof-demotion offset switched back ON --
        see ``_V3_DEMOTION_ARMS``. Demotion is OFF in the deployed method as of
        2026-07-30, so this is the ablation; pair them with ``spectre3_adaptive`` and
        ``abl_nocov_norec_adaptive`` respectively)
    spectre_lenctx/seed_<s>/<pid>.json   {problem_id, stratum, fp, order}
        (T1 length-only-context intervention: adaptive rollout with identity-
        scrambled same-length failure contexts; fp = mean over surrogate draws)

Usage::

    python experiments/spectre/precompute_dd2d_cache.py            # default methods
    python experiments/spectre/precompute_dd2d_cache.py --force
    python experiments/spectre/precompute_dd2d_cache.py --methods piginet spectre2
    python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v3 --force
    # v3 + its ablation arms (dd2d_v4 is the only collection with v3 checkpoints)
    python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 \
        --methods spectre3

This is a bridge driver: it imports the vendored ``piginet.eval`` scorer, so it is
excluded from strict mypy/pylint like the marimo notebook it feeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from alphatamp.approaches.spectre import eda
from alphatamp.approaches.spectre.compare import stratum_of
from alphatamp.approaches.spectre.inference import (
    init_inference_state,
    load_checkpoint,
    load_prior_for_checkpoint,
)
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]
DD2D = REPO / "src" / "alphatamp" / "approaches" / "spectre" / "envs" / "dd2d"
# StickButton2D keeps its PIGINet artifacts beside its other derived data rather than
# in a vendored env tree, so the paths below need the derived root at module scope.
DERIVED_ROOT = REPO / "data" / "spectre" / "derived"
SEEDS = [0, 1, 2]
DEFAULT_ENV_VARIANT = "dd2d_v2"

# PIGINet reads its own native DD2D JSON (not the SPECTRE EpisodeRecord pickles) and
# keeps a CLIP cache + BCE checkpoint whose locations were operator-chosen per
# collection, so they cannot be derived from ``env_variant`` by a string swap — they
# are pinned per variant here. Add an entry to onboard a new collection.
_PIGINET_PATHS = {
    "dd2d_v2": {
        "ckpt": DD2D / "out_dd2d" / "piginet_bce" / "ckpt.pt",
        "data": DD2D / "data" / "dd2d" / "raw_v2",
        "cache": DD2D / "out_dd2d" / "clip_cache_v2",
    },
    "dd2d_v3": {
        "ckpt": DD2D / "out_dd2d" / "piginet_bce_v3" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v3",  # the repo-root re-collection
        "cache": DD2D / "out_dd2d" / "clip_cache_v3",
    },
    # dd2d_v4 is the first collection where PIGINet has a real seed axis: `train.py`
    # gained `--seed` on 2026-07-28, so `{seed}` appears in the checkpoint path and the
    # cache is written per seed. Earlier variants have one deterministic run each and
    # keep their flat, seedless cache layout -- the reader detects which it is looking at.
    "dd2d_v4": {
        "ckpt": DD2D / "out_dd2d" / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4",
        "cache": DD2D / "out_dd2d" / "clip_cache_v4",
    },
    # StickButton2D, 2026-08-01. `data` is the SPECTRE data root rather than a record
    # tree: this collection has no PIGINet JSON on disk, so `SB2DDomain` builds the
    # examples from the same `EpisodeRecord` pickles SPECTRE trains on -- which is what
    # makes the two methods' labels identical by construction rather than by agreement.
    # `domain` selects the adapter; absent, the reader keeps its DD2D default.
    "stickbutton2d_v1": {
        "ckpt": DERIVED_ROOT / "stickbutton2d_v1" / "piginet_bce_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "spectre",
        "cache": DERIVED_ROOT / "stickbutton2d_v1" / "clip_cache",
        "domain": "stickbutton2d",
    },
    # StickButton2D with kinder-rendered PIGINet crops (2026-08-02). Same records as
    # stickbutton2d_v1 -- SPECTRE is image-free and grafts from v1; only PIGINet's crops
    # differ, so its checkpoint and CLIP cache repoint to the `_kinder` derived subdir.
    # The factory in `make_sb2d_domain` reads the kinder PNGs for this variant. See
    # experiments/spectre/sb2d_render_convert.py.
    "stickbutton2d_v1_kinder": {
        "ckpt": DERIVED_ROOT
        / "stickbutton2d_v1_kinder"
        / "piginet_bce_s{seed}"
        / "ckpt.pt",
        "data": REPO / "data" / "spectre",
        "cache": DERIVED_ROOT / "stickbutton2d_v1_kinder" / "clip_cache",
        "domain": "stickbutton2d",
    },
}

# Which SPECTRE-v2 deployed checkpoint a collection uses. The training run name encodes
# the config: the standard `_ov` model is evidence+prior+overlap, but the short-first
# *prior* over-biases cross-length ordering on the easier grasp-fixed dd2d_v3 (it buries
# the long s3 feasibles), so dd2d_v3 uses the no-prior `evidence_ov` model, chosen by
# val deployed-FP (`decisions.md` 2026-07-25). Pinned per variant here.
_V2_CKPT_SUBDIR = {
    "dd2d_v2": "checkpoints_v2_evidence_prior_ov",
    "dd2d_v3": "checkpoints_v2_evidence_ov",
    # dd2d_v4 inherits v3's config (no prior): it is the same domain and difficulty, and
    # this row is the v3 migration's *yardstick* -- the deployed v2.2 model every v3 gate
    # is measured against -- so it must be the same recipe, not a re-tuned one.
    "dd2d_v4": "checkpoints_v2_evidence_ov",
}

# SPECTRE v3 arms: cache sub-dir prefix -> checkpoint sub-dir ({seed} substituted).
#
# `spectre3` is the deployed method and is the only one that also gets a *static* row.
# The rest are the ablation, all held at the SAME matched setting -- `--overlap-mode
# jaccard`, no `--aggregate-records`, no `--evidence-attn` -- so the 2x2 varies only what
# it names. Deployed differs from `abl_cov_rec` by exactly those two consumption
# switches, which is why the headline record-token number is "tokens + machinery" and the
# 2x2's is tokens alone.
#
# Every arm here post-dates G6b, so all were selected by the uncensored whole-split
# selector. Mixing in a `g6_*` arm (censored at 30 attempts, 50 val episodes) would
# compare checkpoints chosen by two different instruments -- `_assert_same_selector`
# enforces this rather than trusting the directory name.
_V3_ARMS: dict[str, str] = {
    # The deployed model. Repointed 2026-07-28 from `checkpoints_v3_v3final_s{seed}` to
    # the state-delta arm, which is now the deployed configuration (`decisions.md`
    # 2026-07-28). **Re-cache with `--force`**: `spectre3_{static,adaptive}/seed_0` was
    # written from the pre-delta checkpoint and `_dir_complete` skips any full directory,
    # so without it seed 0 stays pre-delta while seeds 1-2 are the delta model -- one
    # method row silently mixing two generations.
    # 2026-07-31: repointed to the unified coverage/waste definition, which is now the
    # deployed default. 5.78 +/- 0.10 against the previous arm's 7.44 +/- 0.76 --
    # -1.66 FP, CI [-2.71, -0.71], every seed beating every baseline seed. One directory
    # holding all three seeds, so no `{seed}` substitution here.
    # **Re-cache with `--force`**: `_dir_complete` skips a full directory, so without it
    # the row silently keeps the previous definition's rollout.
    "spectre3": "checkpoints_v3_unified",
    # ^ DD2D's deployed dir. Per-variant overrides live in `_V3_ARM_OVERRIDES` below:
    # a second environment trains the same arm under a different run name, and silently
    # scoring the wrong directory (or, as happened first, skipping the arm entirely with
    # a "missing checkpoint" line buried in a log) is not a failure worth repeating.
    # coverage x record-tokens 2x2
    # NOT `checkpoints_v3_p8_cov_final_s{seed}`, despite autorun_decisions A15 naming it
    # "the clean 3-seed re-run": all three of those runs stopped at **epoch 5 of 30**, so
    # their best.pt is a mid-training stub that scores 26.97 (s0 36.64, where every other
    # arm gets 0.00). Retrained here at identical flags. See notebook.md 2026-07-27.
    "abl_cov_rec": "checkpoints_v3_abl_cov_rec",
    "abl_cov_norec": "checkpoints_v3_norec_p9_cov_norec",
    "abl_nocov_rec": "checkpoints_v3_g8_jac",
    "abl_nocov_norec": "checkpoints_v3_norec_abl_jac_norec_nocov",
    # coverage vs waste, split apart by --coverage-mode
    "abl_cov_only": "checkpoints_v3_abl_cov_only",
    "abl_waste_only": "checkpoints_v3_abl_waste_only",
}
#: Per-variant checkpoint-dir overrides for the v3 arms. StickButton2D trained the
#: deployed config with no `--out-suffix`, so its dir is the bare `checkpoints_v3`;
#: DD2D's carries the `_unified` tag from the 2026-07-31 definition change.
_V3_ARM_OVERRIDES: dict[str, dict[str, str]] = {
    # StickButton2D's arms were trained by `spectre_sweep.py --preset sb2dabl`, which
    # writes one directory per (arm, seed) -- hence `{seed}` -- and `train_v3` prefixes
    # `_norec` onto any `--no-records` run, so two of the six carry it. DD2D's arms
    # predate that sweep and use their own historical names, which is exactly why this
    # map is per-variant rather than a string rule.
    "stickbutton2d_v1": {
        "spectre3": "checkpoints_v3",
        "abl_cov_rec": "checkpoints_v3_abl_cov_rec_s{seed}",
        "abl_cov_norec": "checkpoints_v3_norec_abl_cov_norec_s{seed}",
        "abl_nocov_rec": "checkpoints_v3_abl_nocov_rec_s{seed}",
        "abl_nocov_norec": "checkpoints_v3_norec_abl_nocov_norec_s{seed}",
        "abl_cov_only": "checkpoints_v3_abl_cov_only_s{seed}",
        "abl_waste_only": "checkpoints_v3_abl_waste_only_s{seed}",
    },
}


def _v3_arm_dir(arm: str, env_variant: str) -> str:
    """Checkpoint sub-dir for one v3 arm on one collection."""
    return _V3_ARM_OVERRIDES.get(env_variant, {}).get(arm, _V3_ARMS[arm])


# Deploy-time diagnostic: the deployed checkpoint with its evidence memory emptied at
# every step. Not a method result -- it is a train/deploy mismatch on purpose, to
# separate "training with records damaged the weights" from "the model ignores them".
_V3_SUPPRESS_ARMS: dict[str, str] = {
    "abl_suppress_records": "checkpoints_v3_v3final_s{seed}",
}
# The proof-demotion ablation, INVERTED on 2026-07-30: demotion was cut from the deployed
# method, so every arm above now runs without it and the *diagnostic* is switching it back
# ON. Prices what the deployed model gives up by being purely learned.
#
# These need their own registry rather than `--v3-arm` because `--v3-arm` carries only
# `prefix:ckpt_subdir` and leaves `apply_demotion` at its default: pointing it at the
# deployed checkpoint would write a **byte-identical copy of `spectre3_adaptive`** under a
# name asserting demotion is on, i.e. render the ablation as exactly 0.00 with nothing
# looking wrong. Pair each entry with its demotion-OFF twin:
#   abl_with_demotion       <-> spectre3          (the deployed model)
#   abl_floor_with_demotion <-> abl_nocov_norec   (jaccard only, no coverage, no tokens)
_V3_DEMOTION_ARMS: dict[str, str] = {
    "abl_with_demotion": "checkpoints_v3_v3delta_s{seed}",
    "abl_floor_with_demotion": "checkpoints_v3_norec_abl_jac_norec_nocov",
}

# Env-variant-dependent path globals. The cache functions read these as module globals
# at call time, so ``main`` rebinds them via ``_configure_paths`` from ``--env-variant``
# before dispatching. The literal defaults below are the historical ``dd2d_v2`` values
# (kept byte-identical so importing the module is unchanged); ``_configure_paths``
# overrides them and, unlike these, derives ``N_PROBLEMS`` from the real test split.
ENV_VARIANT = DEFAULT_ENV_VARIANT
SPECTRE_TEST = REPO / "data" / "spectre" / "raw" / DEFAULT_ENV_VARIANT / "test"
VOCAB_PATH = (
    REPO / "data" / "spectre" / "derived" / DEFAULT_ENV_VARIANT / "train_vocab.json"
)
# SPECTRE v1 (abstract-only) checkpoints; SPECTRE v2.2 deployed (observed proof-
# demotion = evidence + prior + overlap). Both ship seed_0 only → 1-seed dev.
CKPT_DIR = REPO / "data" / "spectre" / "checkpoints" / DEFAULT_ENV_VARIANT
V2_CKPT_DIR = (
    REPO
    / "data"
    / "spectre"
    / _V2_CKPT_SUBDIR[DEFAULT_ENV_VARIANT]
    / DEFAULT_ENV_VARIANT
)
PIGINET_CKPT = _PIGINET_PATHS[DEFAULT_ENV_VARIANT]["ckpt"]
PIGINET_DATA = _PIGINET_PATHS[DEFAULT_ENV_VARIANT]["data"]
PIGINET_CACHE = _PIGINET_PATHS[DEFAULT_ENV_VARIANT]["cache"]
# Which adapter supplies PIGINet's vocabulary, normalisers and split.
# `None` = DD2D, so the historical defaults below stay literal.
PIGINET_DOMAIN = _PIGINET_PATHS[DEFAULT_ENV_VARIANT].get("domain")
CACHE_DIR = (
    REPO / "data" / "spectre" / "derived" / DEFAULT_ENV_VARIANT / "compare_cache"
)
N_PROBLEMS = 140


def _count_test_problems(test_dir: Path) -> int:
    """Test-split episode count — drives ``_dir_complete`` and ``meta.json``.

    Falls back to the historical 140 if the split is not on disk yet (e.g. importing the
    module before the collection exists).
    """
    eps = test_dir / "episodes"
    n = len(list(eps.glob("*.pkl.gz"))) if eps.is_dir() else 0
    return n or 140


def _configure_paths(env_variant: str) -> None:
    """(Re)bind every env-variant-dependent module global from ``env_variant``."""
    global ENV_VARIANT, SPECTRE_TEST, VOCAB_PATH, CKPT_DIR, V2_CKPT_DIR
    global PIGINET_CKPT, PIGINET_DATA, PIGINET_CACHE, PIGINET_DOMAIN
    global CACHE_DIR, N_PROBLEMS
    known = set(_V2_CKPT_SUBDIR) | set(_PIGINET_PATHS)
    if env_variant not in known:
        raise SystemExit(
            f"unknown --env-variant {env_variant!r}; known: {sorted(known)} "
            "(add a _V2_CKPT_SUBDIR or _PIGINET_PATHS entry to onboard a collection)"
        )
    # PIGINet is optional per variant: it trains on the *native* DD2D JSON with its own
    # CLIP cache, so onboarding a collection for the SPECTRE methods does not
    # automatically give it a PIGINet row. Missing paths become None and only fail if
    # `--methods piginet` actually asks for it.
    piginet = _PIGINET_PATHS.get(env_variant, {})
    ENV_VARIANT = env_variant
    SPECTRE_TEST = REPO / "data" / "spectre" / "raw" / env_variant / "test"
    VOCAB_PATH = (
        REPO / "data" / "spectre" / "derived" / env_variant / "train_vocab.json"
    )
    CKPT_DIR = REPO / "data" / "spectre" / "checkpoints" / env_variant
    # A collection with no SPECTRE v2.2 checkpoint (StickButton2D: v2 was scoped out) gets
    # `None` rather than a KeyError, so `--methods spectre2` fails on that method alone
    # instead of the whole driver refusing to start.
    _v2_sub = _V2_CKPT_SUBDIR.get(env_variant)
    V2_CKPT_DIR = (
        None if _v2_sub is None else REPO / "data" / "spectre" / _v2_sub / env_variant
    )
    PIGINET_CKPT = piginet.get("ckpt")
    PIGINET_DATA = piginet.get("data")
    PIGINET_CACHE = piginet.get("cache")
    PIGINET_DOMAIN = piginet.get("domain")
    CACHE_DIR = REPO / "data" / "spectre" / "derived" / env_variant / "compare_cache"
    N_PROBLEMS = _count_test_problems(SPECTRE_TEST)


def _write(path: Path, obj: dict, force: bool) -> bool:
    """Atomically write ``obj`` unless the file already exists.

    Returns True if written.
    """
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)
    return True


def _dir_complete(d: Path) -> bool:
    return d.is_dir() and len(list(d.glob("*.json"))) >= N_PROBLEMS


def _round_rows(rows) -> list[list[float | None]]:
    """Round a per-step score matrix to 4 dp, non-finite entries -> ``None``.

    4 dp matches the geometry-storage convention and roughly halves the JSON, well below
    any display or ordering tolerance. The v2 model masks its own failure context, so a
    step's row carries ``-inf`` for every already-attempted candidate; ``-inf`` is not
    strict JSON, and ``null`` says the right thing anyway ("not available at this
    step").
    """
    return [
        [None if not math.isfinite(float(x)) else round(float(x), 4) for x in row]
        for row in rows
    ]


def _static_order(scores) -> list[int]:
    """Descending-score attempt order for a static ranker (stable ties)."""
    return [int(i) for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable")]


def _refine_seconds(ep, order) -> float:
    """Refinement wall-clock (s) summed along ``order`` up to and including the first
    success, reused from the stored per-candidate ``refinement_wall_clock_s`` (dd2d_v3/v4;
    0.0 on collections without it). Every method sums the *same* per-candidate times over its
    own order, so the cross-method comparison is fair even though the absolute seconds are a
    within-collection relative measure."""
    total = 0.0
    for idx in order:
        o = ep.outcomes[idx]
        total += float(o.refinement_wall_clock_s or 0.0)
        if o.outcome == "success":
            break
    return round(total, 6)


# Deployed per-candidate refinement-abandonment cap (seconds). A skeleton is refined for at
# most this long before the deployment moves on; a feasible candidate over the cap is
# abandoned (its outcome no longer counts as a stopping success). ~4.5x the feasible p95
# (0.44s on dd2d_v4), so only genuine near-feasible outliers are cut. The §2b wall-clock
# table reports every pool-ranking method under it; the uncapped fields stay for the FP
# headline. See decisions/07 2026-08-02.
REFINE_CAP_S = 2.0


def _fp_and_refine_capped(ep, order, cap: float) -> tuple[float, float]:
    """``(fp_capped, refine_s_capped)`` for a **fixed-order** method under the cap.

    Walks the score-order (independent of refine time, so no model re-run is needed) and
    stops at the first candidate that refines *within* the cap. A feasible candidate whose
    stored ``refinement_wall_clock_s`` exceeds ``cap`` is abandoned -- charged ``cap`` and
    skipped -- exactly as the deployed refiner would. ``fp_capped`` is the failed-attempt
    count before that stop; ``refine_s_capped`` sums ``min(t, cap)`` up to and including it.
    If nothing refines within the cap (every feasible candidate is slow -- does not happen
    on dd2d_v4, where every problem keeps a sub-0.25s feasible candidate), the whole order
    is charged and ``fp_capped == len(order)``.
    """
    total = 0.0
    for k, idx in enumerate(order):
        o = ep.outcomes[idx]
        t = float(o.refinement_wall_clock_s or 0.0)
        total += min(t, cap)
        if o.outcome == "success" and t <= cap:
            return float(k), round(total, 6)
    return float(len(order)), round(total, 6)


def _feasibility_at_risk(cap: float) -> int | None:
    """Count test problems whose *every* feasible candidate refines slower than ``cap``.

    These are the only problems a per-candidate cap could turn from solved into
    censored. ``None`` when the split carries no per-candidate times (nothing to check).
    """
    n_at_risk = 0
    saw_times = False
    for ep in eda.load_split_episodes(SPECTRE_TEST).episodes:
        feas = [
            float(o.refinement_wall_clock_s or 0.0)
            for o in ep.outcomes
            if o.outcome == "success"
        ]
        if any(o.refinement_wall_clock_s for o in ep.outcomes):
            saw_times = True
        if feas and min(feas) > cap:
            n_at_risk += 1
    return n_at_risk if saw_times else None


def _measure_plan_gen(per_stratum: int = 3) -> dict[str, float]:
    """Per-stratum abstract-plan-generation time (s), shared by all pool-ranking
    methods.

    Not stored at collection, so measured here: regenerate a few problems per stratum
    from the stored ``gen_params`` + seed and time the astar top-k pool enumeration
    (``make_dd2d_planner(prefer='pyperplan', search='astar', heuristic='dist').plan``) —
    the step that produces the ranked candidate pool the models score. A regenerated
    proxy (DD2D's generator is PYTHONHASHSEED-dependent), used as a representative per-
    stratum constant. DD2D only; ``{}`` (and a 0 fallback in the notebook) elsewhere or
    on failure.
    """
    if not ENV_VARIANT.startswith("dd2d"):
        return {}
    try:
        from collections import defaultdict

        from alphatamp.approaches.spectre.envs.dd2d.dd2d.planning import (
            make_dd2d_planner,
        )
        from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import (
            generate_dd2d_problem,
        )
    except Exception as e:  # pragma: no cover - env not importable
        print(f"[plan_gen] unavailable, skipping: {type(e).__name__}: {e}")
        return {}

    groups: dict[int, list] = defaultdict(list)
    for ep in eda.load_split_episodes(SPECTRE_TEST).episodes:
        groups[stratum_of(int(ep.provenance.problem_id))].append(ep)
    planner = make_dd2d_planner(prefer="pyperplan", search="astar", heuristic="dist")
    out: dict[str, float] = {}
    for s in sorted(groups):
        times: list[float] = []
        for ep in groups[s][:per_stratum]:
            gp = dict(ep.provenance.gen_params.get("gen_params", {}))
            gp["certify"] = (
                False  # deployment does not certify; time only pool production
            )
            seed = int(
                getattr(ep.provenance, "problem_seed", 0) or ep.provenance.problem_id
            )
            try:
                problem = generate_dd2d_problem(seed=seed, **gp)
                t0 = time.perf_counter()
                planner.plan(problem, len(ep.skeleton_pool))
                times.append(time.perf_counter() - t0)
            except Exception as e:  # keep going; a per-problem failure is not fatal
                print(f"[plan_gen] s{s} seed{seed} skipped: {type(e).__name__}: {e}")
        if times:
            out[str(s)] = round(sum(times) / len(times), 6)
            print(f"[plan_gen] s{s}: {out[str(s)]:.4f}s (n={len(times)})", flush=True)
    return out


def cache_astar(force: bool) -> None:
    """astar-dist: planner enumeration order (score = -plan_idx)."""
    out = CACHE_DIR / "astar"
    if _dir_complete(out) and not force:
        print("[astar] complete; skipping")
        return
    test = eda.load_split_episodes(SPECTRE_TEST)
    n = 0
    for ep in test.episodes:
        pid = int(ep.provenance.problem_id)
        labels = [1 if o.outcome == "success" else 0 for o in ep.outcomes]
        scores = [float(-j) for j in range(len(ep.outcomes))]  # ascending plan_idx
        order = _static_order(scores)
        fp_cap, refine_cap = _fp_and_refine_capped(ep, order, REFINE_CAP_S)
        n += _write(
            out / f"{pid}.json",
            {
                "problem_id": pid,
                "stratum": stratum_of(pid),
                "scores": scores,
                "labels": labels,
                "refine_s": _refine_seconds(ep, order),
                "refine_s_capped": refine_cap,
                "fp_capped": fp_cap,
                "infer_s": 0.0,  # astar-dist = planner default order, no model inference
            },
            force,
        )
    print(f"[astar] wrote {n} problems -> {out}")


@torch.no_grad()
def cache_piginet(force: bool, device: str) -> None:
    """PIGINet (BCE-trained, paper baseline): fresh inference; cache logits + labels.

    **Seeded only where the checkpoint path says so.** A variant whose ``ckpt`` contains
    ``{seed}`` has one trained model per seed and gets the per-seed layout
    (``piginet/seed_<n>/``); a variant without it has a single deterministic run and
    keeps the flat ``piginet/<pid>.json`` layout it was written with. PIGINet only
    gained a ``--seed`` flag on 2026-07-28, so dd2d_v2/v3 are legitimately seedless --
    inventing a ``seed_0`` directory for them would claim a seed axis that was never
    sampled.
    """
    if PIGINET_CKPT is None:
        raise SystemExit(
            f"--methods piginet requested but {ENV_VARIANT!r} has no _PIGINET_PATHS "
            "entry. PIGINet trains on the native DD2D JSON with its own CLIP cache, so a "
            "new collection needs it retrained first (piginet/train.py), then "
            "an entry added here."
        )
    seeded = "{seed}" in str(PIGINET_CKPT)
    for seed in SEEDS if seeded else [None]:
        out = (
            CACHE_DIR / "piginet"
            if seed is None
            else CACHE_DIR / "piginet" / f"seed_{seed}"
        )
        tag = "piginet" if seed is None else f"piginet seed {seed}"
        if _dir_complete(out) and not force:
            print(f"[{tag}] complete; skipping")
            continue
        ckpt = Path(str(PIGINET_CKPT).replace("{seed}", str(seed)))
        if not ckpt.is_file():
            print(f"[{tag}] !! missing {ckpt}; skipping", flush=True)
            continue
        # Local import: vendored piginet stack (pulls in open_clip / CLIP).
        from alphatamp.approaches.spectre.piginet.eval import score_split

        print(f"[{tag}] running fresh inference on test split ...")
        domain = None
        if PIGINET_DOMAIN == "stickbutton2d":
            from alphatamp.approaches.spectre.piginet.sb2d_adapter import (
                make_sb2d_domain,
            )

            # Factory picks the crop source by variant (kinder PNGs vs schematic).
            domain = make_sb2d_domain(str(PIGINET_DATA), ENV_VARIANT)
        _t0 = time.perf_counter()
        rows, _thr, _temp = score_split(
            str(ckpt),
            str(PIGINET_DATA),
            str(PIGINET_CACHE),
            "test",
            device=device,
            domain=domain,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        # PIGINet is a static predictor: it scores every candidate up front, then ranks,
        # then refines -- so the whole split's scoring IS its inference-to-first-success.
        # Per-problem infer = total / n. Caveat: CLIP image features are read from the
        # clip_cache, so this is the BCE-head cost, not a from-scratch CLIP encode.
        by_pid: dict[str, list[tuple[int, int, float]]] = {}
        for pid, _stratum, plan_idx, _length, label, score in rows:
            by_pid.setdefault(pid, []).append((int(plan_idx), int(label), float(score)))
        infer_per_problem = round((time.perf_counter() - _t0) / max(1, len(by_pid)), 6)
        ep_by_pid = {
            int(e.provenance.problem_id): e
            for e in eda.load_split_episodes(SPECTRE_TEST).episodes
        }
        n = 0
        for pid_str, triples in by_pid.items():
            triples.sort(key=lambda t: t[0])  # order by plan_idx
            pid = int(pid_str.split("_s")[-1])
            scores = [t[2] for t in triples]
            ep = ep_by_pid.get(pid)
            order = _static_order(scores)
            fp_cap, refine_cap = (
                _fp_and_refine_capped(ep, order, REFINE_CAP_S)
                if ep is not None
                else (0.0, 0.0)
            )
            n += _write(
                out / f"{pid}.json",
                {
                    "problem_id": pid,
                    "stratum": stratum_of(pid),
                    "scores": scores,
                    "labels": [t[1] for t in triples],
                    "refine_s": (_refine_seconds(ep, order) if ep is not None else 0.0),
                    "refine_s_capped": refine_cap,
                    "fp_capped": fp_cap,
                    "infer_s": infer_per_problem,
                },
                force,
            )
        print(f"[{tag}] wrote {n} problems -> {out}")


@torch.no_grad()
def cache_spectre(force: bool, device: str) -> None:
    """SPECTRE adaptive (rollout FP) + static (empty-context logits) per seed."""
    vocab = Vocab.from_json(VOCAB_PATH)
    # RAW: `init_inference_state` canonicalizes, so a pre-canonicalized split would be
    # canonicalized twice -- see `_RawSplit`. This was wrong until 2026-07-27; the v1
    # rows in any cache built before then are stale by ~1.5 FP and need `--force`.
    test = _RawSplit(SPECTRE_TEST)
    for seed in SEEDS:
        adir = CACHE_DIR / "spectre_adaptive" / f"seed_{seed}"
        sdir = CACHE_DIR / "spectre_static" / f"seed_{seed}"
        need_a = force or not _dir_complete(adir)
        need_s = force or not _dir_complete(sdir)
        if not need_a and not need_s:
            print(f"[spectre seed {seed}] complete; skipping")
            continue
        ck = CKPT_DIR / f"seed_{seed}" / "best.pt"
        model = load_checkpoint(ck, vocab, device=device)
        prior = load_prior_for_checkpoint(ck)

        if need_a:
            # Traced variant: same loop/cost as spectre_evaluate, but also returns
            # the realized attempt order per problem so the notebook's T0 length
            # ladder (does adaptive climb to longer plans as it fails?) needs no
            # re-run. ``fp`` is byte-identical to the untraced path.
            res, traces = eda.spectre_evaluate_traced(
                test,
                model,
                vocab,
                attempt_budget=200,
                prior=prior,
                device=device,
                freeze_context=False,
            )
            na = 0
            for pid, att, trace in zip(res.problem_ids, res.attempts, traces):
                pid = int(pid)
                na += _write(
                    adir / f"{pid}.json",
                    {
                        "problem_id": pid,
                        "stratum": stratum_of(pid),
                        "fp": float(att) - 1.0,
                        "order": [int(cs.idx) for cs in trace],
                        "step_scores": _round_rows(
                            [cs.scores or [] for cs in trace],
                        ),
                        # v1 has no proof-demotion — recorded explicitly (as empty
                        # dead sets) rather than omitted, so the notebook can tell
                        # "no demotion in this method" from "field missing".
                        "step_dead": [[] for _ in trace],
                    },
                    force,
                )
            print(f"[spectre-adaptive seed {seed}] wrote {na} -> {adir}")

        if need_s:
            ns = 0
            for ep in test.episodes:
                pid = int(ep.provenance.problem_id)
                state = init_inference_state(
                    model, ep, vocab, prior=prior, device=device
                )
                dim = state.e_S.size(-1)
                f_emb = torch.zeros(
                    1, 1, dim, device=state.e_S.device, dtype=state.e_S.dtype
                )
                f_mask = torch.zeros(1, 1, dtype=torch.bool, device=state.e_S.device)
                c = model.encode_context(f_emb, f_mask)
                logits = model.score(
                    state.e_S.unsqueeze(0),
                    c,
                    state.priors.unsqueeze(0),
                    prior_dropout=False,
                )[0]
                labels = [1 if o.outcome == "success" else 0 for o in ep.outcomes]
                ns += _write(
                    sdir / f"{pid}.json",
                    {
                        "problem_id": pid,
                        "stratum": stratum_of(pid),
                        "scores": [float(x) for x in logits.tolist()],
                        "labels": labels,
                    },
                    force,
                )
            print(f"[spectre-static seed {seed}] wrote {ns} -> {sdir}")


class _RawSplit:
    """``eda.LoadedSplit``-shaped view over **un-canonicalized** episodes.

    ``eda.load_split_episodes`` canonicalizes on load, which is right for the EDA
    baselines (they key on canonical skeletons) but wrong for anything that then calls a
    tensorizer, because ``build_v2_example`` / ``build_v3_example`` /
    ``inference.init_inference_state`` all canonicalize again and
    ``canonicalize_episode`` is not idempotent. Double canonicalization silently changes
    the object->tag binding relative to training, which loads raw.

    **Every model cache function must load through this**, v1 included. Measured on
    dd2d_v3, feeding v1 pre-canonicalized episodes moved its mean FP 21.41 -> 22.93 and
    changed per-problem FP on 39/100 problems; the same defect is why the dd2d_v3 v2
    number (13.68) was retracted (``decisions.md`` 2026-07-26). Training loads raw, so
    raw is what makes evaluation match training.

    Exposes ``.episodes`` -- the only attribute the cache functions and
    ``eda.spectre_evaluate_traced`` (via ``_trainable_episodes``) actually read.
    """

    def __init__(self, split_dir: Path) -> None:
        from alphatamp.approaches.spectre.io import list_episodes, load_episode

        self.episodes = [load_episode(p) for p in list_episodes(split_dir)]


def _load_v2_model(ckpt: Path, vocab: Vocab, device: str):
    """Rebuild a trained SpectreV2Model from a checkpoint.

    The checkpoint's ``cfg`` records ``use_prior`` / ``use_overlap`` (the deployed
    ``_ov`` checkpoint has both), which size the scorer's extra inputs — a strict
    ``load_state_dict`` fails unless the model is reconstructed with them.
    """
    from alphatamp.approaches.spectre.model_v2 import (
        N_OVERLAP,
        N_PRIOR,
        SpectreV2Model,
    )

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = SpectreV2Model(
        n_ops=int(ck["n_ops"]),
        max_arity=vocab.max_operator_arity,
        max_tags=int(cfg["max_tags"]),
        n_overlap_feats=N_OVERLAP if cfg.get("use_overlap") else 0,
        n_prior_feats=N_PRIOR if cfg.get("use_prior") else 0,
        dropout_p=0.0,
    )
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model


@torch.no_grad()
def cache_spectre2(force: bool, device: str) -> None:
    """SPECTRE v2.2 static (empty-context logits) + adaptive (deployed_rollout).

    Two deployment modes of the same ``_ov`` checkpoint (evidence + prior + overlap,
    observed proof-demotion). Static ranks the pool once at ``F=∅``; adaptive is the
    full deployed ranker (model scores + sound proof-demotion, re-ranked per failure).
    1-seed dev (only seed_0 exists).
    """
    from alphatamp.approaches.spectre.dataset_v2 import build_v2_example, collate_v2
    from alphatamp.approaches.spectre.evidence import deployed_rollout_traced

    vocab = Vocab.from_json(VOCAB_PATH)
    # RAW episodes, deliberately not `eda.load_split_episodes`. That helper returns
    # *canonicalized* episodes, and `build_v2_example` canonicalizes again --
    # `canonicalize_episode` is **not idempotent** (a second pass permutes object names
    # differently, e.g. item_10 -> item_2), so the doubly-canonicalized episode carries a
    # different object->tag binding than the singly-canonicalized one training sees.
    # Measured on dd2d_v4: identical scene poses, but per-problem FP differs on 35/100
    # and per-stratum by up to 2-3 FP (s2 26.00 vs 23.92, s3 26.44 vs 29.32).
    # Training loads raw (`SpectreV2Dataset.__getitem__` -> `load_episode`), so raw is
    # what makes evaluation match training.
    test = _RawSplit(SPECTRE_TEST)
    for seed in SEEDS:
        adir = CACHE_DIR / "spectre2_adaptive" / f"seed_{seed}"
        sdir = CACHE_DIR / "spectre2_static" / f"seed_{seed}"
        need_a = force or not _dir_complete(adir)
        need_s = force or not _dir_complete(sdir)
        if not need_a and not need_s:
            print(f"[spectre2 seed {seed}] complete; skipping")
            continue
        ck = V2_CKPT_DIR / f"seed_{seed}" / "best.pt"
        model = _load_v2_model(ck, vocab, device)

        na = ns = 0
        for ep in test.episodes:
            if (
                ep.scene_geometry is None
            ):  # v2 needs geometry (all λ=0.8 test eps have it)
                continue
            pid = int(ep.provenance.problem_id)
            labels = [1 if o.outcome == "success" else 0 for o in ep.outcomes]
            if need_s:
                ex = build_v2_example(ep, vocab, rng=None, evidence=False)
                batch = collate_v2([ex], max_arity=vocab.max_operator_arity).to(device)
                logits, _ = model(batch)
                scores = [float(x) for x in logits[0].detach().cpu().numpy()]
                ns += _write(
                    sdir / f"{pid}.json",
                    {
                        "problem_id": pid,
                        "stratum": stratum_of(pid),
                        "scores": scores,
                        "labels": labels,
                    },
                    force,
                )
            if need_a:
                attempts, trace = deployed_rollout_traced(
                    model,
                    ep,
                    vocab,
                    device,
                    demotion_source="observed",
                )
                na += _write(
                    adir / f"{pid}.json",
                    {
                        "problem_id": pid,
                        "stratum": stratum_of(pid),
                        "fp": float(attempts) - 1.0,
                        "order": trace.order,
                        "step_scores": _round_rows(trace.step_scores),
                        "step_dead": trace.step_dead,
                    },
                    force,
                )
        print(f"[spectre2-static seed {seed}] wrote {ns} -> {sdir}")
        print(f"[spectre2-adaptive seed {seed}] wrote {na} -> {adir}")


def _v3_ckpt(ckpt_subdir: str, seed: int) -> Path:
    """Resolve a v3 arm's checkpoint for ``seed``.

    v3 multi-seed arms write **one top-level directory per seed**
    (``checkpoints_v3_v3final_s3/dd2d_v4/seed_3/best.pt``), which the v1/v2
    ``<dir>/<env>/seed_<n>`` pattern cannot express -- so ``{seed}`` in the sub-dir is
    substituted as well as the path component, exactly as ``spectre_score_v3.py`` does.
    """
    return (
        REPO
        / "data"
        / "spectre"
        / ckpt_subdir.replace("{seed}", str(seed))
        / ENV_VARIANT
        / f"seed_{seed}"
        / "best.pt"
    )


def _warn_if_undertrained(arms: dict[str, str]) -> None:
    """Flag arms whose training log never reached the configured epoch count.

    A killed run leaves a complete-looking ``best.pt`` from whatever epoch it had reached,
    and nothing downstream can tell it from a finished model: it loads, it scores, it
    fills a cache directory. Three ``p8_cov_final`` seeds were killed at epoch 5 of 30 and
    were cited for months as "the clean 3-seed re-run"; the stub scores 26.97 against the
    ~8 the finished config gets, and its s0 is 36.64 where every other arm gets 0.00.

    The log is the only record of how far a run actually got -- the checkpoint carries the
    *configured* epoch count, not the reached one -- so this reads the log, and warns
    rather than raises so a deliberately short run stays cacheable.
    """
    logs = REPO / "data" / "spectre" / "logs"
    for prefix, subdir in arms.items():
        ckpt = _v3_ckpt(subdir, SEEDS[0])
        if not ckpt.is_file():
            continue
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
        total = int(cfg.get("epochs", 0))
        # arm name = the checkpoint dir minus the checkpoints_v3[_norec][_noov]_ prefix
        name = subdir.replace("{seed}", str(SEEDS[0]))
        for junk in (
            "checkpoints_v3_norec_noov_",
            "checkpoints_v3_norec_",
            "checkpoints_v3_noov_",
            "checkpoints_v3_",
        ):
            if name.startswith(junk):
                name = name[len(junk) :]
                break
        log = logs / f"{name}.log"
        if not log.is_file() or not total:
            continue
        reached = [
            int(m) for m in re.findall(r"epoch (\d+)/", log.read_text(errors="ignore"))
        ]
        if reached and max(reached) < total:
            print(
                f"!! {prefix}: {log.name} stops at epoch {max(reached)}/{total} — "
                f"{ckpt.name} is a MID-TRAINING stub, not a finished model"
            )


def _assert_same_selector(arms: dict[str, str]) -> None:
    """Refuse to cache arms whose checkpoints were selected by different instruments.

    G6 ran the selector censored at 30 attempts on a 50-episode val subsample; G6b
    retracted that, because DD2D s2/s3 episodes routinely need 30-40+ attempts, so the
    censored statistic clipped exactly the tail that separates models (``decisions.md``
    2026-07-26). An ablation table mixing the two generations is comparing checkpoints
    chosen by two different instruments, and the difference would be read as the feature
    under test. Cheap to check, and invisible in the directory name -- so check it.
    """
    seen: dict[tuple, list[str]] = {}
    for prefix, subdir in arms.items():
        ckpt = _v3_ckpt(subdir, SEEDS[0])
        if not ckpt.is_file():
            continue
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
        key = (cfg.get("select_budget"), cfg.get("val_episodes"))
        seen.setdefault(key, []).append(prefix)
    if len(seen) > 1:
        detail = "; ".join(
            f"budget={k[0]} val_episodes={k[1]}: {', '.join(v)}"
            for k, v in sorted(seen.items(), key=lambda kv: str(kv[0]))
        )
        raise SystemExit(
            "refusing to cache arms selected by different instruments -- "
            f"{detail}. See decisions.md 2026-07-26 (censored selectors)."
        )


def _is_mid_training(ckpt: Path) -> bool:
    """True if a live training run still owns this checkpoint directory.

    ``train_v3`` writes ``best.pt`` the first time selection improves -- at epoch 1 of 30
    -- so the file existing says nothing about the run being finished. Caching it
    produces a full, complete-looking directory of numbers from a half-trained model,
    and because ``_dir_complete`` then skips it, a later run without ``--force`` leaves
    the bad row in place silently. This is the failure the scorer's mtime warning was
    meant to catch, but a warning in a buffered log is not a guard.

    ``train_v3._claim_out_dir`` writes a ``.owner`` pid marker for exactly this class of
    problem, so read it: a live owner means skip, a stale one means the run died and the
    checkpoint is the last good one. Falls back to an mtime heuristic for ``train_v2``
    checkpoints, which predate the marker.
    """
    marker = ckpt.parent / ".owner"
    if marker.is_file():
        try:
            owner = int(marker.read_text().strip())
            os.kill(owner, 0)  # signal 0 = liveness probe, sends nothing
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            return False  # stale marker -> the run is gone, checkpoint is final
        return True
    return (time.time() - ckpt.stat().st_mtime) < 120


@torch.no_grad()
def cache_spectre3(
    force: bool,
    device: str,
    arms: dict[str, str],
    static_arms: frozenset[str] = frozenset({"spectre3"}),
    suppress_records: bool = False,
    apply_demotion: bool = False,
) -> None:
    """SPECTRE v3 adaptive (deployed rollout) + static (empty-context logits) per seed.

    ``arms`` maps *cache sub-dir prefix* -> *checkpoint sub-dir* (which may contain
    ``{seed}``). The deployed method writes ``spectre3_{static,adaptive}``; every other
    arm is an ablation and writes ``<prefix>_adaptive`` only -- the ablation table is an
    FP table and needs no static row.

    Feature switches are read back off each checkpoint by ``load_v3_checkpoint`` and
    splatted into both the tensorizer and the rollout, so an arm cannot be deployed
    under a different ``overlap_mode``/``coverage_mode`` than it trained under.

    ``suppress_records`` and ``apply_demotion`` are the two *deploy-time* diagnostics, and
    they are deliberately NOT read off the checkpoint: they are properties of how a run is
    scored, not of how it was trained. Both are train/deploy mismatches on purpose. Note
    ``apply_demotion`` defaults to **False**, matching the deployed method since
    2026-07-30: v3 is a purely learned ranker and nothing outside the network touches its
    ordering. An arm cached with ``apply_demotion=True`` differs from its demotion-OFF twin
    only in the ranking offset -- same weights, same seeds, same episodes -- so the pair is
    exactly paired and a *zero* difference means the switch never took effect rather than
    that the offset is worthless.
    """
    from alphatamp.approaches.spectre.dataset_v3 import build_v3_example, collate_v3
    from alphatamp.approaches.spectre.domain import spec_for
    from alphatamp.approaches.spectre.inference_v3 import (
        deployed_rollout_v3_traced,
        load_v3_checkpoint,
    )

    vocab = Vocab.from_json(VOCAB_PATH)
    spec = spec_for(ENV_VARIANT)
    test = _RawSplit(SPECTRE_TEST)  # raw: `build_v3_example` canonicalizes
    episodes = [ep for ep in test.episodes if ep.scene_geometry is not None]

    for prefix, ckpt_subdir in arms.items():
        want_static = prefix in static_arms
        for seed in SEEDS:
            adir = CACHE_DIR / f"{prefix}_adaptive" / f"seed_{seed}"
            sdir = CACHE_DIR / f"{prefix}_static" / f"seed_{seed}"
            need_a = force or not _dir_complete(adir)
            need_s = want_static and (force or not _dir_complete(sdir))
            if not need_a and not need_s:
                print(f"[{prefix} seed {seed}] complete; skipping")
                continue
            ckpt = _v3_ckpt(ckpt_subdir, seed)
            if not ckpt.is_file():
                print(f"[{prefix} seed {seed}] !! missing {ckpt}; skipping")
                continue
            if _is_mid_training(ckpt):
                print(
                    f"[{prefix} seed {seed}] !! SKIPPING — {ckpt} is still owned by a "
                    f"live training run; re-run with --force once it finishes",
                    flush=True,
                )
                continue
            model, deploy = load_v3_checkpoint(ckpt, vocab, device)
            # Warm up so one-time CUDA init/autotune does not land in the first problem's
            # measured inference time (no-op cost on cpu).
            if episodes and device.startswith("cuda"):
                _wex, _wr = build_v3_example(
                    episodes[0],
                    vocab,
                    rng=None,
                    evidence=True,
                    context_f=frozenset(),
                    augment_tags=False,
                    spec=spec,
                    **deploy,
                )
                _wb = collate_v3(
                    [_wex], max_arity=vocab.max_operator_arity, records=[_wr]
                ).to(device)
                for _ in range(3):
                    model(_wb)
                torch.cuda.synchronize()

            na = ns = 0
            for ep in episodes:
                pid = int(ep.provenance.problem_id)
                if need_s:
                    # Time the full static-inference path: tensorize + collate + forward.
                    _t0 = time.perf_counter()
                    ex, recs = build_v3_example(
                        ep,
                        vocab,
                        rng=None,
                        evidence=True,
                        context_f=frozenset(),  # F=∅ is the deployment start
                        augment_tags=False,
                        spec=spec,
                        **deploy,
                    )
                    batch = collate_v3(
                        [ex], max_arity=vocab.max_operator_arity, records=[recs]
                    ).to(device)
                    logits, _ = model(batch)
                    scores = [float(x) for x in logits[0].detach().cpu().numpy()]
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    infer_s = round(time.perf_counter() - _t0, 6)
                    order = _static_order(scores)
                    fp_cap, refine_cap = _fp_and_refine_capped(ep, order, REFINE_CAP_S)
                    ns += _write(
                        sdir / f"{pid}.json",
                        {
                            "problem_id": pid,
                            "stratum": stratum_of(pid),
                            "scores": scores,
                            "labels": [
                                1 if o.outcome == "success" else 0 for o in ep.outcomes
                            ],
                            "refine_s": _refine_seconds(ep, order),
                            "refine_s_capped": refine_cap,
                            "fp_capped": fp_cap,
                            "infer_s": infer_s,
                        },
                        force,
                    )
                if need_a:
                    attempts, trace = deployed_rollout_v3_traced(
                        model,
                        ep,
                        vocab,
                        device,
                        spec=spec,
                        mode="strict",
                        suppress_records=suppress_records,
                        apply_demotion=apply_demotion,
                        **deploy,
                    )
                    # Second, capped rollout: a slow-feasible candidate over the cap is
                    # abandoned into the failure context, so the adaptive order can
                    # diverge from the uncapped one -- it must be re-run, not derived by
                    # capping the uncapped order's stored times.
                    attempts_cap, trace_cap = deployed_rollout_v3_traced(
                        model,
                        ep,
                        vocab,
                        device,
                        spec=spec,
                        mode="strict",
                        suppress_records=suppress_records,
                        apply_demotion=apply_demotion,
                        refine_cap_s=REFINE_CAP_S,
                        **deploy,
                    )
                    na += _write(
                        adir / f"{pid}.json",
                        {
                            "problem_id": pid,
                            "stratum": stratum_of(pid),
                            "fp": float(attempts) - 1.0,
                            "order": trace.order,
                            "step_scores": _round_rows(trace.step_scores),
                            "step_dead": trace.step_dead,
                            "refine_s": _refine_seconds(ep, trace.order),
                            "refine_s_capped": trace_cap.refine_capped_seconds,
                            "fp_capped": float(attempts_cap) - 1.0,
                            "order_capped": trace_cap.order,
                            "infer_s": round(trace.infer_seconds, 6),
                        },
                        force,
                    )
            if want_static:
                print(f"[{prefix}-static seed {seed}] wrote {ns} -> {sdir}")
            print(f"[{prefix}-adaptive seed {seed}] wrote {na} -> {adir}")


@torch.no_grad()
def cache_lenctx(force: bool, device: str, repeats: int = 3) -> None:
    """T1 length-only-context intervention: adaptive rollout with identity-

    scrambled (same-length, random-id) failure contexts. Per seed, run the intervention
    ``repeats`` times with distinct surrogate RNGs and cache the per-problem FP averaged
    over repeats (damps Monte-Carlo noise). Layout mirrors ``spectre_adaptive``:
    ``spectre_lenctx/seed_<s>/<pid>.json {problem_id, stratum, fp, order}`` (``order``
    from the first repeat). If this matches ``spectre_adaptive`` FP, Ψ ignores failed-
    skeleton identity (H2).
    """
    vocab = Vocab.from_json(VOCAB_PATH)
    test = _RawSplit(SPECTRE_TEST)  # raw, for the reason in `_RawSplit`
    for seed in SEEDS:
        out = CACHE_DIR / "spectre_lenctx" / f"seed_{seed}"
        if _dir_complete(out) and not force:
            print(f"[lenctx seed {seed}] complete; skipping")
            continue
        ck = CKPT_DIR / f"seed_{seed}" / "best.pt"
        model = load_checkpoint(ck, vocab, device=device)
        prior = load_prior_for_checkpoint(ck)

        fp_sum: dict[int, float] = {}
        order0: dict[int, list[int]] = {}
        for rep in range(repeats):
            res, traces = eda.spectre_evaluate_length_only_context(
                test,
                model,
                vocab,
                attempt_budget=200,
                prior=prior,
                device=device,
                seed=seed * 100 + rep,
                scramble=True,
            )
            for pid, att, trace in zip(res.problem_ids, res.attempts, traces):
                pid = int(pid)
                fp_sum[pid] = fp_sum.get(pid, 0.0) + (float(att) - 1.0)
                if rep == 0:
                    order0[pid] = [int(cs.idx) for cs in trace]
        n = 0
        for pid, total in fp_sum.items():
            n += _write(
                out / f"{pid}.json",
                {
                    "problem_id": pid,
                    "stratum": stratum_of(pid),
                    "fp": total / repeats,
                    "order": order0[pid],
                },
                force,
            )
        print(f"[lenctx seed {seed}] wrote {n} (mean of {repeats} draws) -> {out}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["astar", "piginet", "spectre", "spectre2"],
        choices=["astar", "piginet", "spectre", "spectre2", "spectre3", "lenctx"],
    )
    parser.add_argument(
        "--v3-arm",
        action="append",
        default=[],
        help='"cache_subdir_prefix:ckpt_subdir" to cache one v3 arm instead of the '
        "default registry; the checkpoint sub-dir may contain {seed}. Repeatable.",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="with --methods spectre3, cache only the deployed arm (skip the ablation "
        "arms and the suppress-records diagnostic)",
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--env-variant",
        default=DEFAULT_ENV_VARIANT,
        # The union of the per-variant maps, not `_V2_CKPT_SUBDIR` alone. That was the
        # implicit definition until 2026-08-01 and it means "collections with a SPECTRE
        # v2.2 checkpoint" -- so StickButton2D, where v2.2 was deliberately never trained,
        # was rejected at the CLI even though it has PIGINet and v3 rows. A variant is
        # runnable if *any* method map knows it; a method it lacks fails on its own.
        choices=sorted(set(_V2_CKPT_SUBDIR) | set(_PIGINET_PATHS)),
        help="Which collection to score (repoints test split, vocab, checkpoints, "
        "PIGINet artifacts, and the cache dir).",
    )
    parser.add_argument(
        "--lenctx-repeats",
        type=int,
        default=3,
        help="Surrogate draws per seed for the T1 length-only-context intervention",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Training seeds to cache (default: 0). Each becomes a seed_<n> sub-dir; "
            "the reader averages over whatever is present, and --force is not needed "
            "to add a seed because complete dirs are skipped individually."
        ),
    )
    args = parser.parse_args()

    if args.seeds:
        global SEEDS  # noqa: PLW0603 - module-level config, mirrored by _configure_paths
        SEEDS = list(dict.fromkeys(args.seeds))

    _configure_paths(args.env_variant)
    print(f"env_variant={ENV_VARIANT}  test={SPECTRE_TEST}  N_PROBLEMS={N_PROBLEMS}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if "astar" in args.methods:
        cache_astar(args.force)
    if "piginet" in args.methods:
        cache_piginet(args.force, args.device)
    if "spectre" in args.methods:
        cache_spectre(args.force, args.device)
    if "spectre2" in args.methods:
        cache_spectre2(args.force, args.device)
    if "spectre3" in args.methods:
        if args.v3_arm:
            arms = dict(a.split(":", 1) for a in args.v3_arm)
            suppress: dict[str, str] = {}
            nodemo: dict[str, str] = {}
        elif args.no_ablations:
            arms = {"spectre3": _v3_arm_dir("spectre3", ENV_VARIANT)}
            suppress, nodemo = {}, {}
        else:
            # Through `_v3_arm_dir`, so a collection whose arms live under different run
            # names gets them. Reading `_V3_ARMS` raw here was the bug that made
            # StickButton2D's six trained arms invisible to the cache.
            arms = {a: _v3_arm_dir(a, ENV_VARIANT) for a in _V3_ARMS}
            suppress = dict(_V3_SUPPRESS_ARMS)
            nodemo = dict(_V3_DEMOTION_ARMS)
            if ENV_VARIANT.startswith("stickbutton2d"):
                # Proof-tier demotion was cut from the method, and StickButton2D resolves
                # to EMPTY_SPEC, so `licenses_demotion` is always False -- a demotion arm
                # would be bit-identical to its base. Skipped as vacuous, not overlooked.
                # `suppress` needs a `v3final`-named checkpoint this collection never
                # trained, so it goes too.
                suppress, nodemo = {}, {}
        _assert_same_selector({**arms, **suppress, **nodemo})
        _warn_if_undertrained({**arms, **suppress, **nodemo})
        cache_spectre3(args.force, args.device, arms)
        if suppress:
            cache_spectre3(
                args.force,
                args.device,
                suppress,
                static_arms=frozenset(),
                suppress_records=True,
            )
        if nodemo:
            # Separate call, exactly as `suppress` is: the diagnostic is a property of the
            # scoring run, so it cannot ride in the arms dict without changing what every
            # other arm means. Demotion is OFF everywhere else, so THIS is the arm that
            # turns it on.
            cache_spectre3(
                args.force,
                args.device,
                nodemo,
                static_arms=frozenset(),
                apply_demotion=True,
            )
    if "lenctx" in args.methods:
        cache_lenctx(args.force, args.device, repeats=args.lenctx_repeats)

    # Per-stratum abstract-plan-generation time (shared across pool-ranking methods). Preserve
    # a prior good measurement if this run measured nothing (e.g. a non-DD2D or lenctx-only run).
    meta_path = CACHE_DIR / "meta.json"
    plan_gen_s = _measure_plan_gen()
    if not plan_gen_s and meta_path.exists():
        try:
            plan_gen_s = json.loads(meta_path.read_text()).get("plan_gen_s", {})
        except Exception:  # pragma: no cover - corrupt/legacy meta
            plan_gen_s = {}
    # The per-candidate cap only reorders the pool; a problem is lost only if *every*
    # feasible candidate exceeds the cap. Log that count so a future collection where it is
    # non-zero (a domain with slow-feasible plans) is caught rather than silently censored.
    at_risk = _feasibility_at_risk(REFINE_CAP_S)
    if at_risk is not None:
        tag = "OK" if at_risk == 0 else "!! WARN"
        print(
            f"[cap] refine_cap_s={REFINE_CAP_S}s: {at_risk} problem(s) with all feasible "
            f"candidates > cap (would be censored) [{tag}]"
        )
    meta_path.write_text(
        json.dumps(
            {
                "env_variant": ENV_VARIANT,
                "methods": args.methods,
                "seeds": SEEDS,
                "n_problems": N_PROBLEMS,
                "device": args.device,
                "plan_gen_s": plan_gen_s,
                "refine_cap_s": REFINE_CAP_S,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"cache ready at {CACHE_DIR}")


if __name__ == "__main__":
    main()
