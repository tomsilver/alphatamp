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
    spectre3_static/seed_<s>/<pid>.json  {problem_id, stratum, scores, labels}
    spectre3_adaptive/seed_<s>/<pid>.json{problem_id, stratum, fp, order,
                                          step_scores, step_dead}
        (SPECTRE deployed ranker: observed coverage/waste + record tokens + the record
        state delta. ``order`` = the realized attempt sequence of pool indices, until
        first success; consumed by the notebook's realized-order + length-ladder views.
        ``step_scores[t]`` = the raw (K,) logits the step-t pick was made from, before the
        tried-mask. **No proof-demotion** -- cut from the method 2026-07-30, so nothing
        outside the network touches the ordering; ``step_dead`` is retained as an
        always-empty list so the cache schema is unchanged.)
    abl_<arm>_adaptive/seed_<s>/<pid>.json
        (one dir per ablation arm -- see ``_V3_ARMS``; adaptive shape only)
    lazy_adaptive/seed_<s>/<pid>.json     {problem_id, stratum, fp, order, ...}
        (LAZY policy-guided adaptive re-ranker; see ``cache_lazy``)

Usage::

    python experiments/spectre/precompute_dd2d_cache.py            # default methods
    python experiments/spectre/precompute_dd2d_cache.py --force
    python experiments/spectre/precompute_dd2d_cache.py --methods piginet spectre3
    python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v3 --force
    # SPECTRE + its ablation arms (dd2d_v4 is the only collection with checkpoints)
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
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]
DD2D_OUT = REPO / "data" / "dd2d" / "out_dd2d"
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
        "ckpt": DD2D_OUT / "piginet_bce" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v2",
        "cache": DD2D_OUT / "clip_cache_v2",
    },
    "dd2d_v3": {
        "ckpt": DD2D_OUT / "piginet_bce_v3" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v3",  # the repo-root re-collection
        "cache": DD2D_OUT / "clip_cache_v3",
    },
    # dd2d_v4 is the first collection where PIGINet has a real seed axis: `train.py`
    # gained `--seed` on 2026-07-28, so `{seed}` appears in the checkpoint path and the
    # cache is written per seed. Earlier variants have one deterministic run each and
    # keep their flat, seedless cache layout -- the reader detects which it is looking at.
    "dd2d_v4": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4",
        "cache": DD2D_OUT / "clip_cache_v4",
    },
    # DD2D shape-only generalization set (2026-08-04): scored train-old / test-new via
    # `--test-variant`, so PIGINet's checkpoint comes from the dd2d_v4 (train) entry above
    # while `data`/`cache` point at the gen collection's native JSON + a fresh CLIP cache
    # (auto-built by `precompute_clip_cache`). `ckpt` here is the same v4 head, so even a
    # standalone `--env-variant dd2d_v4gen_shapeonly` PIGINet run stays train-old.
    "dd2d_v4gen_shapeonly": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4gen_shapeonly",
        "cache": DD2D_OUT / "clip_cache_v4gen_shapeonly",
    },
    # Shape-SIZE sweep (2026-08-06): the physically-shrunk tee/cross collection (x0.7
    # linear). A real, PIGINet-able variant -- the collector wrote native JSON + crops --
    # scored train-old / test-new like the shape-only set (same v4 head).
    "dd2d_v4gen_shapeonly_sz07": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4gen_shapeonly_sz07",
        "cache": DD2D_OUT / "clip_cache_v4gen_shapeonly_sz07",
    },
    # Inference-time geometry interventions (2026-08-06): the shape-only episodes with
    # tee/cross area (hullarea) or boundary (hullshape) rewritten to their convex hull, to
    # probe SPECTRE's geometry representation. SPECTRE + astar only -- these are derived
    # from EpisodeRecord pickles, not a fresh collection, so there are no intervention-
    # specific PIGINet crops; run with `--methods spectre3 astar`. `data`/`cache` reuse the
    # shape-only crops only so an accidental PIGINet run does not crash (it would be the
    # unmodified image, hence meaningless -- do not run `--methods piginet` here).
    "dd2d_v4gen_shapeonly_hullarea": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4gen_shapeonly",
        "cache": DD2D_OUT / "clip_cache_v4gen_shapeonly",
    },
    "dd2d_v4gen_shapeonly_hullshape": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4gen_shapeonly",
        "cache": DD2D_OUT / "clip_cache_v4gen_shapeonly",
    },
    # scale07: the paired input-only x0.7 shrink (same problems + labels as the shape-only
    # set; only tee/cross boundary+area shrunk in the model input). SPECTRE + astar only,
    # like the other interventions.
    "dd2d_v4gen_shapeonly_scale07": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4gen_shapeonly",
        "cache": DD2D_OUT / "clip_cache_v4gen_shapeonly",
    },
    # fresh un-shrunk control (band 7): bounds collection variance for the sz07 shrink.
    # A real collection with native crops, so PIGINet-able if wanted; scored SPECTRE+astar.
    "dd2d_v4gen_shapeonly_fresh": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4gen_shapeonly_fresh",
        "cache": DD2D_OUT / "clip_cache_v4gen_shapeonly_fresh",
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
    # ------------------------------------------------------------------------------ #
    # Held-out-STRATUM generalization (2026-08-09). The learned methods are trained on
    # s0-s2 (DD2D) / b1-b3 (SB2D) and evaluated on the held-out stratum s3 / b5, via the
    # `--train-strata 0 1 2` filter -- NOT a re-collection. Each variant's raw dir is a
    # symlink to its backing collection (so `test/episodes` + `images` resolve unchanged);
    # only the trained checkpoint differs. astar + VLMPlan are training-free and reused.
    #
    # DD2D single cache: the holdout PIGINet head; data + CLIP cache reuse dd2d_v4's
    # (same test images -- CLIP is checkpoint-independent).
    "dd2d_v4_holdout_s3": {
        "ckpt": DD2D_OUT / "piginet_bce_v4_holdout_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "dd2d" / "raw_v4",
        "cache": DD2D_OUT / "clip_cache_v4",
    },
    # SB2D SPECTRE-only cache (instrumented v1 refiner). No PIGINet is run here -- SPECTRE
    # is image-free -- but the variant must be a known `--env-variant`, so it mirrors
    # `stickbutton2d_v1`; the `ckpt` below is inert (never loaded without `--methods
    # piginet`).
    "stickbutton2d_v1_holdout_b5": {
        "ckpt": DERIVED_ROOT / "stickbutton2d_v1" / "piginet_bce_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "spectre",
        "cache": DERIVED_ROOT / "stickbutton2d_v1" / "clip_cache",
        "domain": "stickbutton2d",
    },
    # SB2D primary cache (kinder crops): the holdout PIGINet head trained on kinder crops;
    # data + CLIP cache reuse the deployed kinder variant's. `make_sb2d_domain` reads the
    # kinder PNGs for this variant (registered in `sb2d_adapter._SB2D_CROP_SOURCE`).
    "stickbutton2d_v1_kinder_holdout_b5": {
        "ckpt": DERIVED_ROOT
        / "stickbutton2d_v1_kinder"
        / "piginet_bce_holdout_s{seed}"
        / "ckpt.pt",
        "data": REPO / "data" / "spectre",
        "cache": DERIVED_ROOT / "stickbutton2d_v1_kinder" / "clip_cache",
        "domain": "stickbutton2d",
    },
    # ------------------------------------------------------------------------------ #
    # b5-correct-size collection (2026-08-09). `stickbutton2d_v2` reuses v1's b1/b2/b3
    # (and val/test) and collects b5 TRAIN to the full 100, so the held-out-b5 contrast
    # is a proper ~25% perturbation instead of v1's near-null 6% (17 episodes). v1 stays
    # frozen. v2 is the SPECTRE (image-free) side; `stickbutton2d_v2_kinder` carries the
    # full-strata PIGINet head on the re-rendered kinder crops. Only the FULL model is new
    # -- the subset (b1/b2/b3) is identical to v1 and reuses the held-out checkpoints.
    "stickbutton2d_v2": {
        "ckpt": DERIVED_ROOT / "stickbutton2d_v1" / "piginet_bce_s{seed}" / "ckpt.pt",
        "data": REPO / "data" / "spectre",
        "cache": DERIVED_ROOT / "stickbutton2d_v1" / "clip_cache",
        "domain": "stickbutton2d",
    },
    "stickbutton2d_v2_kinder": {
        "ckpt": DERIVED_ROOT
        / "stickbutton2d_v2_kinder"
        / "piginet_bce_s{seed}"
        / "ckpt.pt",
        "data": REPO / "data" / "spectre",
        "cache": DERIVED_ROOT / "stickbutton2d_v2_kinder" / "clip_cache",
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

# Env-variant-dependent path globals. The cache functions read these as module globals
# at call time, so ``main`` rebinds them via ``_configure_paths`` from ``--env-variant``
# before dispatching. The literal defaults below are the historical ``dd2d_v2`` values
# (kept byte-identical so importing the module is unchanged); ``_configure_paths``
# overrides them and, unlike these, derives ``N_PROBLEMS`` from the real test split.
ENV_VARIANT = DEFAULT_ENV_VARIANT
# The collection whose vocab + checkpoints are scored. Equals ENV_VARIANT except for a
# train-old / test-new run (`--test-variant`), where ENV_VARIANT is the TEST/episode
# collection and CKPT_VARIANT is the TRAIN one.
CKPT_VARIANT = DEFAULT_ENV_VARIANT
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


def _configure_paths(env_variant: str, ckpt_variant: str | None = None) -> None:
    """(Re)bind every env-variant-dependent module global.

    ``env_variant`` is the TEST/episode collection: the raw test split scored, PIGINet's
    data + CLIP cache, the output ``compare_cache``, ``N_PROBLEMS`` and the refine cap.
    ``ckpt_variant`` (default ``env_variant``) is the TRAIN collection whose vocab and
    checkpoints are loaded. They differ only for a train-old / test-new generalization
    run (``--test-variant``), splitting exactly the way ``spectre_score.py`` does: model
    + vocab from ``--env-variant``, episodes from ``--test-variant``.
    """
    global ENV_VARIANT, CKPT_VARIANT, SPECTRE_TEST, VOCAB_PATH, CKPT_DIR, V2_CKPT_DIR
    global PIGINET_CKPT, PIGINET_DATA, PIGINET_CACHE, PIGINET_DOMAIN
    global CACHE_DIR, N_PROBLEMS, REFINE_CAP_S
    ckpt_variant = ckpt_variant or env_variant
    known = set(_V2_CKPT_SUBDIR) | set(_PIGINET_PATHS)
    for _v, _flag in ((env_variant, "--test-variant"), (ckpt_variant, "--env-variant")):
        if _v not in known:
            raise SystemExit(
                f"unknown {_flag} {_v!r}; known: {sorted(known)} "
                "(add a _V2_CKPT_SUBDIR or _PIGINET_PATHS entry to onboard a collection)"
            )
    ENV_VARIANT = env_variant
    CKPT_VARIANT = ckpt_variant
    # --- from the TEST/episode variant: what is scored and where the cache is written ---
    SPECTRE_TEST = REPO / "data" / "spectre" / "raw" / env_variant / "test"
    CACHE_DIR = REPO / "data" / "spectre" / "derived" / env_variant / "compare_cache"
    N_PROBLEMS = _count_test_problems(SPECTRE_TEST)
    REFINE_CAP_S = _REFINE_CAP_S.get(env_variant, _DEFAULT_REFINE_CAP_S)
    # --- from the TRAIN/checkpoint variant: the vocab and checkpoints being scored ---
    VOCAB_PATH = (
        REPO / "data" / "spectre" / "derived" / ckpt_variant / "train_vocab.json"
    )
    CKPT_DIR = REPO / "data" / "spectre" / "checkpoints" / ckpt_variant
    # A collection with no SPECTRE v2.2 checkpoint (StickButton2D: v2 was scoped out) gets
    # `None` rather than a KeyError, so `--methods spectre2` fails on that method alone
    # instead of the whole driver refusing to start.
    _v2_sub = _V2_CKPT_SUBDIR.get(ckpt_variant)
    V2_CKPT_DIR = (
        None if _v2_sub is None else REPO / "data" / "spectre" / _v2_sub / ckpt_variant
    )
    # PIGINet is optional per variant: it trains on the *native* DD2D JSON with its own
    # CLIP cache, so onboarding a collection for the SPECTRE methods does not
    # automatically give it a PIGINet row. Missing paths become None and only fail if
    # `--methods piginet` actually asks for it. The checkpoint comes from the TRAIN
    # variant (train-old); the data + cache from the TEST variant (test-new).
    _ckpt_pig = _PIGINET_PATHS.get(ckpt_variant, {})
    _test_pig = _PIGINET_PATHS.get(env_variant, {})
    PIGINET_CKPT = _ckpt_pig.get("ckpt")
    PIGINET_DATA = _test_pig.get("data")
    PIGINET_CACHE = _test_pig.get("cache")
    PIGINET_DOMAIN = _test_pig.get("domain")


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


# Deployed per-candidate refinement-abandonment cap (seconds), **per env-variant**. A
# skeleton is refined for at most this long before the deployment moves on; a feasible
# candidate over the cap is abandoned (its outcome no longer counts as a stopping success).
# The §2b wall-clock table reports every pool-ranking method under it; the uncapped fields
# stay for the FP headline. See decisions/07 2026-08-02 (DD2D) / 2026-08-03 (SB2D).
#
# The right value depends on where a domain's feasible refines sit relative to its budget:
#   - dd2d_v4: 2.0s = ~4.5x the feasible p95 (0.44s) -- the cap sits *above the whole
#     feasible distribution*, so no feasible is ever cut and only the 20s-budget dead-ends
#     are; `_feasibility_at_risk(2.0) == 0`.
#   - stickbutton2d_v1{,_kinder}: 10.0s. SB2D feasible refines are seconds (p95 10.6s), too
#     slow for a DD2D-style cap-above-the-distribution to fit under the 20s budget. 10.0s
#     instead clears the worst *per-problem fastest-feasible* (max 8.84s) with margin --
#     `_feasibility_at_risk(10.0) == 0`, no problem censored -- while still cutting the many
#     budget-exhausting failures (33% of all per-candidate refines exceed it). The two SB2D
#     variants MUST share one value: the kinder §2b grafts SPECTRE timing from the v1 cache,
#     so their capped fields have to be computed under the same cap.
_DEFAULT_REFINE_CAP_S = 2.0
_REFINE_CAP_S: dict[str, float] = {
    "dd2d_v4": 2.0,
    "stickbutton2d_v1": 10.0,
    "stickbutton2d_v1_kinder": 10.0,
    # Held-out-stratum variants share their backing collection's cap so §2b is comparable.
    "dd2d_v4_holdout_s3": 2.0,
    "stickbutton2d_v1_holdout_b5": 10.0,
    "stickbutton2d_v1_kinder_holdout_b5": 10.0,
    # b5-correct-size collection (SB2D matched full control).
    "stickbutton2d_v2": 10.0,
    "stickbutton2d_v2_kinder": 10.0,
}
# Rebound per-variant by `_configure_paths`; the module default keeps the historical dd2d
# behaviour for any variant not listed above.
REFINE_CAP_S = _DEFAULT_REFINE_CAP_S


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


def _measure_plan_gen_sb2d(per_stratum: int) -> dict[str, float]:
    """StickButton2D analog of :func:`_measure_plan_gen`'s DD2D body.

    Per stratum, rebuild the kinder env for that button count and time the acyclic pool
    draw (``collect.time_pool_generation``) on a few test problems — the same generator
    the collection used. The config mirrors ``sb2d_collect._config`` (``K_max=200``, 60s
    abstract-plan timeout, default ``closed_form`` ``plan_generator`` — which routes
    ``stickbutton2d`` to the geometry-aware ``AcyclicPlanGenerator``), so the timed pool
    is the collected pool. A regenerated proxy (a representative per-stratum constant),
    like DD2D's. Runs for both ``stickbutton2d_v1`` and ``…_kinder`` — same underlying
    kinder env and pid encoding.
    """
    try:
        from collections import defaultdict

        from alphatamp.approaches.spectre.collect import time_pool_generation
        from alphatamp.approaches.spectre.config import CollectionConfig
        from alphatamp.approaches.spectre.envs.stickbutton2d import strata
    except Exception as e:  # pragma: no cover - env not importable
        print(f"[plan_gen] sb2d unavailable, skipping: {type(e).__name__}: {e}")
        return {}

    groups: dict[int, list[int]] = defaultdict(list)
    for ep in eda.load_split_episodes(SPECTRE_TEST).episodes:
        groups[stratum_of(int(ep.provenance.problem_id))].append(
            int(ep.provenance.problem_id)
        )
    out: dict[str, float] = {}
    for s in sorted(groups):
        num_buttons = strata.BUTTON_COUNTS[s]
        cfg = CollectionConfig(
            env_id=strata.env_id(num_buttons),
            env_variant=ENV_VARIANT,
            model_name="stickbutton2d",
            model_kwargs={"num_buttons": num_buttons},
            split="test",
            num_problems=1,
            problem_seed_start=0,
            problem_seed_end=1,
            K_max=200,
            abstract_plan_timeout_s=60.0,
            refinement_timeout_s=20.0,
            num_sampling_attempts_per_step=5,
            max_trajectory_steps=200,
        )
        times: list[float] = []
        for pid in sorted(groups[s])[:per_stratum]:
            try:
                times.append(time_pool_generation(cfg, pid))
            except Exception as e:  # keep going; a per-problem failure is not fatal
                print(f"[plan_gen] s{s} pid{pid} skipped: {type(e).__name__}: {e}")
        if times:
            out[str(s)] = round(sum(times) / len(times), 6)
            print(f"[plan_gen] s{s}: {out[str(s)]:.4f}s (n={len(times)})", flush=True)
    return out


def _measure_plan_gen(per_stratum: int = 3) -> dict[str, float]:
    """Per-stratum abstract-plan-generation time (s), shared by all pool-ranking
    methods.

    Not stored at collection, so measured here: regenerate a few problems per stratum
    from the stored ``gen_params`` + seed and time the astar top-k pool enumeration
    (``make_dd2d_planner(prefer='pyperplan', search='astar', heuristic='dist').plan``) —
    the step that produces the ranked candidate pool the models score. A regenerated
    proxy (DD2D's generator is PYTHONHASHSEED-dependent), used as a representative per-
    stratum constant. StickButton2D dispatches to :func:`_measure_plan_gen_sb2d`; any
    other variant returns ``{}`` (and a 0 fallback in the notebook), as on failure.
    """
    if ENV_VARIANT.startswith("stickbutton2d"):
        return _measure_plan_gen_sb2d(per_stratum)
    if not ENV_VARIANT.startswith("dd2d"):
        return {}
    try:
        from collections import defaultdict

        from alphatamp.approaches.spectre.envs.dd2d.drawer.planning import (
            make_dd2d_planner,
        )
        from alphatamp.approaches.spectre.envs.dd2d.drawer.problem import (
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
        from alphatamp.approaches.spectre.baselines.piginet.eval import score_split

        print(f"[{tag}] running fresh inference on test split ...")
        domain = None
        if PIGINET_DOMAIN == "stickbutton2d":
            from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import (
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


class _RawSplit:
    """``eda.LoadedSplit``-shaped view over **un-canonicalized** episodes.

    ``eda.load_split_episodes`` canonicalizes on load, which is right for the EDA
    baselines (they key on canonical skeletons) but wrong for anything that then calls a
    tensorizer, because ``build_example`` canonicalizes again and
    ``canonicalize_episode`` is not idempotent. Double canonicalization silently changes
    the object->tag binding relative to training, which loads raw.

    **Every model cache function must load through this.** Double-canonicalization is
    why the dd2d_v3 v2 number (13.68) was retracted (``decisions.md`` 2026-07-26).
    Training loads raw, so raw is what makes evaluation match training.

    Exposes ``.episodes`` -- the only attribute the cache functions read.
    """

    def __init__(self, split_dir: Path) -> None:
        from alphatamp.approaches.spectre.io import list_episodes, load_episode

        self.episodes = [load_episode(p) for p in list_episodes(split_dir)]


def _v3_ckpt(ckpt_subdir: str, seed: int) -> Path:
    """Resolve a v3 arm's checkpoint for ``seed``.

    v3 multi-seed arms write **one top-level directory per seed**
    (``checkpoints_v3_v3final_s3/dd2d_v4/seed_3/best.pt``), which the v1/v2
    ``<dir>/<env>/seed_<n>`` pattern cannot express -- so ``{seed}`` in the sub-dir is
    substituted as well as the path component, exactly as ``spectre_score.py`` does.
    The env component is ``CKPT_VARIANT`` (the TRAIN collection), so a train-old /
    test-new run loads the dd2d_v4 checkpoint while scoring the gen episodes.
    """
    return (
        REPO
        / "data"
        / "spectre"
        / ckpt_subdir.replace("{seed}", str(seed))
        / CKPT_VARIANT
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
) -> None:
    """SPECTRE adaptive (deployed rollout) + static (empty-context logits) per seed.

    ``arms`` maps *cache sub-dir prefix* -> *checkpoint sub-dir* (which may contain
    ``{seed}``). The deployed method writes ``spectre3_{static,adaptive}``; every other
    arm is an ablation and writes ``<prefix>_adaptive`` only -- the ablation table is an
    FP table and needs no static row.

    Feature switches are read back off each checkpoint by ``load_checkpoint`` and
    splatted into both the tensorizer and the rollout, so an arm cannot be deployed
    under a different ``overlap_mode``/``coverage_mode`` than it trained under.

    ``suppress_records`` is a *deploy-time* diagnostic, deliberately NOT read off the
    checkpoint: it is a property of how a run is scored, not of how it was trained -- a
    train/deploy mismatch on purpose. The deployed method is a purely learned ranker
    (proof-tier demotion was cut on 2026-07-30), so nothing outside the network reorders
    the pool.
    """
    from alphatamp.approaches.spectre.dataset import build_example, collate
    from alphatamp.approaches.spectre.domain import spec_for
    from alphatamp.approaches.spectre.inference import (
        deployed_rollout_traced,
        load_checkpoint,
    )

    vocab = Vocab.from_json(VOCAB_PATH)
    # The domain contract is a property of the trained model, so it tracks the checkpoint
    # (train) variant -- CKPT_VARIANT -- not the scored episodes. Identical for a dd2d ->
    # dd2d-gen run (both resolve to `_DD2D`), but correct if they ever diverge.
    spec = spec_for(CKPT_VARIANT)
    test = _RawSplit(SPECTRE_TEST)  # raw: `build_example` canonicalizes
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
            model, deploy = load_checkpoint(ckpt, vocab, device)
            # Warm up so one-time CUDA init/autotune does not land in the first problem's
            # measured inference time (no-op cost on cpu).
            if episodes and device.startswith("cuda"):
                _wex, _wr = build_example(
                    episodes[0],
                    vocab,
                    rng=None,
                    evidence=True,
                    context_f=frozenset(),
                    augment_tags=False,
                    spec=spec,
                    **deploy,
                )
                _wb = collate(
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
                    ex, recs = build_example(
                        ep,
                        vocab,
                        rng=None,
                        evidence=True,
                        context_f=frozenset(),  # F=∅ is the deployment start
                        augment_tags=False,
                        spec=spec,
                        **deploy,
                    )
                    batch = collate(
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
                    attempts, trace = deployed_rollout_traced(
                        model,
                        ep,
                        vocab,
                        device,
                        spec=spec,
                        suppress_records=suppress_records,
                        **deploy,
                    )
                    # Second, capped rollout: a slow-feasible candidate over the cap is
                    # abandoned into the failure context, so the adaptive order can
                    # diverge from the uncapped one -- it must be re-run, not derived by
                    # capping the uncapped order's stored times.
                    attempts_cap, trace_cap = deployed_rollout_traced(
                        model,
                        ep,
                        vocab,
                        device,
                        spec=spec,
                        suppress_records=suppress_records,
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
def cache_lazy(force: bool, device: str) -> None:
    """LAZY (policy-guided lazy search, Khodeir et al): adaptive pool re-ranker.

    Per seed, load ``checkpoints/<CKPT_VARIANT>/lazy_s<seed>/ckpt.pt`` (the GAT policy +
    the fitted feasibility prior ϕ), then per test problem run the online π̄=π·ϕ/Σ rollout
    (uncapped + capped) and write the adaptive record shape. The policy π is evaluated
    once per problem (one batched GAT forward over the prefix tree); the capped rollout is
    re-run rather than derived, exactly as ``cache_spectre3``.
    """
    # Local imports: the vendored torch_geometric GAT stack.
    from alphatamp.approaches.spectre.baselines.lazy.dataset import load_structs
    from alphatamp.approaches.spectre.baselines.lazy.domain import make_lazy_domain
    from alphatamp.approaches.spectre.baselines.lazy.eval import (
        load_lazy_checkpoint,
        rollout_episode,
    )
    from alphatamp.approaches.spectre.baselines.lazy.graph import build_feature_spec

    vocab = Vocab.from_json(VOCAB_PATH)
    spec = build_feature_spec(vocab)
    # Scales are family-based (dd2d cm vs sb2d config), so the scored (ENV) variant is fine;
    # the vocab/model come from CKPT_VARIANT above (train-old / test-new safe).
    domain = make_lazy_domain(ENV_VARIANT)
    structs = load_structs(
        SPECTRE_TEST, vocab, spec, domain.frame_extent, domain.shape_max
    )

    for seed in SEEDS:
        out = CACHE_DIR / "lazy_adaptive" / f"seed_{seed}"
        if _dir_complete(out) and not force:
            print(f"[lazy seed {seed}] complete; skipping")
            continue
        ckpt = (
            REPO
            / "data"
            / "spectre"
            / "checkpoints"
            / CKPT_VARIANT
            / f"lazy_s{seed}"
            / "ckpt.pt"
        )
        if not ckpt.is_file():
            print(f"[lazy seed {seed}] !! missing {ckpt}; skipping", flush=True)
            continue
        model, phi = load_lazy_checkpoint(ckpt, device)
        # Warm up CUDA so init does not land in the first problem's measured infer time.
        if structs and device.startswith("cuda"):
            for _ in range(3):
                rollout_episode(model, structs[0], vocab, spec, phi, device)
            torch.cuda.synchronize()
        n = 0
        for st in structs:
            ep = st.episode
            pid = int(ep.provenance.problem_id)
            _t0 = time.perf_counter()
            r = rollout_episode(model, st, vocab, spec, phi, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            infer_s = round(time.perf_counter() - _t0, 6)
            r_cap = rollout_episode(
                model, st, vocab, spec, phi, device, cap=REFINE_CAP_S
            )
            n += _write(
                out / f"{pid}.json",
                {
                    "problem_id": pid,
                    "stratum": stratum_of(pid),
                    "fp": float(r.attempts) - 1.0,
                    "order": r.order,
                    "refine_s": r.refine_s,
                    "refine_s_capped": r_cap.refine_s,
                    "fp_capped": float(r_cap.attempts) - 1.0,
                    "order_capped": r_cap.order,
                    "infer_s": infer_s,
                },
                force,
            )
        print(f"[lazy seed {seed}] wrote {n} -> {out}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["astar", "piginet", "spectre3"],
        choices=[
            "astar",
            "piginet",
            "spectre3",
            "lazy",
        ],
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
        help="Which collection to score: repoints vocab, checkpoints and PIGINet's "
        "checkpoint; also the test split, cache dir and PIGINet data UNLESS "
        "--test-variant overrides them. With --test-variant this is the TRAIN variant.",
    )
    parser.add_argument(
        "--test-variant",
        default=None,
        choices=sorted(set(_V2_CKPT_SUBDIR) | set(_PIGINET_PATHS)),
        help="Score THIS collection's test episodes while loading the vocab, model config "
        "and checkpoints from --env-variant -- the train-old / test-new generalization "
        "eval (e.g. --env-variant dd2d_v4 --test-variant dd2d_v4gen_shapeonly). The "
        "compare_cache, N_PROBLEMS, refine cap and PIGINet data/CLIP-cache come from here; "
        "the SPECTRE + PIGINet checkpoints stay on --env-variant. Mirrors "
        "spectre_score.py's --test-variant.",
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

    # --env-variant is the checkpoint/train variant; --test-variant (if given) is the
    # scored episode variant. Absent, they coincide (the ordinary same-collection run).
    _episode_variant = args.test_variant or args.env_variant
    _configure_paths(_episode_variant, ckpt_variant=args.env_variant)
    print(
        f"episode_variant={ENV_VARIANT}  ckpt_variant={CKPT_VARIANT}  "
        f"test={SPECTRE_TEST}  N_PROBLEMS={N_PROBLEMS}"
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if "astar" in args.methods:
        cache_astar(args.force)
    if "piginet" in args.methods:
        cache_piginet(args.force, args.device)
    if "spectre3" in args.methods:
        if args.v3_arm:
            arms = dict(a.split(":", 1) for a in args.v3_arm)
            suppress: dict[str, str] = {}
        elif args.no_ablations:
            arms = {"spectre3": _v3_arm_dir("spectre3", CKPT_VARIANT)}
            suppress = {}
        else:
            # Through `_v3_arm_dir`, so a collection whose arms live under different run
            # names gets them. Reading `_V3_ARMS` raw here was the bug that made
            # StickButton2D's six trained arms invisible to the cache. Keyed on
            # CKPT_VARIANT: the arm-dir name is a property of the trained checkpoint.
            arms = {a: _v3_arm_dir(a, CKPT_VARIANT) for a in _V3_ARMS}
            suppress = dict(_V3_SUPPRESS_ARMS)
            if ENV_VARIANT.startswith("stickbutton2d"):
                # `suppress` needs a `v3final`-named checkpoint this collection never
                # trained, so it is skipped here.
                suppress = {}
        _assert_same_selector({**arms, **suppress})
        _warn_if_undertrained({**arms, **suppress})
        cache_spectre3(args.force, args.device, arms)
        if suppress:
            cache_spectre3(
                args.force,
                args.device,
                suppress,
                static_arms=frozenset(),
                suppress_records=True,
            )
    if "lazy" in args.methods:
        cache_lazy(args.force, args.device)

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
                "ckpt_variant": CKPT_VARIANT,
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
