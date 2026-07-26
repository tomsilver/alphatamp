"""Precompute + cache per-problem test scores for the DD2D method comparison.

Runs each method once on the DD2D test split and writes the *raw per-skeleton
scores* per problem so ``experiments/spectre/compare_dd2d_methods.py`` can load
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
    spectre_lenctx/seed_<s>/<pid>.json   {problem_id, stratum, fp, order}
        (T1 length-only-context intervention: adaptive rollout with identity-
        scrambled same-length failure contexts; fp = mean over surrogate draws)

Usage::

    python experiments/spectre/precompute_dd2d_cache.py            # default methods
    python experiments/spectre/precompute_dd2d_cache.py --force
    python experiments/spectre/precompute_dd2d_cache.py --methods piginet spectre2
    python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v3 --force

This is a bridge driver: it imports the vendored ``piginet.eval`` scorer, so it is
excluded from strict mypy/pylint like the marimo notebook it feeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from alphatamp.approaches.spectre import eda
from alphatamp.approaches.spectre.dd2d_compare import stratum_of
from alphatamp.approaches.spectre.inference import (
    init_inference_state,
    load_checkpoint,
    load_prior_for_checkpoint,
)
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]
DD2D = REPO / "src" / "alphatamp" / "approaches" / "spectre" / "envs" / "dd2d"
SEEDS = [0]
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
    global PIGINET_CKPT, PIGINET_DATA, PIGINET_CACHE, CACHE_DIR, N_PROBLEMS
    if env_variant not in _V2_CKPT_SUBDIR:
        raise SystemExit(
            f"unknown --env-variant {env_variant!r}; known: {sorted(_V2_CKPT_SUBDIR)} "
            "(add a _V2_CKPT_SUBDIR entry to onboard a new collection)"
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
    V2_CKPT_DIR = REPO / "data" / "spectre" / _V2_CKPT_SUBDIR[env_variant] / env_variant
    PIGINET_CKPT = piginet.get("ckpt")
    PIGINET_DATA = piginet.get("data")
    PIGINET_CACHE = piginet.get("cache")
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
        n += _write(
            out / f"{pid}.json",
            {
                "problem_id": pid,
                "stratum": stratum_of(pid),
                "scores": scores,
                "labels": labels,
            },
            force,
        )
    print(f"[astar] wrote {n} problems -> {out}")


@torch.no_grad()
def cache_piginet(force: bool, device: str) -> None:
    """PIGINet (BCE-trained, paper baseline): fresh inference; cache logits + labels."""
    if PIGINET_CKPT is None:
        raise SystemExit(
            f"--methods piginet requested but {ENV_VARIANT!r} has no _PIGINET_PATHS "
            "entry. PIGINet trains on the native DD2D JSON with its own CLIP cache, so a "
            "new collection needs it retrained first (envs/dd2d/piginet/train.py), then "
            "an entry added here."
        )
    out = CACHE_DIR / "piginet"
    if _dir_complete(out) and not force:
        print("[piginet] complete; skipping")
        return
    # Local import: vendored piginet stack (pulls in open_clip / CLIP).
    from alphatamp.approaches.spectre.envs.dd2d.piginet.eval import score_split

    print("[piginet] running fresh inference on test split ...")
    rows, _thr, _temp = score_split(
        str(PIGINET_CKPT),
        str(PIGINET_DATA),
        str(PIGINET_CACHE),
        "test",
        device=device,
    )
    by_pid: dict[str, list[tuple[int, int, float]]] = {}
    for pid, _stratum, plan_idx, _length, label, score in rows:
        by_pid.setdefault(pid, []).append((int(plan_idx), int(label), float(score)))
    n = 0
    for pid_str, triples in by_pid.items():
        triples.sort(key=lambda t: t[0])  # order by plan_idx
        pid = int(pid_str.split("_s")[-1])
        n += _write(
            out / f"{pid}.json",
            {
                "problem_id": pid,
                "stratum": stratum_of(pid),
                "scores": [t[2] for t in triples],
                "labels": [t[1] for t in triples],
            },
            force,
        )
    print(f"[piginet] wrote {n} problems -> {out}")


@torch.no_grad()
def cache_spectre(force: bool, device: str) -> None:
    """SPECTRE adaptive (rollout FP) + static (empty-context logits) per seed."""
    vocab = Vocab.from_json(VOCAB_PATH)
    test = eda.load_split_episodes(SPECTRE_TEST)
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
    tensorizer, because ``build_v2_example`` canonicalizes again and
    ``canonicalize_episode`` is not idempotent. Double canonicalization silently changes
    the object->tag binding relative to training, which loads raw.

    Exposes only ``.episodes`` -- the attribute the model cache functions use.
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
    test = eda.load_split_episodes(SPECTRE_TEST)
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
        choices=["astar", "piginet", "spectre", "spectre2", "lenctx"],
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--env-variant",
        default=DEFAULT_ENV_VARIANT,
        choices=sorted(_V2_CKPT_SUBDIR),
        help="Which DD2D collection to score (repoints test split, vocab, checkpoints, "
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
    if "lenctx" in args.methods:
        cache_lenctx(args.force, args.device, repeats=args.lenctx_repeats)

    (CACHE_DIR / "meta.json").write_text(
        json.dumps(
            {
                "env_variant": ENV_VARIANT,
                "methods": args.methods,
                "seeds": SEEDS,
                "n_problems": N_PROBLEMS,
                "device": args.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"cache ready at {CACHE_DIR}")


if __name__ == "__main__":
    main()
