"""Precompute + cache per-problem test scores for the DD2D method comparison.

Runs each method once on the DD2D test split and writes the *raw per-skeleton
scores* per problem so ``experiments/spectre/compare_dd2d_methods.py`` can load
them and derive rollout-FPs (and other metrics) without re-running any inference.

Cache layout under ``data/spectre/derived/dd2d_v2/compare_cache/`` (one JSON per
problem, keyed by the integer seed = ``problem_id``; resumable — existing files
are skipped unless ``--force``):

    astar/<pid>.json                    {problem_id, stratum, scores, labels}
    piginet_v3/<pid>.json               {problem_id, stratum, scores, labels}
    spectre_static/seed_<s>/<pid>.json  {problem_id, stratum, scores, labels}
    spectre_adaptive/seed_<s>/<pid>.json{problem_id, stratum, fp, order}
        (``order`` = the realized attempt sequence of pool indices, until first
        success; consumed by the notebook's T0 length-ladder)
    spectre_lenctx/seed_<s>/<pid>.json  {problem_id, stratum, fp, order}
        (T1 length-only-context intervention: adaptive rollout with identity-
        scrambled same-length failure contexts; fp = mean over surrogate draws)

Usage::

    python experiments/spectre/precompute_dd2d_cache.py            # all methods
    python experiments/spectre/precompute_dd2d_cache.py --force
    python experiments/spectre/precompute_dd2d_cache.py --methods piginet spectre

This is a bridge driver: it imports the vendored ``piginet.eval`` scorer, so it is
excluded from strict mypy/pylint like the marimo notebook it feeds.
"""

from __future__ import annotations

import argparse
import json
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
SPECTRE_TEST = REPO / "data" / "spectre" / "raw" / "dd2d_v2" / "test"
VOCAB_PATH = REPO / "data" / "spectre" / "derived" / "dd2d_v2" / "train_vocab.json"
CKPT_DIR = REPO / "data" / "spectre" / "checkpoints" / "dd2d_v2"
PIGINET_CKPT = DD2D / "out_dd2d" / "piginet_v3" / "ckpt.pt"
PIGINET_DATA = DD2D / "data" / "dd2d" / "raw_v2"
PIGINET_CACHE = DD2D / "out_dd2d" / "clip_cache_v2"
CACHE_DIR = REPO / "data" / "spectre" / "derived" / "dd2d_v2" / "compare_cache"
SEEDS = [0, 1, 2]
N_PROBLEMS = 124


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
    """PIGINet_v3: fresh inference; cache per-skeleton logits + labels."""
    out = CACHE_DIR / "piginet_v3"
    if _dir_complete(out) and not force:
        print("[piginet_v3] complete; skipping")
        return
    # Local import: vendored piginet stack (pulls in open_clip / CLIP).
    from alphatamp.approaches.spectre.envs.dd2d.piginet.eval import score_split

    print("[piginet_v3] running fresh inference on test split ...")
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
    print(f"[piginet_v3] wrote {n} problems -> {out}")


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
        default=["astar", "piginet", "spectre"],
        choices=["astar", "piginet", "spectre", "lenctx"],
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--lenctx-repeats",
        type=int,
        default=3,
        help="Surrogate draws per seed for the T1 length-only-context intervention",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if "astar" in args.methods:
        cache_astar(args.force)
    if "piginet" in args.methods:
        cache_piginet(args.force, args.device)
    if "spectre" in args.methods:
        cache_spectre(args.force, args.device)
    if "lenctx" in args.methods:
        cache_lenctx(args.force, args.device, repeats=args.lenctx_repeats)

    (CACHE_DIR / "meta.json").write_text(
        json.dumps(
            {
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
