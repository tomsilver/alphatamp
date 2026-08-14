"""Score v3 checkpoints on the test split, uncensored, with paired bootstrap CIs.

The `val_fp` a run prints is a *selection* statistic: censored at a small attempt budget
and computed on a strided val subsample, because it is recomputed every epoch. It ranks
epochs; it is not the reported metric. This is the reported metric -- the full deployed
rollout over every test episode, with no budget -- and it is what a gate is judged on.

**Acceptance is a paired bootstrap over problems, not seed spread.** Development runs one
seed (three are reserved for the final paper numbers), so "within seed noise" is not
measurable. Comparing two methods on the *same* problems is both available now and more
powerful than a seed spread would be, and it is the instrument the P1/P4/P5 gates already
used.

Usage::

    python experiments/spectre/spectre_score.py \\
        --arm "records+overlap:checkpoints_spectre_g6_recON_ovON" \\
        --arm "records only:checkpoints_spectre_noov_g6_recON_ovOFF" \\
        --baseline "no records:checkpoints_spectre_norec_noov_g6_recOFF_ovOFF"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from alphatamp.approaches.spectre.compare import rollout_fp, stratum_of
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference import deployed_rollout_traced
from alphatamp.approaches.spectre.inference import load_checkpoint as load_v3
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]


def score(
    model,
    episodes,
    vocab,
    device,
    spec,
    deploy: dict | None = None,
) -> dict[int, float]:
    """Uncensored deployed FP per problem id."""
    out = {}
    for ep in episodes:
        attempts, _ = deployed_rollout_traced(
            model,
            ep,
            vocab,
            device,
            spec=spec,
            **(deploy or {}),
        )
        out[int(ep.provenance.problem_id)] = float(attempts) - 1.0
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0):
    """Mean of ``a - b`` with a 95% CI, resampling *problems* (shared indices).

    Paired on purpose: the two arms see identical problems, so pairing removes
    between-problem variance, which dominates between-arm variance here.
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def main(argv: list[str] | None = None) -> int:
    """Score every requested arm on the test split and print the comparison table."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-variant", default="dd2d_v4")
    ap.add_argument(
        "--test-variant",
        default=None,
        help="score on THIS variant's test episodes while loading the vocab, model "
        "config and checkpoints from --env-variant -- the train-old / test-new "
        "generalization eval (e.g. --env-variant dd2d_v4 --test-variant "
        "dd2d_v4gen_count). Stratum "
        "recovery is pid arithmetic (variant-independent) and the DD2D domain spec is "
        "shared across dd2d_* variants, so only the episodes change.",
    )
    ap.add_argument(
        "--astar-baseline",
        action="store_true",
        help="add an astar-dist (planner default-order) arm computed from each "
        "episode's stored plan order (score = -plan_idx) and use it as the "
        "paired-bootstrap baseline -- the SPECTREv3-vs-astar comparison. No checkpoint.",
    )
    ap.add_argument("--arm", action="append", default=[], help='"label:ckpt_subdir"')
    ap.add_argument("--baseline", help='"label:ckpt_subdir" to compare arms against')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="aggregate every arm over these seeds. A subdir may contain '{seed}' when "
        "the arm writes one directory per seed (e.g. checkpoints_spectre_v3final_s{seed}); "
        "the checkpoint path's own seed_<n> component is substituted regardless, which "
        "is what lets a single-directory arm aggregate too. Missing seeds are skipped "
        "with a warning. Reports mean +- std ACROSS SEEDS of the per-stratum mean, "
        "which is the spread a gate is judged on.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    data = REPO / "data" / "spectre"
    # Vocab, model config and checkpoints come from the *training* variant; only the test
    # episodes come from --test-variant (train-old / test-new generalization). The DD2D
    # domain spec is shared across dd2d_* variants, so the train variant's spec is used.
    test_variant = a.test_variant or a.env_variant
    vocab = Vocab.from_json(data / "derived" / a.env_variant / "train_vocab.json")
    spec = spec_for(a.env_variant)
    episodes = [
        e
        for e in (
            load_episode(p) for p in list_episodes(data / "raw" / test_variant / "test")
        )
        if e.scene_geometry is not None
    ]
    pids = [int(e.provenance.problem_id) for e in episodes]
    strata = np.array([stratum_of(p) for p in pids])

    specs = list(a.arm) + ([a.baseline] if a.baseline else [])
    # per label: one dict[problem_id -> FP] per seed that actually had a checkpoint
    results: dict[str, list[dict[int, float]]] = {}
    for entry in specs:
        label, _, subdir = entry.partition(":")
        # `--seeds` applies to every arm: the checkpoint path's own `seed_<n>` component
        # always varies, and `{seed}` in the subdir is substituted as well when the arm
        # writes one directory per seed. Missing seeds are reported and skipped, so an
        # arm with fewer seeds than another still aggregates over what exists.
        seeds = a.seeds or [a.seed]
        for sd in seeds:
            path = subdir.replace("{seed}", str(sd))
            ckpt = data / path / a.env_variant / f"seed_{sd}" / "best.pt"
            if not ckpt.is_file():
                print(f"!! missing {ckpt}")
                continue
            # `train_*` rewrites best.pt every time selection improves, so a checkpoint
            # from a *running* job is a mid-training model that scores like a bad one.
            # Reading one silently is how a baseline gets unfairly flattered; it nearly
            # happened here (a 3-seed v2.2 read 17.56 while two seeds were at epoch 10).
            age = time.time() - ckpt.stat().st_mtime
            if age < 120:
                print(
                    f"!! {ckpt} was written {age:.0f}s ago — a run may still own it; "
                    f"this is a MID-TRAINING model, not a result"
                )
            model, ov_mode = load_v3(ckpt, vocab, a.device)
            results.setdefault(label, []).append(
                score(
                    model,
                    episodes,
                    vocab,
                    a.device,
                    spec,
                    ov_mode,
                )
            )

    if a.astar_baseline:
        # astar-dist: the planner's default enumeration order, score = -plan_idx, scored
        # by the same rollout_fp accounting every static method shares (compare.py). No
        # checkpoint -- read straight off each episode's stored outcomes.
        astar_fp: dict[int, float] = {}
        for ep in episodes:
            labels = [1.0 if o.outcome == "success" else 0.0 for o in ep.outcomes]
            scores = [float(-j) for j in range(len(ep.outcomes))]
            fp = rollout_fp(scores, labels)
            if fp is not None:  # every kept problem has >=1 feasible, so this holds
                astar_fp[int(ep.provenance.problem_id)] = float(fp)
        results["astar-dist"] = [astar_fp]

    title = (
        a.env_variant if not a.test_variant else f"{a.env_variant}->{a.test_variant}"
    )
    print(f"\n# {title} test, uncensored deployed FP, n={len(pids)}")
    wide = any(len(v) > 1 for v in results.values())
    w = 15 if wide else 8
    print(
        f"{'arm':<26}" + "".join(f"{c:>{w}}" for c in ["ALL", "s0", "s1", "s2", "s3"])
    )
    for label, runs in results.items():
        mat = np.stack([[r[p] for p in pids] for r in runs])  # (n_seeds, n_problems)
        cells = []
        for sel in [np.ones_like(strata, bool)] + [strata == s for s in (0, 1, 2, 3)]:
            per_seed = mat[:, sel].mean(axis=1)
            cells.append(
                f"{per_seed.mean():.2f} ± {per_seed.std(ddof=1):.2f}"
                if len(per_seed) > 1
                else f"{per_seed.mean():.2f}"
            )
        n = f" [{len(runs)} seeds]" if len(runs) > 1 else ""
        print(f"{label + n:<26}" + "".join(f"{c:>{w}}" for c in cells))

    base_label = None
    if a.astar_baseline:
        base_label = "astar-dist"
    elif a.baseline:
        base_label = a.baseline.partition(":")[0]
    if base_label:
        base = np.stack([[r[p] for p in pids] for r in results[base_label]]).mean(0)
        print(f"\n# paired bootstrap vs '{base_label}' (negative = arm is better)")
        for label, fps in results.items():
            if label == base_label:
                continue
            v = np.stack([[r[p] for p in pids] for r in fps]).mean(0)
            mean, (lo, hi) = paired_bootstrap(v, base)
            sig = "" if lo <= 0 <= hi else "  *CI excludes 0"
            print(
                f"  {label:<24} delta {mean:+7.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]{sig}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
