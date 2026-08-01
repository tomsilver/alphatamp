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

    python experiments/spectre/spectre_score_v3.py \\
        --arm "records+overlap:checkpoints_v3_g6_recON_ovON" \\
        --arm "records only:checkpoints_v3_noov_g6_recON_ovOFF" \\
        --v2-arm "v2.2 yardstick:checkpoints_v2_evidence_ov" \\
        --baseline "no records:checkpoints_v3_norec_noov_g6_recOFF_ovOFF"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from alphatamp.approaches.spectre.compare import stratum_of
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference_v3 import (
    deployed_rollout_v3_traced,
)
from alphatamp.approaches.spectre.inference_v3 import load_v3_checkpoint as load_v3
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v3 import load_v2_checkpoint
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]


def score(
    model,
    episodes,
    vocab,
    device,
    spec,
    mode: Literal["permissive", "strict"],
    apply_demotion: bool = False,
    deploy: dict | None = None,
) -> dict[int, float]:
    """Uncensored deployed FP per problem id."""
    out = {}
    for ep in episodes:
        attempts, _ = deployed_rollout_v3_traced(
            model,
            ep,
            vocab,
            device,
            spec=spec,
            mode=mode,
            apply_demotion=apply_demotion,
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
    ap.add_argument("--arm", action="append", default=[], help='"label:ckpt_subdir"')
    ap.add_argument(
        "--v2-arm",
        action="append",
        default=[],
        help='"label:ckpt_subdir" for a train_v2 checkpoint, loaded in compat mode '
        "(D-8) so the yardstick is scored by this instrument on these episodes -- "
        "which is what makes a v3-vs-v2.2 paired bootstrap meaningful rather than a "
        "comparison of two separately-produced numbers",
    )
    ap.add_argument("--baseline", help='"label:ckpt_subdir" to compare arms against')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="aggregate every arm over these seeds. A subdir may contain '{seed}' when "
        "the arm writes one directory per seed (e.g. checkpoints_v3_v3final_s{seed}); "
        "the checkpoint path's own seed_<n> component is substituted regardless, which "
        "is what lets a single-directory arm aggregate too. Missing seeds are skipped "
        "with a warning. Reports mean +- std ACROSS SEEDS of the per-stratum mean, "
        "which is the spread a gate is judged on.",
    )
    ap.add_argument("--mode", default="strict", choices=["strict", "permissive"])
    ap.add_argument(
        "--with-demotion",
        action="store_true",
        help="re-enable the proof-demotion offset. OFF by default since 2026-07-30: the "
        "deployed method is a purely learned ranker and nothing outside the network "
        "touches its ordering. Worth 0.23 FP on DD2D; kept available because the "
        "deduction is sound and a domain whose proofs fire more often may want it",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    data = REPO / "data" / "spectre"
    vocab = Vocab.from_json(data / "derived" / a.env_variant / "train_vocab.json")
    spec = spec_for(a.env_variant)
    episodes = [
        e
        for e in (
            load_episode(p)
            for p in list_episodes(data / "raw" / a.env_variant / "test")
        )
        if e.scene_geometry is not None
    ]
    pids = [int(e.provenance.problem_id) for e in episodes]
    strata = np.array([stratum_of(p) for p in pids])

    specs = [(e, False) for e in list(a.arm) + ([a.baseline] if a.baseline else [])]
    specs += [(e, True) for e in a.v2_arm]
    # per label: one dict[problem_id -> FP] per seed that actually had a checkpoint
    results: dict[str, list[dict[int, float]]] = {}
    for entry, is_v2 in specs:
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
            if is_v2:
                model, _ = load_v2_checkpoint(ckpt)
                model = model.eval().to(a.device)
                ov_mode: dict = {}
            else:
                model, ov_mode = load_v3(ckpt, vocab, a.device)
            # argparse `choices` constrains this; `cast` tells mypy the same
            mode = cast(Literal["permissive", "strict"], a.mode)
            results.setdefault(label, []).append(
                score(
                    model,
                    episodes,
                    vocab,
                    a.device,
                    spec,
                    mode,
                    a.with_demotion,
                    ov_mode,
                )
            )

    demo = f"on ({a.mode})" if a.with_demotion else "off (deployed default)"
    print(
        f"\n# {a.env_variant} test, uncensored deployed FP, "
        f"demotion={demo}, n={len(pids)}"
    )
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

    if a.baseline:
        base_label = a.baseline.partition(":")[0]
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
