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
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from alphatamp.approaches.spectre.dd2d_compare import stratum_of
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference_v3 import deployed_rollout_v3_traced
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v3 import (
    SpectreV3Model,
    V3Config,
    load_v2_checkpoint,
)
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]


def load_v3(ckpt: Path, vocab: Vocab, device: str) -> tuple[SpectreV3Model, dict]:
    """Rebuild a v3 model from its checkpoint, with dropout off for evaluation.

    Returns ``(model, overlap_mode)``: the mode is read back off the checkpoint rather
    than passed in, because deploying a model under a different ``overlap_mode`` than it
    trained under feeds it a feature column it has never seen populated (or blanks one
    it relies on) -- a silent train/deploy mismatch of exactly the kind §6.6 warns
    about.
    """
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = SpectreV3Model(
        n_ops=int(ck["n_ops"]),
        max_arity=vocab.max_operator_arity,
        cfg=V3Config(
            n_overlap_feats=2 if cfg.get("use_overlap") else 0,
            n_prior_feats=0,
            max_tags=int(cfg.get("max_tags", 32)),
            dropout_p=0.0,
            use_records=bool(cfg.get("use_records")),
            sinusoidal_pos=bool(cfg.get("sinusoidal_pos")),
        ),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval().to(device), {
        "overlap_mode": str(cfg.get("overlap_mode", "both")),
        "aggregate_records": bool(cfg.get("aggregate_records")),
    }


def score(
    model,
    episodes,
    vocab,
    device,
    spec,
    mode: Literal["permissive", "strict"],
    apply_demotion: bool = True,
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
    ap.add_argument("--mode", default="strict", choices=["strict", "permissive"])
    ap.add_argument(
        "--no-demotion",
        action="store_true",
        help="withhold the proof-demotion offset, measuring the model's own ordering "
        "(the eval-time axis of the G7 2x2)",
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
    results: dict[str, dict[int, float]] = {}
    for entry, is_v2 in specs:
        label, _, subdir = entry.partition(":")
        ckpt = data / subdir / a.env_variant / f"seed_{a.seed}" / "best.pt"
        if not ckpt.is_file():
            print(f"!! missing {ckpt}")
            continue
        if is_v2:
            model, _ = load_v2_checkpoint(ckpt)
            model = model.eval().to(a.device)
            ov_mode = {}
        else:
            model, ov_mode = load_v3(ckpt, vocab, a.device)
        # argparse `choices` already constrains this; `cast` tells mypy the same thing
        mode = cast(Literal["permissive", "strict"], a.mode)
        results[label] = score(
            model, episodes, vocab, a.device, spec, mode, not a.no_demotion, ov_mode
        )

    demo = "off" if a.no_demotion else f"on ({a.mode})"
    print(
        f"\n# {a.env_variant} test, uncensored deployed FP, "
        f"demotion={demo}, n={len(pids)}"
    )
    print(f"{'arm':<24} {'ALL':>8} {'s0':>8} {'s1':>8} {'s2':>8} {'s3':>8}")
    for label, fps in results.items():
        v = np.array([fps[p] for p in pids])
        cells = [f"{v.mean():8.2f}"] + [
            f"{v[strata == s].mean():8.2f}" for s in (0, 1, 2, 3)
        ]
        print(f"{label:<24} " + " ".join(cells))

    if a.baseline:
        base_label = a.baseline.partition(":")[0]
        base = np.array([results[base_label][p] for p in pids])
        print(f"\n# paired bootstrap vs '{base_label}' (negative = arm is better)")
        for label, fps in results.items():
            if label == base_label:
                continue
            v = np.array([fps[p] for p in pids])
            mean, (lo, hi) = paired_bootstrap(v, base)
            sig = "" if lo <= 0 <= hi else "  *CI excludes 0"
            print(
                f"  {label:<24} delta {mean:+7.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]{sig}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
