"""Step-11 acceptance (P5): does the learned typed-evidence pathway beat untyped
failure-counting?

For each test episode we run the deployed evidence rollout with facts **on** and **off**
(the *evidence increment* = FP_off − FP_on) and compare v2-evidence to the **LAZY** untyped
adaptive baseline (default-order prior − β·action-overlap-with-failed, β tuned on train —
the exact untyped failure-conditioning typing must beat). Reported per seed and averaged
over ≥3 checksum-distinct evidence checkpoints, paired stratified bootstrap, on strata ≥ 2
(where within-episode evidence can matter). The scramble gauge (facts-are-used detector) is
recomputed on each loaded ``best.pt`` (the training-log gauge is the final-epoch model).

**P5 passes** when, wherever the evidence increment is nonzero, v2-evidence ≤ LAZY with a
paired CI excluding zero. On-distribution the increment may be small (a strong static model
legitimately leaves little to recover, P-D); the CI-clean shift claim is Step 12.

    python experiments/spectre/spectre_eval_p5.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset_v2 import build_v2_example, collate_v2
from alphatamp.approaches.spectre.eda import (
    _lazy_rollout,
    assert_distinct_seed_checkpoints,
)
from alphatamp.approaches.spectre.evidence import evidence_rollout, scramble_gauge
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v2 import SpectreV2Model
from alphatamp.approaches.spectre.vocab import Vocab


def _gauge_batch(val_eps, vocab: Vocab, n: int = 24):
    """A fixed val batch with a nonempty failed context, for recomputing the gauge on
    the actual best.pt checkpoint (the training log's gauge is the final-epoch
    model)."""
    exs = []
    for ep in val_eps[:n]:
        fails = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if fails:
            exs.append(
                build_v2_example(
                    ep,
                    vocab,
                    rng=None,
                    evidence=True,
                    context_f=frozenset(fails[:3]),
                    augment_tags=False,
                )
            )
    return collate_v2(exs, max_arity=vocab.max_operator_arity) if exs else None


def _staged(skel) -> frozenset[str]:
    return frozenset(
        op.parameters[0].name for op in skel.operator_seq if op.name == "place-buffer"
    )


def _stratum(ep) -> int:
    subs = [_staged(s) for s in ep.skeleton_pool]
    feas = [o.outcome == "success" for o in ep.outcomes]
    return min((len(subs[i]) for i in range(len(subs)) if feas[i]), default=-1)


def _action_sets(ep) -> list[set]:
    """Per-skeleton ground-action set (the canonical 'these plans are similar' key)."""
    return [
        {(op.name,) + tuple(p.name for p in op.parameters) for op in s.operator_seq}
        for s in ep.skeleton_pool
    ]


def _load_model(ckpt_path: Path, vocab: Vocab, device: str) -> SpectreV2Model:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpectreV2Model(
        n_ops=ckpt["n_ops"],
        max_arity=vocab.max_operator_arity,
        max_tags=ckpt["cfg"]["max_tags"],
        dropout_p=ckpt["cfg"]["dropout_p"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _tune_lazy_beta(train_eps, budget: int) -> float:
    betas = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    best, best_mean = 0.0, float("inf")
    for b in betas:
        vals = [
            _lazy_rollout(_action_sets(e), [o.outcome for o in e.outcomes], b, budget)[
                0
            ]
            for e in train_eps
        ]
        m = float(np.mean(vals)) if vals else float("inf")
        if m < best_mean:
            best, best_mean = b, m
    return best


def _paired_ci(diff: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    if len(diff) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step-11 P5 acceptance")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-stratum", type=int, default=2)
    ap.add_argument("--budget", type=int, default=200)
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data_root)
    vocab = Vocab.from_json(root / "derived" / args.env / "train_vocab.json")

    ckpt_dir = root / "checkpoints_v2_evidence" / args.env
    ckpts = [ckpt_dir / f"seed_{s}" / "best.pt" for s in args.seeds]
    sums = assert_distinct_seed_checkpoints(ckpts)
    print(f"# checksum-distinct evidence checkpoints: {len(sums)}", flush=True)

    test_eps = [
        load_episode(p) for p in list_episodes(root / "raw" / args.env / "test")
    ]
    test_eps = [
        e
        for e in test_eps
        if e.scene_geometry is not None and e.summary.num_success >= 1
    ]
    train_eps = [
        load_episode(p) for p in list_episodes(root / "raw" / args.env / "train")
    ]
    val_eps = [load_episode(p) for p in list_episodes(root / "raw" / args.env / "val")]
    strata = np.array([_stratum(e) for e in test_eps])
    gauge_batch = _gauge_batch(val_eps, vocab)
    gauge_rng = np.random.default_rng(0)

    beta = _tune_lazy_beta(train_eps, args.budget)
    lazy_fp = np.array(
        [
            _lazy_rollout(
                _action_sets(e), [o.outcome for o in e.outcomes], beta, args.budget
            )[0]
            for e in test_eps
        ]
    )
    print(f"# LAZY beta*={beta}", flush=True)

    on_by_seed, off_by_seed, gauges = [], [], []
    for s in args.seeds:
        model = _load_model(ckpt_dir / f"seed_{s}" / "best.pt", vocab, device)
        fp_on = np.array(
            [
                evidence_rollout(model, e, vocab, device, use_facts=True)
                for e in test_eps
            ]
        )
        fp_off = np.array(
            [
                evidence_rollout(model, e, vocab, device, use_facts=False)
                for e in test_eps
            ]
        )
        on_by_seed.append(fp_on)
        off_by_seed.append(fp_off)
        gauges.append(
            scramble_gauge(model, gauge_batch, device, gauge_rng)
            if gauge_batch is not None
            else 0.0
        )
        m = strata >= args.min_stratum
        inc = (fp_off - fp_on)[m]
        vs_lazy = (lazy_fp - fp_on)[m]
        print(
            f"# seed {s}: gauge={gauges[-1]:.3f} | strata>={args.min_stratum} "
            f"evid_on={fp_on[m].mean():.2f} evid_off={fp_off[m].mean():.2f} "
            f"increment={inc.mean():+.2f} | LAZY={lazy_fp[m].mean():.2f} "
            f"(v2-LAZY {vs_lazy.mean():+.2f})",
            flush=True,
        )

    on = np.mean(on_by_seed, axis=0)
    off = np.mean(off_by_seed, axis=0)
    m = strata >= args.min_stratum
    inc_mean, inc_lo, inc_hi = _paired_ci((off - on)[m])
    vs_mean, vs_lo, vs_hi = _paired_ci((lazy_fp - on)[m])
    print(
        f"\n# === P5 (mean over {len(args.seeds)} seeds, strata>={args.min_stratum}, "
        f"n={int(m.sum())}) ===",
        flush=True,
    )
    print(
        f"# scramble gauge (final): {np.mean(gauges):.3f} ± {np.std(gauges):.3f}",
        flush=True,
    )
    print(
        f"# evidence increment (FP_off - FP_on): {inc_mean:+.2f} CI ({inc_lo:+.2f},{inc_hi:+.2f})",
        flush=True,
    )
    print(
        f"# v2-evidence vs LAZY (FP_lazy - FP_on): {vs_mean:+.2f} CI ({vs_lo:+.2f},{vs_hi:+.2f})",
        flush=True,
    )
    increment_nonzero = inc_lo > 0 or inc_mean > 0.05
    p5 = (not increment_nonzero) or (vs_lo > 0)
    print(
        f"# P5: {'PASS' if p5 else 'no'} "
        f"(increment {'nonzero' if increment_nonzero else '~0'}; "
        f"where nonzero, v2-evidence {'<=' if vs_lo > 0 else 'not <'} LAZY)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
