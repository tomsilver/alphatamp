"""Step-9 evaluation: run trained v2-static checkpoints on the test split, then the
elimination ladder + rollout-FP vs baselines (P1; P2 vs PIGINet noted as deferred).

Reports, over >= 3 checksum-distinct seeds:
  - η²(length) and the nested variance ladder (length -> +slack -> +proximity -> residual);
  - the paired rollout-FP margin v2-static − slack ordering on strata >= 2 (bootstrap CI);
  - mean rollout-FP of v2-static vs slack / default-order / LAZY / oracle.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from alphatamp.approaches.spectre import eda
from alphatamp.approaches.spectre.dataset_v2 import build_v2_example, collate_v2
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.ladder import beats_slack_paired, variance_ladder
from alphatamp.approaches.spectre.model_v2 import SpectreV2Model
from alphatamp.approaches.spectre.vocab import Vocab


def load_v2_model(ckpt_path: Path, vocab: Vocab, device: str) -> SpectreV2Model:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = SpectreV2Model(
        n_ops=ck["n_ops"],
        max_arity=vocab.max_operator_arity,
        max_tags=cfg["max_tags"],
        dropout_p=0.0,
    )
    model.load_state_dict(ck["state_dict"])
    return model.eval().to(device)


def v2_scores(model, episode, vocab, device) -> np.ndarray:
    ex = build_v2_example(episode, vocab, rng=None, max_tags=32)
    batch = collate_v2([ex], vocab.max_operator_arity).to(device)
    with torch.no_grad():
        logits, _ = model(batch)
    return logits[0].cpu().numpy()  # per-skeleton, pad = -inf


def _staged(skel) -> list[str]:
    return [
        op.parameters[0].name for op in skel.operator_seq if op.name == "place-buffer"
    ]


def episode_features(episode):
    """Per-skeleton (length, slack, proximity, feasible) + the problem stratum."""
    geo = episode.scene_geometry
    area = {o.name: float(o.area) for o in geo.objects}
    pose = {o.name: o.pose for o in geo.objects}
    buf = next((c for c in geo.containers if c.kind == "buffer"), None)
    buffer_area = (
        (buf.bounds[2] - buf.bounds[0]) * (buf.bounds[3] - buf.bounds[1])
        if buf
        else 1.0
    )
    lengths, slacks, prox, feas = [], [], [], []
    for skel, out in zip(episode.skeleton_pool, episode.outcomes):
        staged = _staged(skel)
        lengths.append(float(len(skel.operator_seq)))
        slacks.append(buffer_area - sum(area.get(s, 0.0) for s in staged))
        # proximity: mean pairwise distance among staged items (smaller = more crowded)
        d = []
        for a in range(len(staged)):
            for b in range(a + 1, len(staged)):
                pa, pb = pose.get(staged[a]), pose.get(staged[b])
                if pa and pb:
                    d.append(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        prox.append(float(np.mean(d)) if d else 0.0)
        feas.append(out.outcome == "success")
    feas = np.array(feas, dtype=bool)
    staged_counts = [len(_staged(s)) for s, f in zip(episode.skeleton_pool, feas) if f]
    stratum = min(staged_counts) if staged_counts else -1
    return (
        np.array(lengths),
        np.array(slacks),
        np.array(prox),
        feas,
        stratum,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step-9 v2 evaluation (ladder + P1/P2)")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args(argv)
    root = Path(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.from_json(root / "derived" / args.env / "train_vocab.json")
    test_eps = [
        load_episode(p) for p in list_episodes(root / "raw" / args.env / "test")
    ]
    test_eps = [
        e for e in test_eps if e.summary.num_success >= 1 and len(e.skeleton_pool) >= 2
    ]

    # precompute per-episode features (seed-independent)
    feats = [episode_features(e) for e in test_eps]
    lengths = np.concatenate([f[0] for f in feats])
    slacks = np.concatenate([f[1] for f in feats])
    prox = np.concatenate([f[2] for f in feats])
    strata = np.array([f[4] for f in feats])
    slack_per_ep = [f[1] for f in feats]
    feas_per_ep = [f[3] for f in feats]

    seeds = [int(s) for s in args.seeds.split(",")]
    ckpts = [
        root / "checkpoints_v2" / args.env / f"seed_{s}" / "best.pt" for s in seeds
    ]
    eda.assert_distinct_seed_checkpoints(ckpts)  # guard: distinct seeds
    print(f"# seed checkpoints distinct: {len(ckpts)} seeds", flush=True)

    ladder_rows, gate_rows, fp_rows = [], [], []
    for s, ck in zip(seeds, ckpts):
        model = load_v2_model(ck, vocab, device)
        scores_per_ep = [v2_scores(model, e, vocab, device) for e in test_eps]
        all_scores = np.concatenate(scores_per_ep)
        finite = np.isfinite(all_scores)
        rungs = variance_ladder(
            all_scores[finite], lengths[finite], slacks[finite], prox[finite]
        )
        ladder_rows.append(rungs)
        gate = beats_slack_paired(
            scores_per_ep, slack_per_ep, feas_per_ep, strata, min_stratum=2
        )
        gate_rows.append(gate)
        # mean rollout-FP: v2 vs slack vs default (per problem, strata>=2)
        mask = strata >= 2
        v2fp, slfp, defp = [], [], []
        for i, (sc, sl, fe) in enumerate(zip(scores_per_ep, slack_per_ep, feas_per_ep)):
            if not mask[i] or not fe.any():
                continue
            from alphatamp.approaches.spectre.ladder import _rollout_fp

            v2fp.append(_rollout_fp(np.argsort(-sc), fe))
            slfp.append(_rollout_fp(np.argsort(-sl), fe))
            defp.append(_rollout_fp(np.arange(len(fe)), fe))
        fp_rows.append((np.mean(v2fp), np.mean(slfp), np.mean(defp)))
        print(
            f"seed {s}: ladder[{rungs.as_row()}] | beat-slack Δ={gate['mean_diff']:.2f} "
            f"CI={tuple(round(x,2) for x in gate['ci'])} pass={gate['passes']} | "
            f"FP v2={fp_rows[-1][0]:.2f} slack={fp_rows[-1][1]:.2f} default={fp_rows[-1][2]:.2f}",
            flush=True,
        )

    eta_len = np.mean([r.r2_length for r in ladder_rows])
    residual = np.mean([r.residual for r in ladder_rows])
    mean_gate = np.mean([g["mean_diff"] for g in gate_rows])
    all_pass = all(g["passes"] for g in gate_rows)
    print("", flush=True)
    print(
        f"# LADDER (mean over {len(seeds)} seeds): η²(length)={eta_len:.3f}, "
        f"residual={residual:.3f}",
        flush=True,
    )
    print(
        f"# P1 gate: v2-static beats slack on strata>=2 by Δ={mean_gate:.2f} FP, "
        f"all-seeds-CI-excludes-zero={all_pass} → {'PASS' if all_pass else 'CHECK'}",
        flush=True,
    )
    print(
        "# P2 (>= PIGINet per stratum): deferred — needs PIGINet retrained on λ=0.8 data.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
