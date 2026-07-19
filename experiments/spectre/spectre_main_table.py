"""Consolidated in-distribution main table (λ=0.8 dd2d_v2 test split, existing checkpoints).

Every method is scored by the same uncensored rollout (budget = pool cap) and reported as
mean **attempts** to first success (1-indexed; FP = attempts − 1) overall and per stratum,
plus mean refinement wall-clock of the attempted prefix. Learned models average over their
≥3 checksum-distinct seeds. No new training — this reuses the Step 9 (v2-static) and Step 11
(v2-evidence) checkpoints, the Step 10 proof-demotion, and the eda LAZY baseline.

    python experiments/spectre/spectre_main_table.py

PIGINet (the low-level P2 comparator) is intentionally absent (deferred, needs renders +
CNN training); every method here is available from stored checkpoints/geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset_v2 import build_v2_example, collate_v2
from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (
    target_blocked_after_removing,
)
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v2 import SpectreV2Model
from alphatamp.approaches.spectre.proof_demotion import ProofState, demote
from alphatamp.approaches.spectre.vocab import Vocab


def _staged(skel) -> frozenset[str]:
    return frozenset(
        op.parameters[0].name for op in skel.operator_seq if op.name == "place-buffer"
    )


def _stratum(ep) -> int:
    subs = [_staged(s) for s in ep.skeleton_pool]
    feas = [o.outcome == "success" for o in ep.outcomes]
    return min((len(subs[i]) for i in range(len(subs)) if feas[i]), default=-1)


def _wall(ep, tried: list[int]) -> float:
    return float(sum(ep.outcomes[i].refinement_wall_clock_s for i in tried))


def _metrics(ep, tried: list[int]) -> tuple[int, float]:
    """(attempts, wall) from the attempted sequence ending in the first success."""
    return len(tried), _wall(ep, tried)


# --------------------------------------------------------------------------- #
# non-learned rollouts (return the attempted index sequence, success last)
# --------------------------------------------------------------------------- #
def _rollout_order(ep, order: list[int]) -> list[int]:
    tried: list[int] = []
    for idx in order:
        tried.append(idx)
        if ep.outcomes[idx].outcome == "success":
            return tried
    return tried


def _slack_order(ep) -> list[int]:
    sg = ep.scene_geometry
    area = {o.name: float(o.area) for o in sg.objects}
    buf = next((c for c in sg.containers if c.kind == "buffer"), None)
    buf_area = (
        (buf.bounds[2] - buf.bounds[0]) * (buf.bounds[3] - buf.bounds[1])
        if buf
        else 1.0
    )
    slack = [
        buf_area - sum(area.get(s, 0.0) for s in _staged(sk)) for sk in ep.skeleton_pool
    ]
    return sorted(range(len(ep.skeleton_pool)), key=lambda i: -slack[i])


def _lazy_order_rollout(ep, beta: float) -> list[int]:
    action_sets = [
        {(op.name,) + tuple(p.name for p in op.parameters) for op in s.operator_seq}
        for s in ep.skeleton_pool
    ]
    remaining = [i for i, o in enumerate(ep.outcomes) if o.outcome != "error"]
    failed: list[set] = []
    tried: list[int] = []
    while remaining:
        pick = max(
            remaining,
            key=lambda i: (
                -float(i)
                - beta * max((len(action_sets[i] & f) for f in failed), default=0),
                -i,
            ),
        )
        tried.append(pick)
        if ep.outcomes[pick].outcome == "success":
            return tried
        failed.append(action_sets[pick])
        remaining.remove(pick)
    return tried


def _handrule_rollout(ep) -> list[int]:
    """Default order + sound proof-demotion of blocked-at-contents subsets (Step 10)."""
    subsets = [_staged(s) for s in ep.skeleton_pool]
    blocked: dict = {}

    def is_blocked(fs) -> bool:
        if fs not in blocked:
            blocked[fs] = target_blocked_after_removing(ep.scene_geometry, fs)
        return blocked[fs]

    state = ProofState(subsets=subsets)
    remaining = list(range(len(subsets)))
    tried: list[int] = []
    while remaining:
        remaining = demote(remaining, state.dead)
        idx = remaining.pop(0)
        tried.append(idx)
        if ep.outcomes[idx].outcome == "success":
            return tried
        if is_blocked(subsets[idx]):
            state.observe_failure(idx, blocked=True, pack_impossible=False)
    return tried


def _random_attempts(ep, rng: np.random.Generator, n: int = 200) -> tuple[float, float]:
    order = list(range(len(ep.skeleton_pool)))
    ats, walls = [], []
    for _ in range(n):
        rng.shuffle(order)
        tried = _rollout_order(ep, list(order))
        a, w = _metrics(ep, tried)
        ats.append(a)
        walls.append(w)
    return float(np.mean(ats)), float(np.mean(walls))


# --------------------------------------------------------------------------- #
# learned-model rollout (tried sequence, tracking wall)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _model_rollout(model, ep, vocab, device, use_facts: bool) -> list[int]:
    success = {i for i, o in enumerate(ep.outcomes) if o.outcome == "success"}
    k = len(ep.skeleton_pool)
    tried: list[int] = []
    while len(tried) < k:
        ex = build_v2_example(
            ep,
            vocab,
            rng=None,
            evidence=True,
            context_f=frozenset(tried),
            hide_facts=not use_facts,
            augment_tags=False,
        )
        batch = collate_v2([ex], max_arity=vocab.max_operator_arity).to(device)
        logits, _ = model(batch)
        row = logits[0].clone()
        if tried:
            row[tried] = float("-inf")
        pick = int(torch.argmax(row).item())
        tried.append(pick)
        if pick in success:
            return tried
    return tried


def _load(ckpt_path: Path, vocab: Vocab, device: str) -> SpectreV2Model:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpectreV2Model(
        n_ops=ckpt["n_ops"],
        max_arity=vocab.max_operator_arity,
        max_tags=ckpt["cfg"]["max_tags"],
        dropout_p=ckpt["cfg"]["dropout_p"],
    ).to(device)
    model.load_state_dict(
        ckpt["state_dict"], strict=False
    )  # v2-static lacks fact weights
    model.eval()
    return model


def _tune_lazy_beta(train_eps) -> float:
    best, best_mean = 0.0, float("inf")
    for b in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        m = float(np.mean([len(_lazy_order_rollout(e, b)) for e in train_eps]))
        if m < best_mean:
            best, best_mean = b, m
    return best


def _summary(name: str, attempts, walls, strata: np.ndarray, out: list) -> None:
    a = np.asarray(attempts, float)
    row = {"method": name, "all": a.mean(), "wall": float(np.mean(walls))}
    for s in (0, 1, 2, 3):
        m = strata == s
        row[f"s{s}"] = a[m].mean() if m.any() else float("nan")
    out.append(row)
    print(
        f"{name:24s} all={row['all']:6.2f} "
        + " ".join(f"s{s}={row[f's{s}']:6.2f}" for s in (0, 1, 2, 3))
        + f"  wall={row['wall']:.1f}s",
        flush=True,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="In-distribution main table")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data_root)
    vocab = Vocab.from_json(root / "derived" / args.env / "train_vocab.json")

    test = [load_episode(p) for p in list_episodes(root / "raw" / args.env / "test")]
    test = [
        e for e in test if e.scene_geometry is not None and e.summary.num_success >= 1
    ]
    train = [load_episode(p) for p in list_episodes(root / "raw" / args.env / "train")]
    strata = np.array([_stratum(e) for e in test])
    print(
        f"# test episodes: {len(test)} | strata "
        + str({s: int((strata == s).sum()) for s in (0, 1, 2, 3)}),
        flush=True,
    )

    rows: list = []
    rng = np.random.default_rng(0)

    # oracle: a perfect ranker tries a success first (attempts = 1).
    _summary(
        "oracle",
        np.ones(len(test)),
        [_wall(e, [e.summary.first_success_idx or 0]) for e in test],
        strata,
        rows,
    )

    rnd = [_random_attempts(e, rng) for e in test]
    _summary("random", [r[0] for r in rnd], [r[1] for r in rnd], strata, rows)

    dft = [
        _metrics(e, _rollout_order(e, list(range(len(e.skeleton_pool))))) for e in test
    ]
    _summary("default-order", [d[0] for d in dft], [d[1] for d in dft], strata, rows)

    slk = [_metrics(e, _rollout_order(e, _slack_order(e))) for e in test]
    _summary("slack-order", [s[0] for s in slk], [s[1] for s in slk], strata, rows)

    beta = _tune_lazy_beta(train)
    lz = [_metrics(e, _lazy_order_rollout(e, beta)) for e in test]
    _summary(f"LAZY(β={beta})", [x[0] for x in lz], [x[1] for x in lz], strata, rows)

    hr = [_metrics(e, _handrule_rollout(e)) for e in test]
    _summary("hand-rule(proofs)", [x[0] for x in hr], [x[1] for x in hr], strata, rows)

    for label, sub, facts in [
        ("v2-static", "checkpoints_v2", False),
        ("v2-evidence", "checkpoints_v2_evidence", True),
    ]:
        seed_at, seed_wl = [], []
        for s in args.seeds:
            ckpt = root / sub / args.env / f"seed_{s}" / "best.pt"
            if not ckpt.exists():
                continue
            model = _load(ckpt, vocab, device)
            mm = [
                _metrics(e, _model_rollout(model, e, vocab, device, facts))
                for e in test
            ]
            seed_at.append(np.array([m[0] for m in mm], float))
            seed_wl.append(np.array([m[1] for m in mm], float))
        if seed_at:
            _summary(
                label, np.mean(seed_at, axis=0), np.mean(seed_wl, axis=0), strata, rows
            )

    # markdown table
    print("\n| method | all | s0 | s1 | s2 | s3 | wall(s) |", flush=True)
    print("|---|---|---|---|---|---|---|", flush=True)
    for r in rows:
        print(
            f"| {r['method']} | {r['all']:.2f} | {r['s0']:.2f} | {r['s1']:.2f} | "
            f"{r['s2']:.2f} | {r['s3']:.2f} | {r['wall']:.1f} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
