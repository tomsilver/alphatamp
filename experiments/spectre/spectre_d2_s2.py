"""D2 — why does the deployed ranker lose stratum 2?

s2 is the one stratum where the learned ranker trails the non-learned planner order
(dd2d_v4: 26.00 vs astar's 17.08). The v3 plan's headline addition -- necessity
conditioning -- only helps if the loss is **cross-length**: the ranker picking plans of the
wrong size. If instead the ranker picks the right size and cannot tell *which* subset of
that size works, the bottleneck is within-length discrimination, and conditioning on
predicted difficulty cannot fix it.

**Everything is compared like with like.** The obvious version of this diagnostic -- a
length-restricted *static* score against astar's *unrestricted* 17.08 -- mixes three
differences at once (static vs adaptive, restricted vs full pool, model vs planner) and can
report an "oracle" that is worse than the thing it bounds. So every row here is a rollout
over the same pool restriction:

* ``deployed``           -- the adaptive ranker, full pool (what gets reported)
* ``length-oracle``      -- the adaptive ranker, but only minimum-length candidates are
  attemptable: the counterfactual where cross-length error is removed entirely
* ``astar``              -- planner order, full pool
* ``astar length-oracle``-- planner order restricted the same way: the fair reference for
  the row above, since restricting the pool helps *any* method

The fork, pre-registered:

* **length-oracle << deployed, and <~ astar length-oracle** => the ranker orders correctly
  *within* a length and gets the length wrong. Necessity conditioning (G8) is the right
  mechanism.
* **length-oracle ~ deployed, or >> astar length-oracle** => even handed the correct plan
  length the ranker cannot find the feasible subset. G8 alone will not save s2, and the
  honest conclusion is a representation finding rather than a fix.

**The length oracle is a diagnostic, never a method.** It restricts the pool using the
episode's minimal feasible length -- i.e. the stratum, which is the *answer*. Nothing here
may reach a model input; it exists only to bound what better length calibration could buy.

Usage::

    python experiments/spectre/spectre_d2_s2.py --env-variant dd2d_v4
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from alphatamp.approaches.spectre.dd2d_compare import stratum_of
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference_v3 import deployed_rollout_v3_traced
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v3 import load_v2_checkpoint
from alphatamp.approaches.spectre.necessity import necessity_labels
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]


def _static_scores(model, episode, vocab, device) -> np.ndarray:
    """Empty-context (t = 0) logits -- the ranker's opening opinion."""
    from alphatamp.approaches.spectre.dataset_v3 import build_v3_example, collate_v3

    example, records = build_v3_example(
        episode,
        vocab,
        rng=None,
        evidence=True,
        context_f=frozenset(),
        augment_tags=False,
    )
    batch = collate_v3(
        [example], max_arity=vocab.max_operator_arity, records=[records]
    ).to(device)
    with torch.no_grad():
        logits, _ = model(batch)
    return logits[0].detach().cpu().numpy().astype(float)


def _fp_from_order(order: np.ndarray, feasible: np.ndarray) -> float | None:
    """Failed attempts before the first feasible candidate, given an attempt order."""
    for i, idx in enumerate(order):
        if feasible[idx]:
            return float(i)
    return None


def _restricted_fp(
    scores: np.ndarray, sizes: np.ndarray, feasible: np.ndarray, min_size: int
) -> float | None:
    """FP when only minimum-length candidates may be attempted, ranked by ``scores``."""
    keep = np.flatnonzero(sizes == min_size)
    if keep.size == 0 or not feasible[keep].any():
        return None
    order = keep[np.argsort(-scores[keep], kind="stable")]
    return _fp_from_order(order, feasible)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-variant", default="dd2d_v4")
    ap.add_argument("--ckpt-subdir", default="checkpoints_v2_evidence_ov")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratum", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    data = REPO / "data" / "spectre"
    vocab = Vocab.from_json(data / "derived" / a.env_variant / "train_vocab.json")
    model, _ = load_v2_checkpoint(
        data / a.ckpt_subdir / a.env_variant / f"seed_{a.seed}" / "best.pt"
    )
    model.to(a.device)
    spec = spec_for(a.env_variant)

    real_fp, oracle_fp = [], []
    astar_fp, astar_oracle_fp = [], []
    n_minlen, pool_sizes = [], []
    wrong_len, right_len = 0, 0
    failure_modes: Counter = Counter()
    dhat_err = []

    for path in list_episodes(data / "raw" / a.env_variant / "test"):
        episode = load_episode(path)
        pid = int(episode.provenance.problem_id)
        if episode.scene_geometry is None or stratum_of(pid) != a.stratum:
            continue

        subsets = spec.subsets(episode)
        sizes = np.array([len(s) for s in subsets])
        feasible = np.array([o.outcome == "success" for o in episode.outcomes])
        if not feasible.any():
            continue
        min_size = int(sizes[feasible].min())

        attempts, trace = deployed_rollout_v3_traced(
            model, episode, vocab, a.device, spec=spec
        )
        real_fp.append(float(attempts) - 1.0)

        # attribute each *failed* attempt: wrong plan length, or right length but the
        # wrong subset of that length?
        for idx in trace.order[:-1]:
            if sizes[idx] != min_size:
                wrong_len += 1
            else:
                right_len += 1
            meta = episode.outcomes[idx].refiner_metadata or {}
            failure_modes[str(meta.get("failure_action", "?")).split("(")[0]] += 1

        # Length oracle: the model's own ranking, restricted to minimum-length
        # candidates. Uses the *adaptive* first-step scores so it is the same ranker as
        # the deployed row, differing only in which candidates may be attempted.
        scores = _static_scores(model, episode, vocab, a.device)
        ofp = _restricted_fp(scores, sizes, feasible, min_size)
        if ofp is not None:
            oracle_fp.append(ofp)

        # astar = the planner's own enumeration order = pool index order. Restricting it
        # the same way is the fair reference: a smaller pool helps *any* method.
        idx_order = np.arange(len(sizes))
        afp = _fp_from_order(idx_order, feasible)
        aofp = _restricted_fp(-idx_order.astype(float), sizes, feasible, min_size)
        if afp is not None:
            astar_fp.append(afp)
        if aofp is not None:
            astar_oracle_fp.append(aofp)
        n_minlen.append(int((sizes == min_size).sum()))
        pool_sizes.append(len(sizes))

        labels = necessity_labels(episode, spec)
        if labels is not None:
            dhat_err.append(labels.d_hat - min_size)

    n = len(real_fp)
    total_failed = wrong_len + right_len
    print(f"=== D2: stratum {a.stratum} on {a.env_variant} (n={n} episodes) ===\n")
    print(
        f"pool: {np.mean(pool_sizes):.0f} candidates, of which "
        f"{np.mean(n_minlen):.0f} are minimum-length ({100*np.mean(n_minlen)/np.mean(pool_sizes):.0f}%)\n"
    )
    print(f"{'method':<26} {'full pool':>10} {'length-oracle':>14}")
    print(
        f"{'SPECTREv2 (adaptive)':<26} {np.mean(real_fp):>10.2f} "
        f"{np.mean(oracle_fp):>14.2f}"
    )
    print(
        f"{'astar-dist (planner order)':<26} {np.mean(astar_fp):>10.2f} "
        f"{np.mean(astar_oracle_fp):>14.2f}"
    )
    print()
    print("failed attempts by cause:")
    print(
        f"  wrong plan length         : {wrong_len:5d}  ({100*wrong_len/max(total_failed,1):.1f}%)"
    )
    print(
        f"  right length, wrong subset: {right_len:5d}  ({100*right_len/max(total_failed,1):.1f}%)"
    )
    print(f"\nrefiner failure modes among attempts: {dict(failure_modes)}")
    if dhat_err:
        print(
            f"\nnecessity d_hat - true min size: mean {np.mean(dhat_err):+.4f} "
            f"max|.| {np.max(np.abs(dhat_err)):.4f}"
        )

    deployed, oracle = np.mean(real_fp), np.mean(oracle_fp)
    astar_o = np.mean(astar_oracle_fp)
    # Cross-length is the bottleneck only if removing it recovers most of the gap AND the
    # ranker's within-length order is competitive with the planner's on the same pool.
    recovers = oracle <= 0.5 * deployed
    competitive = oracle <= astar_o * 1.25
    print("\n" + "=" * 72)
    print(
        f"length-oracle recovers {100*(deployed-oracle)/max(deployed,1e-9):.0f}% of the "
        f"deployed FP; vs astar on the same restricted pool: {oracle:.2f} vs {astar_o:.2f}"
    )
    if recovers and competitive:
        print(
            "FORK => CROSS-LENGTH calibration is the bottleneck.\n"
            "        Necessity conditioning (G8) is the right mechanism."
        )
    else:
        print(
            "FORK => WITHIN-LENGTH discrimination is the bottleneck.\n"
            "        G8 alone will NOT fix s2; report it as a representation finding."
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
