"""W2 evidence-composition sweep (docs/failed_records_fix_part2.md §2), eval-only.

Does the record/evidence ranker do BETTER conditioning on fewer, more-recent candidate
failures? For each ``k`` in ``--ks`` (0 = uncapped), re-run the deployed rollout over the test
split with the failure EVIDENCE capped to the k most-recently-tried failures
(``deployed_rollout_traced(evidence_cap_k=k)``) -- the re-try mask and, on the X2 residual, the
|F| gate stay on the full context, so only the evidence memory shrinks. Report per-stratum FP vs
k. The sweep, not any single k, is the finding:

  * FP non-monotone at s2/s3, small-k < full  -> the model drowns composing many records ->
    the fix is X1 (compiled aggregation).
  * monotone (full <= every k)                 -> composition exonerated -> freeze.

No training, no checkpoints written. Primary arm = the X2 residual; a jointly-trained ``abl_records``
control (no gate) isolates whether the gate changes the story.

Usage::

    python experiments/spectre/w2_sweep.py \
        --arm "residual:checkpoints_spectre_noov_atoms_residual_records" \
        --arm "abl_records:checkpoints_spectre_noov_atoms_abl_records" \
        --ks 1 2 4 8 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from alphatamp.approaches.spectre.compare import stratum_of
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference import deployed_rollout_traced
from alphatamp.approaches.spectre.inference import load_checkpoint as load_v3
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]


def sweep_arm(model, episodes, strata, vocab, device, spec, deploy, ks):
    """Return {k: per-stratum-mean-FP array (len 4)} plus the ALL mean."""
    out: dict = {}
    for k in ks:
        cap = None if k == 0 else k
        fps: list[float] = []
        t0 = time.time()
        for j, ep in enumerate(episodes, 1):
            attempts, _ = deployed_rollout_traced(
                model, ep, vocab, device, spec=spec, evidence_cap_k=cap, **deploy
            )
            fps.append(float(attempts) - 1.0)
            if j % 50 == 0:
                el = time.time() - t0
                print(
                    f"    k={k:>2} [{j}/{len(episodes)}] {el:.0f}s "
                    f"(running mean FP {np.mean(fps):.2f})",
                    flush=True,
                )
        fps_a = np.array(fps)
        per = np.array(
            [
                fps_a[strata == s].mean() if (strata == s).any() else float("nan")
                for s in range(4)
            ]
        )
        out[k] = (per, float(fps_a.mean()))
        print(
            f"  k={k if k else 'full':>4}  ALL {fps_a.mean():6.2f}  "
            f"s0 {per[0]:5.2f}  s1 {per[1]:5.2f}  s2 {per[2]:5.2f}  s3 {per[3]:5.2f}",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-variant", default="dd2d_v4")
    ap.add_argument("--arm", action="append", default=[], help='"label:ckpt_subdir"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 0])
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
    strata = np.array([stratum_of(int(e.provenance.problem_id)) for e in episodes])
    print(
        f"# W2 evidence-cap sweep on {a.env_variant} test (n={len(episodes)}), "
        f"seed {a.seed}, ks={a.ks} (0=full)\n"
    )

    for entry in a.arm:
        label, _, subdir = entry.partition(":")
        ckpt = data / subdir / a.env_variant / f"seed_{a.seed}" / "best.pt"
        if not ckpt.is_file():
            print(f"!! missing {ckpt}")
            continue
        model, deploy = load_v3(ckpt, vocab, a.device)
        print(f"== {label} ({subdir}) ==", flush=True)
        res = sweep_arm(model, episodes, strata, vocab, a.device, spec, deploy, a.ks)
        # Compact FP-vs-k table for the arm.
        print(f"\n  {label}: FP vs k (per stratum)")
        print(f"    {'k':>5} {'ALL':>6} {'s0':>6} {'s1':>6} {'s2':>6} {'s3':>6}")
        for k in a.ks:
            per, allfp = res[k]
            kk = "full" if k == 0 else str(k)
            print(
                f"    {kk:>5} {allfp:6.2f} {per[0]:6.2f} {per[1]:6.2f} "
                f"{per[2]:6.2f} {per[3]:6.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
