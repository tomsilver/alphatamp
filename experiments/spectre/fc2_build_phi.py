"""Build the F-C2 rollout-aligned |F| curriculum map: phi_e per training episode.

phi_e = a reference policy's *deployed* FP (uncensored attempts - 1) on episode e. A
deployed rollout that ends at attempt phi_e + 1 queries the ranker exactly once at each
|F| = 0, 1, ..., phi_e, so training the context sampler with Uniform{0..phi_e} matches
that rollout's visit distribution (dataset.sample_context, --context-mode rollout). The
reference policy is the *current* deployed checkpoint of the arm being improved (a single
curriculum/DAgger iteration): the map reweights the NEXT training by what the CURRENT
policy actually visits, which is the honest reading of the 2026-08-22 rollout-alignment
guardrail. Using the arm's own checkpoint (not the scalars-on ceiling, whose FP tail is
thinner) keeps the tail the arm truly reaches.

Read-only w.r.t. training data; writes one JSON {problem_id: phi_e} under
data/spectre/derived/<env-variant>/. Emits an ETA heartbeat (working-practices: long
runs self-monitored, never a fixed foreground timer).

Usage::

    python experiments/spectre/fc2_build_phi.py \
        --env-variant dd2d_v4 \
        --arm-subdir checkpoints_spectre_atoms_fr_join --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.inference import deployed_rollout_traced
from alphatamp.approaches.spectre.inference import load_checkpoint as load_v3
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-variant", default="dd2d_v4")
    ap.add_argument(
        "--arm-subdir",
        required=True,
        help="reference-policy checkpoint dir, e.g. checkpoints_spectre_atoms_fr_join",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: derived/<env>/fc2_phi_<split>_<label>.json)",
    )
    ap.add_argument("--label", default=None, help="tag for the default output filename")
    a = ap.parse_args(argv)

    data = REPO / "data" / "spectre"
    vocab = Vocab.from_json(data / "derived" / a.env_variant / "train_vocab.json")
    spec = spec_for(a.env_variant)
    ckpt = data / a.arm_subdir / a.env_variant / f"seed_{a.seed}" / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing reference checkpoint {ckpt}")
    model, deploy = load_v3(ckpt, vocab, a.device)

    episodes = [
        e
        for e in (
            load_episode(p)
            for p in list_episodes(data / "raw" / a.env_variant / a.split)
        )
        if e.scene_geometry is not None
    ]
    n = len(episodes)
    print(
        f"[fc2_build_phi] {a.env_variant}/{a.split}: {n} episodes; ref={a.arm_subdir} "
        f"seed {a.seed}; device={a.device}",
        flush=True,
    )

    phi: dict[str, int] = {}
    t0 = time.time()
    for i, ep in enumerate(episodes, 1):
        attempts, _ = deployed_rollout_traced(
            model, ep, vocab, a.device, spec=spec, **(deploy or {})
        )
        phi[str(int(ep.provenance.problem_id))] = max(0, int(attempts) - 1)
        if i % 25 == 0 or i == n:
            el = time.time() - t0
            eta = el / i * (n - i)
            print(
                f"  [{i}/{n}] elapsed {el:.0f}s eta {eta:.0f}s "
                f"(phi so far: mean {sum(phi.values())/len(phi):.1f} "
                f"max {max(phi.values())})",
                flush=True,
            )

    label = a.label or a.arm_subdir.replace("checkpoints_", "").replace("/", "_")
    out = (
        Path(a.out)
        if a.out
        else (data / "derived" / a.env_variant / f"fc2_phi_{a.split}_{label}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(phi, f)
    vals = list(phi.values())
    vals.sort()
    print(
        f"[fc2_build_phi] wrote {out} ({len(phi)} problems); "
        f"phi mean {sum(vals)/len(vals):.2f} p50 {vals[len(vals)//2]} "
        f"p90 {vals[int(0.9*len(vals))]} max {vals[-1]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
