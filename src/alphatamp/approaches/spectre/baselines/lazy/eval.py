"""Rollout-based evaluation for the LAZY policy (the selection + scoring surface).

Mirrors ``piginet.eval`` in role: turn a trained model into rollout false-positives.
Selection during training and the comparison cache both go through ``rollout_episode``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from alphatamp.approaches.spectre.baselines.lazy.dataset import EpisodeStruct
from alphatamp.approaches.spectre.baselines.lazy.feasibility import PhiStats
from alphatamp.approaches.spectre.baselines.lazy.graph import FeatureSpec
from alphatamp.approaches.spectre.baselines.lazy.model import AttentionPolicy
from alphatamp.approaches.spectre.baselines.lazy.rollout import (
    RolloutResult,
    compute_node_logpi,
    run_rollout,
)
from alphatamp.approaches.spectre.vocab import Vocab


def load_lazy_checkpoint(path: str | Path, device: str) -> tuple[AttentionPolicy, dict]:
    """Rebuild the policy + fitted ϕ prior from a ``ckpt.pt`` written by ``train``."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = AttentionPolicy(
        node_dim=ck["node_dim"],
        edge_dim=ck["edge_dim"],
        op_vocab=ck["op_vocab"],
        max_arity=ck["max_arity"],
        d=ck["d"],
        heads=ck["heads"],
        dropout=ck.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["phi_prior"]


def rollout_episode(
    model: AttentionPolicy,
    struct: EpisodeStruct,
    vocab: Vocab,
    spec: FeatureSpec,
    phi: PhiStats,
    device: str,
    cap: Optional[float] = None,
) -> RolloutResult:
    """One episode's online LAZY rollout (π computed once, ϕ fed back)."""
    node_logpi = compute_node_logpi(
        model, struct.episode, struct.tree, struct.ctx, spec, vocab, device
    )
    return run_rollout(struct.tree, node_logpi, phi, struct.episode, cap=cap)


def mean_rollout_fp(
    model: AttentionPolicy,
    structs: list[EpisodeStruct],
    vocab: Vocab,
    spec: FeatureSpec,
    phi: PhiStats,
    device: str,
) -> float:
    """Mean uncapped rollout-FP (attempts-1) over ``structs`` (skips no-feasible)."""
    model.eval()
    fps: list[float] = []
    for st in structs:
        if not any(o.outcome == "success" for o in st.episode.outcomes):
            continue  # no feasible skeleton -> undefined FP (matches static rollout_fp)
        r = rollout_episode(model, st, vocab, spec, phi, device)
        fps.append(float(r.attempts) - 1.0)
    return sum(fps) / max(1, len(fps))
