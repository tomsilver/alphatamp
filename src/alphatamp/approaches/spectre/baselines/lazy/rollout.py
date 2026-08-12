"""Online LAZY rollout: π̄ = π·ϕ/Σ with failure feedback over the fixed pool.

The policy π is evaluated once per episode (one batched GAT forward over all distinct
prefix-tree nodes). The rollout then greedily attempts the pool skeleton with the highest
path probability ``Π π̄(op|node)`` under the current ϕ, and on each failure updates ϕ
(``feasibility.observe_failure``), which is pure CPU arithmetic — no GAT re-run. This is
LAZY's LevinTS-style ``1/path_prob`` priority over a fixed candidate pool.

Both the uncapped and per-candidate-capped variants reuse the same fixed π (only the ϕ
feedback and the stopping rule differ), matching ``cache_spectre3``: a slow-feasible
candidate over the cap is abandoned into the failure context, so the capped order can
diverge from the uncapped one and must be re-run rather than derived by capping times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from alphatamp.approaches.spectre.baselines.lazy.feasibility import (
    PhiStats,
    copy_stats,
    observe_failure,
    phi_value,
)
from alphatamp.approaches.spectre.baselines.lazy.graph import (
    EpisodeGraphCtx,
    FeatureSpec,
    build_node_data,
    goal_binary_atoms,
)
from alphatamp.approaches.spectre.baselines.lazy.model import AttentionPolicy
from alphatamp.approaches.spectre.baselines.lazy.tree import PrefixTree
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class RolloutResult:
    """One rollout: attempts to first success, realized order, refine seconds."""

    attempts: int
    order: list[int]
    refine_s: float
    found: bool


def _logsumexp(a: np.ndarray) -> float:
    m = float(a.max())
    return m + math.log(float(np.exp(a - m).sum()))


@torch.no_grad()
def compute_node_logpi(
    model: AttentionPolicy,
    episode: EpisodeRecord,
    tree: PrefixTree,
    ctx: EpisodeGraphCtx,
    spec: FeatureSpec,
    vocab: Vocab,
    device: str,
) -> dict[int, np.ndarray]:
    """One batched forward → ``{node_id: log π over that node's ordered actions}``."""
    gb = goal_binary_atoms(episode)
    datas = [
        build_node_data(episode, node, ctx, spec, vocab, gb) for node in tree.nodes
    ]
    batch = Batch.from_data_list(datas, follow_batch=["act_op"]).to(device)
    logp, _ = model.action_log_probs(batch)
    logp_np = logp.detach().cpu().numpy()
    out: dict[int, np.ndarray] = {}
    off = 0
    for node in tree.nodes:  # graphs are batched in tree.nodes order
        k = len(node.actions)
        out[node.node_id] = logp_np[off : off + k]
        off += k
    return out


def _path_logp(
    tree: PrefixTree, node_logpi: dict[int, np.ndarray], phi: PhiStats, i: int
) -> float:
    """Log Π π̄(op|node) along leaf ``i``'s root→leaf path under the current ϕ."""
    total = 0.0
    for nid, key in tree.leaf_decisions[i]:
        node = tree.nodes[nid]
        base = node_logpi[nid]
        logphi = np.array(
            [math.log(phi_value(phi, k)) for k in node.actions], dtype=np.float64
        )
        adj = base.astype(np.float64) + logphi
        total += float(adj[node.action_index(key)] - _logsumexp(adj))
    return total


def run_rollout(
    tree: PrefixTree,
    node_logpi: dict[int, np.ndarray],
    phi_prior: PhiStats,
    episode: EpisodeRecord,
    cap: Optional[float] = None,
) -> RolloutResult:
    """Greedy π̄ rollout over the pool; returns attempts / realized order / refine_s.

    ``cap`` (seconds) abandons any candidate whose stored refinement time exceeds it —
    including a feasible one — into the failure context (LAZY re-plan). ``refine_s``
    sums ``min(t, cap)`` (or full ``t`` when uncapped) along the realized order up to
    and including the stop.
    """
    phi = copy_stats(phi_prior)
    remaining = list(tree.pool_indices)
    order: list[int] = []
    refine_s = 0.0
    attempts = 0
    outcomes = {i: episode.outcomes[i].outcome for i in remaining}
    times = {
        i: float(episode.outcomes[i].refinement_wall_clock_s or 0.0) for i in remaining
    }
    while remaining:
        scores = {i: _path_logp(tree, node_logpi, phi, i) for i in remaining}
        # max path prob; tie-break lowest pool index (== astar default order)
        i = max(remaining, key=lambda k: (scores[k], -k))
        attempts += 1
        order.append(i)
        t = times[i]
        refine_s += min(t, cap) if cap is not None else t
        succeeded = outcomes[i] == "success" and (cap is None or t <= cap)
        if succeeded:
            return RolloutResult(attempts, order, round(refine_s, 6), True)
        observe_failure(phi, episode, i)  # fail OR slow-abandoned success
        remaining.remove(i)
    return RolloutResult(attempts, order, round(refine_s, 6), False)
