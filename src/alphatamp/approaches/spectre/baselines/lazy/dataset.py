"""Behaviour-cloning dataset for the LAZY policy.

Each *feasible* pool skeleton is a demonstration; walking its root→leaf path yields one
per-node classification example (the demonstrated next-operator among that node's
candidates — LAZY-exact cross-entropy). Multiple feasible leaves diverging at a shared
node contribute one single-label example each.

Episodes are canonicalized deterministically (``rng=None``); object-permutation
augmentation is deliberately omitted in v1 (``PROVENANCE.md``). ``keep_strata`` supports
held-out-stratum training; feasible leaves are **stride-sampled** (never truncated) to
``max_demos_per_episode`` so late-in-plan decisions are not systematically dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from alphatamp.approaches.spectre.baselines.lazy.graph import (
    EpisodeGraphCtx,
    FeatureSpec,
    LazyNodeData,
    build_episode_ctx,
    build_node_data,
    goal_binary_atoms,
)
from alphatamp.approaches.spectre.baselines.lazy.tree import (
    PrefixTree,
    build_prefix_tree,
)
from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.compare import stratum_of
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class EpisodeStruct:
    """Canonicalized episode + its prefix tree + static graph context (reused a lot)."""

    episode: EpisodeRecord
    tree: PrefixTree
    ctx: EpisodeGraphCtx


def load_structs(
    split_dir,
    vocab: Vocab,
    spec: FeatureSpec,
    frame_extent: tuple[float, float],
    shape_max,
    keep_strata: Optional[set[int]] = None,
) -> list[EpisodeStruct]:
    """Load, canonicalize and pre-structure every episode under ``split_dir``."""
    structs: list[EpisodeStruct] = []
    for path in list_episodes(split_dir):
        ep = canonicalize_episode(load_episode(path), rng=None)
        if keep_strata is not None:
            if stratum_of(int(ep.provenance.problem_id)) not in keep_strata:
                continue
        tree = build_prefix_tree(ep)
        ctx = build_episode_ctx(ep, vocab, spec, frame_extent, shape_max)
        structs.append(EpisodeStruct(episode=ep, tree=tree, ctx=ctx))
    return structs


def _stride_sample(items: list[int], k: int) -> list[int]:
    """At most ``k`` items, stride-sampled (never a truncating prefix)."""
    if k <= 0 or len(items) <= k:
        return items
    idx = [round(j * (len(items) - 1) / (k - 1)) for j in range(k)]
    return [items[t] for t in sorted(set(idx))]


def build_bc_examples(
    structs: Iterable[EpisodeStruct],
    vocab: Vocab,
    spec: FeatureSpec,
    max_demos_per_episode: int = 16,
) -> list:
    """One ``LazyNodeData`` per (feasible-leaf, decision-node) with its demonstrated
    op."""
    examples: list = []
    for st in structs:
        ep, tree = st.episode, st.tree
        feasible = [i for i, o in enumerate(ep.outcomes) if o.outcome == "success"]
        if not feasible:
            continue
        feasible = _stride_sample(feasible, max_demos_per_episode)
        gb = goal_binary_atoms(ep)
        # Build each distinct node's template once, then clone per demonstration.
        templates: dict[int, LazyNodeData] = {}
        for i in feasible:
            for nid, key in tree.leaf_decisions[i]:
                node = tree.nodes[nid]
                tmpl = templates.get(nid)
                if tmpl is None:
                    tmpl = build_node_data(ep, node, st.ctx, spec, vocab, gb)
                    templates[nid] = tmpl
                d = tmpl.clone()
                d.y_act = torch.tensor([node.action_index(key)], dtype=torch.long)
                examples.append(d)
    return examples
