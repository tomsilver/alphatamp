"""Prefix tree over a fixed candidate-skeleton pool (LAZY, faithful-max).

LAZY builds skeletons incrementally and its policy scores *action extensions* at each
partial-plan state. SPECTRE hands us a fixed pool of complete skeletons instead, so we
recover the same structure by building a trie over the pool's canonicalized operator
sequences: a node is a shared operator prefix, an edge is a next-operator, a leaf is a
pool skeleton. The per-action policy then scores the children at each node and
π(skeleton) is the product of per-action probabilities along its root→leaf path
(``decisions/07`` 2026-08-09 / ``PROVENANCE.md``).

Keys are the renaming-invariant per-operator canonical keys — the analog of LAZY's
``utils.anonymise`` computation-graph key. The episode passed in **must already be
canonicalized** (``canonicalize_episode(ep, rng=None)`` or ``eda.load_split_episodes``),
so ``op.parameters`` are typed-local ids and equal keys ≡ equivalent operators.

Error-outcome skeletons are excluded from the tree, matching every other adaptive
baseline (``eda._lazy_rollout`` / ``adaptive_historical_baseline`` both drop them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from relational_structs import GroundOperator

from alphatamp.approaches.spectre.schema import EpisodeRecord

# One operator's renaming-invariant key: (op_name, (typed-local arg names, ...)).
OpKey = tuple[str, tuple[str, ...]]
# Terminal action sentinel; a real op name never collides with this.
STOP: OpKey = ("<STOP>", ())


def op_key(op: GroundOperator) -> OpKey:
    """Per-operator canonical key of a *canonicalized* ground operator."""
    return (op.name, tuple(a.name for a in op.parameters))


@dataclass
class TreeNode:
    """One prefix-tree node = a shared operator prefix."""

    node_id: int
    prefix_ops: tuple[GroundOperator, ...]  # the operators executed to reach this node
    children: dict[OpKey, int] = field(default_factory=dict)  # child op key -> node_id
    has_stop: bool = False  # some pool skeleton terminates exactly here
    # Ordered candidate actions at this node (children keys sorted, then STOP if any).
    # Populated by :meth:`finalize`.
    actions: list[OpKey] = field(default_factory=list)

    def finalize(self) -> None:
        """Freeze the ordered candidate-action list (deterministic)."""
        acts: list[OpKey] = sorted(self.children.keys())
        if self.has_stop:
            acts.append(STOP)
        self.actions = acts

    def action_index(self, key: OpKey) -> int:
        """Local index of ``key`` in this node's ordered candidate list."""
        return self.actions.index(key)


@dataclass
class PrefixTree:
    """A pool's prefix tree plus the leaf↔pool-index bookkeeping.

    - ``nodes[node_id]`` is a :class:`TreeNode`.
    - ``pool_indices`` are the (non-error) candidate indices this tree covers.
    - ``leaf_decisions[i]`` is pool skeleton ``i``'s root→leaf decision list:
      ``[(node_id, action_key), ...]`` ending in ``(terminal_node_id, STOP)``.
    """

    nodes: list[TreeNode]
    pool_indices: list[int]
    leaf_decisions: dict[int, list[tuple[int, OpKey]]]

    @property
    def root(self) -> TreeNode:
        """The empty-prefix root node."""
        return self.nodes[0]

    def distinct_node_ids(self) -> list[int]:
        """All node ids (each is a distinct partial plan)."""
        return [n.node_id for n in self.nodes]


def build_prefix_tree(
    episode: EpisodeRecord,
    outcomes: Optional[Sequence[str]] = None,
) -> PrefixTree:
    """Build the prefix tree over a canonicalized episode's non-error pool.

    ``outcomes`` overrides ``episode.outcomes[i].outcome`` (unused here, but kept so a
    caller with a filtered view can pass it). Skeletons whose outcome is ``"error"`` are
    dropped before insertion.
    """
    outs = (
        list(outcomes)
        if outcomes is not None
        else [o.outcome for o in episode.outcomes]
    )
    root = TreeNode(node_id=0, prefix_ops=())
    nodes: list[TreeNode] = [root]
    leaf_decisions: dict[int, list[tuple[int, OpKey]]] = {}
    pool_indices: list[int] = []

    for i, skel in enumerate(episode.skeleton_pool):
        if outs[i] == "error":
            continue
        pool_indices.append(i)
        decisions: list[tuple[int, OpKey]] = []
        cur = root
        prefix: list[GroundOperator] = []
        for op in skel.operator_seq:
            k = op_key(op)
            decisions.append((cur.node_id, k))
            child_id = cur.children.get(k)
            if child_id is None:
                child_id = len(nodes)
                child = TreeNode(node_id=child_id, prefix_ops=tuple(prefix) + (op,))
                nodes.append(child)
                cur.children[k] = child_id
            cur = nodes[child_id]
            prefix.append(op)
        # Terminal decision.
        cur.has_stop = True
        decisions.append((cur.node_id, STOP))
        leaf_decisions[i] = decisions

    for n in nodes:
        n.finalize()
    return PrefixTree(
        nodes=nodes, pool_indices=pool_indices, leaf_decisions=leaf_decisions
    )
