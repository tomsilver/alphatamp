"""Tests for the LAZY baseline (baselines/lazy).

The prefix-tree logic is tested with lightweight fakes (no torch). The graph/model
forward and the ϕ fit are exercised on a real episode when the dd2d_v4 collection and
``torch_geometric`` are available, and skipped otherwise (CI without the gitignored data
or the GAT stack).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alphatamp.approaches.spectre.baselines.lazy.tree import (
    STOP,
    build_prefix_tree,
    op_key,
)

REPO = Path(__file__).resolve().parents[3]
DD2D_TEST = REPO / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
DD2D_VOCAB = REPO / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"


def _fake_op(name: str, args: list[str]):
    return SimpleNamespace(
        name=name, parameters=[SimpleNamespace(name=a) for a in args]
    )


def _fake_episode(skeletons: list[list], outcomes: list[str]):
    pool = [SimpleNamespace(operator_seq=tuple(ops)) for ops in skeletons]
    outs = [SimpleNamespace(outcome=o) for o in outcomes]
    return SimpleNamespace(skeleton_pool=pool, outcomes=outs)


def test_prefix_tree_covers_non_error_and_shares_prefix() -> None:
    # Two skeletons share a first op; a third is an error (dropped); a fourth is a
    # prefix of the first (terminates at an internal node via STOP).
    a = _fake_op("pick", ["x0"])
    b = _fake_op("place", ["x0"])
    c = _fake_op("push", ["x1"])
    ep = _fake_episode(
        skeletons=[[a, b], [a, c], [b], [a]],
        outcomes=["success", "fail", "error", "success"],
    )
    tree = build_prefix_tree(ep)
    # error candidate (idx 2) excluded; the other three covered exactly once.
    assert set(tree.pool_indices) == {0, 1, 3}
    assert set(tree.leaf_decisions) == {0, 1, 3}
    # every leaf path ends in STOP and reconstructs its op-key sequence.
    for i in (0, 1, 3):
        decisions = tree.leaf_decisions[i]
        assert decisions[-1][1] == STOP
        keys = [k for _n, k in decisions if k != STOP]
        assert keys == [op_key(o) for o in ep.skeleton_pool[i].operator_seq]
    # skeletons 0 and 1 share the root->pick(x0) edge (one child there).
    root = tree.root
    assert op_key(a) in root.children
    # skeleton 3 (== [pick(x0)]) terminates at the pick node -> that node has_stop AND a
    # child (skeleton 0 continues with place).
    pick_node = tree.nodes[root.children[op_key(a)]]
    assert pick_node.has_stop and pick_node.children


def _load_first_big_episode():
    from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    for p in list_episodes(DD2D_TEST):
        ep = canonicalize_episode(load_episode(p), rng=None)
        if len(ep.skeleton_pool) >= 20:
            return ep
    return None


@pytest.mark.skipif(
    not (DD2D_TEST / "episodes").is_dir() or not DD2D_VOCAB.is_file(),
    reason="dd2d_v4 collection not present",
)
def test_graph_model_forward_and_phi() -> None:
    pytest.importorskip("torch_geometric")
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.utils import scatter

    from alphatamp.approaches.spectre.baselines.lazy.feasibility import (
        fit_phi,
        phi_value,
    )
    from alphatamp.approaches.spectre.baselines.lazy.graph import (
        build_episode_ctx,
        build_feature_spec,
        build_node_data,
        goal_binary_atoms,
    )
    from alphatamp.approaches.spectre.baselines.lazy.model import AttentionPolicy
    from alphatamp.approaches.spectre.baselines.lazy.tree import build_prefix_tree
    from alphatamp.approaches.spectre.vocab import Vocab

    ep = _load_first_big_episode()
    if ep is None:
        pytest.skip("no suitable dd2d_v4 episode")
    vocab = Vocab.from_json(DD2D_VOCAB)
    spec = build_feature_spec(vocab)
    tree = build_prefix_tree(ep)
    ctx = build_episode_ctx(
        ep,
        vocab,
        spec,
        (50.0, 40.0),
        __import__("numpy").array([25.0, 25.0, 150.0, 1.0]),
    )
    gb = goal_binary_atoms(ep)
    datas = [build_node_data(ep, n, ctx, spec, vocab, gb, y_act=0) for n in tree.nodes]
    batch = Batch.from_data_list(datas, follow_batch=["act_op"])
    model = AttentionPolicy(
        node_dim=spec.node_dim,
        edge_dim=spec.edge_dim,
        op_vocab=len(vocab.operators),
        max_arity=spec.max_arity,
    )
    model.eval()
    with torch.no_grad():
        logp, act_batch = model.action_log_probs(batch)
    g = int(batch.num_graphs)
    probsum = scatter(logp.exp(), act_batch, dim=0, dim_size=g, reduce="sum")
    assert torch.allclose(probsum, torch.ones_like(probsum), atol=1e-4)

    # ϕ fit is well-formed and bounded in (0, 1].
    phi = fit_phi([ep])
    for k in phi:
        assert 0.0 < phi_value(phi, k) <= 1.0
