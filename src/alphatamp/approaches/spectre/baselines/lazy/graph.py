"""Per-node object-relation graphs for the LAZY GAT policy.

For each prefix-tree node the policy needs the abstract state *at that partial plan*:
the object set is constant across a problem, but the relations change as operators
execute (``holding``/``handempty`` appear and vanish). We recompute the STRIPS state per
node (``trajectory.reconstruct_trajectory`` over the node's stored prefix), then emit a
``torch_geometric`` graph whose nodes are objects and whose edges are the binary atoms
of that state (plus binary goal atoms). Node features are the object's type, its unary-
atom flags at that state, a goal-membership bit, the broadcast nullary-atom flags (e.g.
``handempty``), and its normalized geometry (pose + shape from ``scene_geometry``).

Candidate next-operators become *action nodes*: each carries an operator-schema id and
its argument object-node indices; the model (``model.py``) scores them by attending over
the object embeddings. Feature layout is fixed from the frozen train vocab so a
checkpoint's dims are stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from alphatamp.approaches.spectre.baselines.lazy.tree import STOP, TreeNode
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory
from alphatamp.approaches.spectre.vocab import Vocab


class LazyNodeData(Data):
    """One prefix-tree node's graph + its candidate actions.

    Custom increment so a ``Batch`` offsets action argument indices into the batched
    object-node space (``edge_index`` is handled by the default rule).
    """

    def __inc__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key == "act_args":
            return self.num_nodes
        if key in ("act_op", "act_args_mask", "y_act"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key in ("act_op", "act_args", "act_args_mask", "y_act"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


@dataclass(frozen=True)
class FeatureSpec:
    """Fixed feature layout derived from the train vocab.

    ``geom_dim`` is 8 by default -- ``(x, y, sinθ, cosθ, w, h, area, concave)`` -- and 9
    for Restock3D v2, which appends a normalized object **height** so the GAT sees the
    F3 axis (a cube and a tall block share a 2D footprint and differ only in height).
    Each env trains its own GAT, so the input dim is env-specific; DD2D/SB2D stay at 8.
    """

    n_types: int
    nullary: tuple[str, ...]  # ordered nullary predicate names
    unary: tuple[str, ...]  # ordered unary predicate names
    n_preds: int  # full predicate vocab width (edge one-hot via pred_idx)
    max_arity: int
    geom_dim: int = 8

    @property
    def node_dim(self) -> int:
        """Total per-object node feature width."""
        return self.n_types + len(self.unary) + len(self.nullary) + 1 + self.geom_dim

    @property
    def edge_dim(self) -> int:
        """Edge feature width: predicate one-hot + goal-bit + (dx, dy, dθ)."""
        return self.n_preds + 1 + 3


def build_feature_spec(vocab: Vocab, geom_dim: int = 8) -> FeatureSpec:
    """Derive the fixed feature layout from a frozen vocab.

    ``geom_dim`` selects the geometry width (8 default; 9 = +height, for Restock3D v2).
    """
    nullary, unary = [], []
    for name, info in vocab.predicates.items():
        if name == "<OOV>":
            continue
        if info["arity"] == 0:
            nullary.append(name)
        elif info["arity"] == 1:
            unary.append(name)
    return FeatureSpec(
        n_types=len(vocab.types),
        nullary=tuple(sorted(nullary)),
        unary=tuple(sorted(unary)),
        n_preds=len(vocab.predicates),
        max_arity=int(vocab.max_operator_arity),
        geom_dim=geom_dim,
    )


@dataclass
class EpisodeGraphCtx:
    """Per-episode static tensors shared across all its tree-node graphs."""

    obj_names: list[str]
    obj_index: dict[str, int]
    type_onehot: np.ndarray  # [N, n_types]
    goal_flag: np.ndarray  # [N]
    geom: np.ndarray  # [N, geom_dim] normed (x,y,sinθ,cosθ,w,h,area,concave[,height])
    pose: np.ndarray  # [N, 3] raw (x, y, θ) for edge transforms
    frame: tuple[float, float]


def build_episode_ctx(
    episode: EpisodeRecord,
    vocab: Vocab,
    spec: FeatureSpec,
    frame_extent: tuple[float, float],
    shape_max: np.ndarray,
) -> EpisodeGraphCtx:
    """Precompute the state-independent per-object features of an episode."""
    objs = sorted(episode.initial_abstract_state.objects, key=lambda o: o.name)
    obj_names = [o.name for o in objs]
    obj_index = {n: i for i, n in enumerate(obj_names)}
    n = len(objs)

    type_onehot = np.zeros((n, spec.n_types), dtype=np.float32)
    for i, o in enumerate(objs):
        type_onehot[i, vocab.type_idx(o.type.name)] = 1.0

    goal_names = {e.name for a in episode.goal_atoms for e in a.objects}
    goal_flag = np.array(
        [1.0 if nm in goal_names else 0.0 for nm in obj_names], dtype=np.float32
    )

    geom = np.zeros((n, spec.geom_dim), dtype=np.float32)
    pose = np.zeros((n, 3), dtype=np.float32)
    fw, fh = float(frame_extent[0]), float(frame_extent[1])
    sm = np.asarray(shape_max, dtype=np.float32)
    if episode.scene_geometry is not None:
        by_name = {o.name: o for o in episode.scene_geometry.objects}
        for i, nm in enumerate(obj_names):
            g = by_name.get(nm)
            if g is None:
                continue
            x, y, th = float(g.pose[0]), float(g.pose[1]), float(g.pose[2])
            pose[i] = (x, y, th)
            ring = np.asarray(g.boundary, dtype=np.float32)
            w = float(ring[:, 0].max() - ring[:, 0].min()) if len(ring) else 0.0
            h = float(ring[:, 1].max() - ring[:, 1].min()) if len(ring) else 0.0
            geom[i, :8] = (
                x / fw,
                y / fh,
                math.sin(th),
                math.cos(th),
                w / sm[0],
                h / sm[1],
                float(g.area) / sm[2],
                float(g.concave) / sm[3],
            )
            # Restock3D v2: 9th feature = normalized object height (the F3 axis: cube vs
            # tall block share a footprint, differ only here). ``shape_max[4]`` is the
            # height normalizer; a footprint-only object (no height) stays 0.
            if spec.geom_dim > 8 and len(sm) > 4:
                height = float(getattr(g, "height", 0.0) or 0.0)
                geom[i, 8] = height / sm[4]
    return EpisodeGraphCtx(
        obj_names=obj_names,
        obj_index=obj_index,
        type_onehot=type_onehot,
        goal_flag=goal_flag,
        geom=geom,
        pose=pose,
        frame=(fw, fh),
    )


def _state_atom_index(state) -> dict[str, list[tuple[str, ...]]]:
    """Map predicate name -> list of object-name tuples present in the state."""
    out: dict[str, list[tuple[str, ...]]] = {}
    for atom in state.atoms:
        out.setdefault(atom.predicate.name, []).append(
            tuple(o.name for o in atom.objects)
        )
    return out


def build_node_data(
    episode: EpisodeRecord,
    node: TreeNode,
    ctx: EpisodeGraphCtx,
    spec: FeatureSpec,
    vocab: Vocab,
    goal_binary: list[tuple[str, str, str]],
    y_act: Optional[int] = None,
) -> LazyNodeData:
    """Build the graph for one prefix-tree node (state = initial ∘ node.prefix_ops).

    ``goal_binary`` is the precomputed list of ``(pred_name, src_name, dst_name)`` binary
    goal atoms (episode-level; passed in to avoid recomputing per node).
    """
    state = reconstruct_trajectory(
        episode.initial_abstract_state, node.prefix_ops, verify_preconditions=False
    )[-1]
    atom_idx = _state_atom_index(state)
    n = len(ctx.obj_names)

    # -- node features ------------------------------------------------------
    unary_feat = np.zeros((n, len(spec.unary)), dtype=np.float32)
    for j, pname in enumerate(spec.unary):
        for (oname,) in atom_idx.get(pname, []):
            oi = ctx.obj_index.get(oname)
            if oi is not None:
                unary_feat[oi, j] = 1.0
    nullary_feat = np.zeros((n, len(spec.nullary)), dtype=np.float32)
    for j, pname in enumerate(spec.nullary):
        if atom_idx.get(pname):
            nullary_feat[:, j] = 1.0
    x = np.concatenate(
        [ctx.type_onehot, unary_feat, nullary_feat, ctx.goal_flag[:, None], ctx.geom],
        axis=1,
    ).astype(np.float32)

    # -- edges: binary state atoms (both directions) + binary goal atoms ----
    src, dst, eattr = [], [], []

    def _add_edge(a: int, b: int, pname: str, goal_bit: float) -> None:
        vec = np.zeros(spec.edge_dim, dtype=np.float32)
        vec[vocab.pred_idx(pname)] = 1.0
        vec[spec.n_preds] = goal_bit
        dp = ctx.pose[b] - ctx.pose[a]
        vec[spec.n_preds + 1] = dp[0] / ctx.frame[0]
        vec[spec.n_preds + 2] = dp[1] / ctx.frame[1]
        vec[spec.n_preds + 3] = float(dp[2])
        src.append(a)
        dst.append(b)
        eattr.append(vec)

    for atom in state.atoms:
        if atom.predicate.arity != 2:
            continue
        names = [o.name for o in atom.objects]
        a, b = ctx.obj_index.get(names[0]), ctx.obj_index.get(names[1])
        if a is None or b is None:
            continue
        _add_edge(a, b, atom.predicate.name, 0.0)
        _add_edge(b, a, atom.predicate.name, 0.0)
    for pname, sname, dname in goal_binary:
        a, b = ctx.obj_index.get(sname), ctx.obj_index.get(dname)
        if a is None or b is None:
            continue
        _add_edge(a, b, pname, 1.0)
        _add_edge(b, a, pname, 1.0)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(np.stack(eattr), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, spec.edge_dim), dtype=torch.float32)

    # -- candidate actions --------------------------------------------------
    n_act = len(node.actions)
    act_op = np.zeros(n_act, dtype=np.int64)
    act_args = np.zeros((n_act, spec.max_arity), dtype=np.int64)
    act_mask = np.zeros((n_act, spec.max_arity), dtype=np.float32)
    for ai, key in enumerate(node.actions):
        if key == STOP:
            act_op[ai] = 0  # OOV/STOP slot
            continue
        op_name, arg_names = key
        act_op[ai] = vocab.op_idx(op_name)
        for si, an in enumerate(arg_names[: spec.max_arity]):
            oi = ctx.obj_index.get(an)
            if oi is not None:
                act_args[ai, si] = oi
                act_mask[ai, si] = 1.0

    # Pass action tensors as Data kwargs (PyG stores arbitrary kwargs as attributes).
    kwargs = {
        "x": torch.from_numpy(x),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "act_op": torch.from_numpy(act_op),
        "act_args": torch.from_numpy(act_args),
        "act_args_mask": torch.from_numpy(act_mask),
        "num_actions": n_act,
    }
    if y_act is not None:
        kwargs["y_act"] = torch.tensor([y_act], dtype=torch.long)
    return LazyNodeData(**kwargs)


def goal_binary_atoms(episode: EpisodeRecord) -> list[tuple[str, str, str]]:
    """Binary goal atoms as ``(pred_name, src_name, dst_name)`` triples."""
    out = []
    for a in episode.goal_atoms:
        if a.predicate.arity == 2:
            names = [o.name for o in a.objects]
            out.append((a.predicate.name, names[0], names[1]))
    return out
