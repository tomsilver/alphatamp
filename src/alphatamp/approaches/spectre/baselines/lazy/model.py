"""The LAZY GAT policy (ported from ``baselines/drake-tamp/learning/policy.py``).

Two ``GATv2Conv`` message-passing layers over the object-relation graph (the real
``torch_geometric`` layer, per the locked decision) encode each object in the partial
state; each candidate next-operator is then scored by an attention-pool + MLP head that
reads its operator-schema embedding, its argument objects' embeddings, and the pooled
graph context. A softmax over a node's candidate actions is π(op|node).

Deviation from the paper (``PROVENANCE.md``): LAZY's third layer is a cross-attention
``GATv2Conv`` from graph nodes to action nodes; ours is attention-pool + MLP over
explicit argument embeddings. The object encoder — the GAT policy proper — is the
literal ``GATv2Conv``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter


class AttentionPolicy(nn.Module):
    """GAT policy: batched graphs -> per-candidate-action logits."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        op_vocab: int,
        max_arity: int,
        d: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d % heads == 0, "d must be divisible by heads"
        self.d = d
        self.max_arity = max_arity
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.gat1 = GATv2Conv(
            d,
            d // heads,
            heads=heads,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=dropout,
        )
        self.gat2 = GATv2Conv(
            d,
            d // heads,
            heads=heads,
            edge_dim=edge_dim,
            add_self_loops=False,
            dropout=dropout,
        )
        self.op_embed = nn.Embedding(op_vocab, d)
        self.action_mlp = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1)
        )

    def _encode_objects(self, batch: Batch) -> torch.Tensor:
        ea = batch.edge_attr if batch.edge_index.numel() else None
        h = self.node_encoder(batch.x)
        h = h + F.gelu(self.gat1(h, batch.edge_index, ea))
        h = h + F.gelu(self.gat2(h, batch.edge_index, ea))
        return h

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits [A], act_batch [A])`` — one logit per candidate action.

        ``act_batch[a]`` is the graph (prefix-tree node) that action ``a`` belongs to.
        """
        h = self._encode_objects(batch)  # [N_obj, d]
        num_graphs = int(batch.num_graphs)
        ctx = scatter(
            h, batch.batch, dim=0, dim_size=num_graphs, reduce="mean"
        )  # [G,d]

        act_op = batch.act_op  # [A]
        act_args = batch.act_args  # [A, max_arity] (global obj indices, pads offset)
        act_mask = batch.act_args_mask  # [A, max_arity]
        act_batch = batch.act_op_batch  # [A] (from follow_batch=['act_op'])

        op_emb = self.op_embed(act_op)  # [A, d]
        gathered = h[act_args]  # [A, max_arity, d]
        m = act_mask.unsqueeze(-1)  # [A, max_arity, 1]
        arg_emb = (gathered * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [A, d]
        ctx_a = ctx[act_batch]  # [A, d]

        a_in = torch.cat([op_emb, arg_emb, ctx_a], dim=-1)  # [A, 3d]
        logits = self.action_mlp(a_in).squeeze(-1)  # [A]
        return logits, act_batch

    def action_log_probs(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Grouped log-softmax: log π(op|node) per action, plus its ``act_batch``."""
        logits, act_batch = self.forward(batch)
        g = int(batch.num_graphs)
        gmax = scatter(logits, act_batch, dim=0, dim_size=g, reduce="max")
        z = logits - gmax[act_batch]
        denom = scatter(z.exp(), act_batch, dim=0, dim_size=g, reduce="sum")
        logsumexp = gmax + denom.clamp(min=1e-12).log()
        return logits - logsumexp[act_batch], act_batch


def bc_loss(model: AttentionPolicy, batch: Batch) -> torch.Tensor:
    """Behaviour-cloning cross-entropy: -log π(demonstrated op | node), averaged.

    ``batch.y_act[g]`` is the demonstrated action's local index within graph ``g``'s
    candidate list; we map it to the global action index via each graph's action offset.
    """
    logp, act_batch = model.action_log_probs(batch)
    g = int(batch.num_graphs)
    counts = scatter(
        torch.ones_like(act_batch), act_batch, dim=0, dim_size=g, reduce="sum"
    )
    offsets = torch.cumsum(counts, dim=0) - counts  # exclusive prefix sum, per graph
    target_global = offsets + batch.y_act.view(-1)  # [G]
    return -(logp[target_global]).mean()
