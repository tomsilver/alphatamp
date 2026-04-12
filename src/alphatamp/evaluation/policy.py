"""Skeleton selection policies for offline evaluation.

Provides a ``SelectionPolicy`` protocol and concrete implementations:

- **IndexPolicy**: Learned policy using the index rule
  ``argmin_{j in C_t} E[T_j|Y_j=1] / P(Y_j=1)``.
- **RandomPolicy**: Uniform random selection among candidates.
- **ShortestFirstPolicy**: Oracle baseline that tries skeletons in ascending
  ground-truth refinement time order.
- **OracleBaseline**: Ground-truth oracle (lower bound on TTFS).
- **SuccessFirstFixedOrder**: Fixed ordering by empirical success rate.
- **ShortestFirstFixedOrder**: Fixed ordering by skeleton plan length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from alphatamp.data.skeleton_dataset import SkeletonDataset, SkeletonItem
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.prediction_heads import JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder
from alphatamp.training.trainer import _pad_op_sequences

__all__ = [
    "IndexPolicy",
    "OracleBaseline",
    "RandomPolicy",
    "SelectionPolicy",
    "ShortestFirstFixedOrder",
    "ShortestFirstPolicy",
    "SuccessFirstFixedOrder",
]


@runtime_checkable
class SelectionPolicy(Protocol):
    """Protocol for skeleton selection policies.

    Policies are stateful: ``reset()`` initialises per-instance state,
    ``select()`` makes sequential decisions within one instance.
    """

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """Initialise state for a new problem instance."""
        ...

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Return index *j* of the next skeleton to attempt."""
        ...


# ---------------------------------------------------------------------------
# IndexPolicy — learned model policy
# ---------------------------------------------------------------------------


class IndexPolicy:
    """Index-rule policy: argmin_{j in C_t} E[T_j|Y_j=1] / P(Y_j=1).

    Wraps the trained model components and pre-computes skeleton embeddings
    once at construction time.  The ``select`` method runs a single no-grad
    forward pass through TokenBuilder → BeliefEncoder → heads to score all
    candidates and returns the one with the lowest index score.

    Parameters
    ----------
    skeleton_encoder:
        Trained :class:`SkeletonEncoder` (already on *device*).
    token_builder:
        Trained :class:`TokenBuilder`.
    belief_encoder:
        Trained :class:`BeliefEncoder`.
    y_head:
        Trained :class:`YHead`.
    t_head:
        Trained :class:`THead`.
    joint_y_head:
        Trained :class:`JointYHead`.
    dataset:
        :class:`SkeletonDataset` whose vocabulary matches the models.
        Used to prepare skeleton-level inputs (op sequences, lengths).
    device:
        Torch device.  Models must already be on this device.
    """

    def __init__(
        self,
        skeleton_encoder: SkeletonEncoder,
        token_builder: TokenBuilder,
        belief_encoder: BeliefEncoder,
        y_head: YHead,
        t_head: THead,
        joint_y_head: JointYHead,
        dataset: SkeletonDataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._token_builder = token_builder
        self._belief_encoder = belief_encoder
        self._y_head = y_head
        self._t_head = t_head
        self._joint_y_head = joint_y_head
        self._device = device

        # Prepare skeleton-level inputs (shared across all instances)
        op_type_ids, obj_ids, skel_lengths_enc = _pad_op_sequences(dataset, device)
        self._skel_lengths_tb = dataset.skeleton_lengths.float().to(device)
        self._M = dataset.M

        # Pre-compute skeleton embeddings (once, no grad)
        skeleton_encoder.eval()
        with torch.no_grad():
            self._skel_embeds = skeleton_encoder(
                op_type_ids, obj_ids, skel_lengths_enc,
            )  # (M, d_skel)

        # Ensure all heads are in eval mode
        for mod in [token_builder, belief_encoder, y_head, t_head, joint_y_head]:
            mod.eval()

        # Per-instance state set in reset()
        self._applicability: Tensor | None = None

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """Store applicability for the current instance."""
        self._applicability = item.applicability.to(self._device)

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Apply the index rule and return the best candidate index."""
        assert self._applicability is not None, "Call reset() before select()"

        with torch.no_grad():
            se = self._skel_embeds.unsqueeze(0)  # (1, M, d_skel)
            tokens = self._token_builder(
                se,
                self._applicability.unsqueeze(0),
                revealed_mask.unsqueeze(0).to(self._device),
                revealed_y.unsqueeze(0).to(self._device),
                revealed_f.unsqueeze(0).to(self._device),
                revealed_t.unsqueeze(0).to(self._device),
                self._skel_lengths_tb.unsqueeze(0),
            )
            pad_mask = torch.zeros(
                1, self._M, dtype=torch.bool, device=self._device,
            )
            ctx, _ = self._belief_encoder(tokens, pad_mask)

            marginal_logits = self._y_head(ctx, pad_mask)  # (1, M)
            y_logits = self._joint_y_head(ctx, marginal_logits, pad_mask)  # (1, M)
            t_dist = self._t_head(ctx, pad_mask)  # LogNormal(1, M)

            p_success = torch.sigmoid(y_logits).clamp(min=1e-6).squeeze(0)  # (M,)
            e_t = t_dist.mean.squeeze(0)  # (M,)

            score = e_t / p_success  # (M,)
            score[~candidate_mask.to(self._device)] = float("inf")
            return int(score.argmin().item())


# ---------------------------------------------------------------------------
# RandomPolicy — uniform random baseline
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Uniform random selection among candidates.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = torch.Generator().manual_seed(seed)

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """No per-instance state needed."""

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Pick a candidate uniformly at random."""
        cand_indices = torch.where(candidate_mask)[0]
        rand_pos = torch.randint(len(cand_indices), (1,), generator=self._rng)
        return int(cand_indices[rand_pos].item())


# ---------------------------------------------------------------------------
# ShortestFirstPolicy — oracle baseline on refinement time
# ---------------------------------------------------------------------------


class ShortestFirstPolicy:
    """Oracle baseline: try skeletons in ascending ground-truth T order.

    Knows true refinement times (oracle), ignores success probability.
    Represents the strategy "try cheap things first."
    """

    def __init__(self) -> None:
        self._sorted_order: list[int] = []
        self._ptr: int = 0

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """Sort applicable skeletons by ascending refinement time."""
        applicable_mask = item.applicability > 0.5
        applicable_indices = torch.where(applicable_mask)[0]
        if len(applicable_indices) == 0:
            self._sorted_order = []
            self._ptr = 0
            return

        t_app = item.refinement_time[applicable_indices]
        order = torch.argsort(t_app, stable=True)
        self._sorted_order = applicable_indices[order].tolist()
        self._ptr = 0

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Return the next unrevealed skeleton from the time-sorted order."""
        while self._ptr < len(self._sorted_order):
            idx = self._sorted_order[self._ptr]
            self._ptr += 1
            if candidate_mask[idx]:
                return idx
        raise RuntimeError("No candidate available in ShortestFirstPolicy")


# ---------------------------------------------------------------------------
# OracleBaseline — ground-truth lower bound on TTFS
# ---------------------------------------------------------------------------


class OracleBaseline:
    """Oracle lower bound: sort candidates by (F descending, T ascending).

    WARNING: This policy uses full ground-truth labels for the current
    instance and represents an unachievable lower bound on TTFS.  It must
    NEVER be used as a training signal — only inside ``OfflineEvaluator``
    for benchmarking purposes.

    Sorting logic:

    - Primary key: steps_completed_fraction (F) descending — prefer
      skeletons that complete more steps (F=1.0 means Y=1).
    - Secondary key: refinement_time (T) ascending — among equal F,
      prefer cheaper skeletons.
    - This guarantees: all Y=1 skeletons are tried first (since Y=1 ⟹
      F=1), ordered by ascending T.  Among failures, those with higher
      partial progress come next.
    """

    def __init__(self) -> None:
        self._sorted_order: list[int] = []
        self._ptr: int = 0

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """Sort applicable skeletons by (F descending, T ascending)."""
        applicable_mask = item.applicability > 0.5
        applicable_indices = torch.where(applicable_mask)[0]
        if len(applicable_indices) == 0:
            self._sorted_order = []
            self._ptr = 0
            return

        f_app = item.steps_completed_fraction[applicable_indices]
        t_app = item.refinement_time[applicable_indices]

        # Two-pass stable sort for lexicographic (F desc, T asc):
        # 1. Sort by secondary key (T ascending)
        order_by_t = torch.argsort(t_app, stable=True)
        reindexed = applicable_indices[order_by_t]
        f_reordered = f_app[order_by_t]
        # 2. Sort by primary key (-F ascending) with stable=True
        order_by_neg_f = torch.argsort(-f_reordered, stable=True)
        self._sorted_order = reindexed[order_by_neg_f].tolist()
        self._ptr = 0

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Return the next skeleton from the oracle-sorted order."""
        while self._ptr < len(self._sorted_order):
            idx = self._sorted_order[self._ptr]
            self._ptr += 1
            if candidate_mask[idx]:
                return idx
        raise RuntimeError("No candidate available in OracleBaseline")


# ---------------------------------------------------------------------------
# SuccessFirstFixedOrder — fixed global ordering by empirical success rate
# ---------------------------------------------------------------------------


class SuccessFirstFixedOrder:
    """Fixed global ordering by empirical success rate (fit on training set).

    For each skeleton *j*, ``success_rate[j] = mean(Y_w(j) for w where
    A_w(j)=1)``.  Skeletons that are never applicable get rate 0.  The
    global ordering ranks by success_rate descending, with ties broken by
    vocabulary index ascending (lower index first).

    This policy ignores all history at evaluation time — it always returns
    the highest-ranked skeleton in the current candidate set C_t.

    Parameters
    ----------
    ordering:
        Pre-computed global ordering as a list of skeleton indices
        (highest success rate first).  Either call :meth:`fit` or
        :meth:`load_ordering` to populate, or pass directly.
    """

    def __init__(self, ordering: list[int] | None = None) -> None:
        self._ordering: list[int] = ordering or []
        self._success_rates: Tensor = torch.tensor([])

    def fit(self, dataset: SkeletonDataset) -> None:
        """Compute per-skeleton success rates from a training dataset."""
        M = dataset.M
        success_count = torch.zeros(M)
        applicable_count = torch.zeros(M)

        for i in range(len(dataset)):
            item = dataset[i]
            app_mask = item.applicability > 0.5
            success_count += item.success * app_mask.float()
            applicable_count += app_mask.float()

        self._success_rates = torch.where(
            applicable_count > 0,
            success_count / applicable_count,
            torch.zeros(M),
        )
        # Stable argsort on -rate: ties broken by vocab index
        self._ordering = torch.argsort(
            -self._success_rates, stable=True,
        ).tolist()

    @property
    def success_rates(self) -> Tensor:
        """Per-skeleton success rates (populated after :meth:`fit`)."""
        return self._success_rates

    def save_ordering(self, path: str | Path) -> None:
        """Save the fitted ordering and success rates to JSON."""
        data = {
            "ordering": self._ordering,
            "success_rates": self._success_rates.tolist(),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_ordering(cls, path: str | Path) -> SuccessFirstFixedOrder:
        """Load a previously saved ordering from JSON."""
        data = json.loads(Path(path).read_text())
        instance = cls(ordering=data["ordering"])
        if "success_rates" in data:
            instance._success_rates = torch.tensor(data["success_rates"])
        return instance

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """No per-instance state needed (fixed global ordering)."""

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Return the highest-ranked candidate in the fixed ordering."""
        for idx in self._ordering:
            if candidate_mask[idx]:
                return idx
        raise RuntimeError("No candidate available in SuccessFirstFixedOrder")


# ---------------------------------------------------------------------------
# ShortestFirstFixedOrder — fixed global ordering by plan length
# ---------------------------------------------------------------------------


class ShortestFirstFixedOrder:
    """Fixed global ordering by skeleton plan length (shortest first).

    Sorts all M skeletons by ascending plan length *L* (from
    ``dataset.skeleton_lengths``), with ties broken by vocabulary index
    ascending.  This is a non-oracle, training-free baseline that assumes
    shorter plans are more likely to succeed or cheaper to evaluate.

    Unlike :class:`ShortestFirstPolicy` (which sorts by per-instance
    ground-truth refinement time *T*), this policy uses only the static
    skeleton metadata and produces the same ordering for every instance.

    Parameters
    ----------
    skeleton_lengths:
        Integer tensor of shape ``(M,)`` giving the number of operators per
        skeleton.  Typically ``dataset.skeleton_lengths``.
    """

    def __init__(self, skeleton_lengths: Tensor) -> None:
        self._ordering: list[int] = torch.argsort(
            skeleton_lengths.int(), stable=True,
        ).tolist()

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        """No per-instance state needed (fixed global ordering)."""

    def select(
        self,
        candidate_mask: Tensor,
        revealed_mask: Tensor,
        revealed_y: Tensor,
        revealed_f: Tensor,
        revealed_t: Tensor,
    ) -> int:
        """Return the shortest available skeleton."""
        for idx in self._ordering:
            if candidate_mask[idx]:
                return idx
        raise RuntimeError("No candidate available in ShortestFirstFixedOrder")
