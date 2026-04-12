"""Offline evaluator for skeleton selection policies.

Simulates sequential refinement rollouts on held-out data and reports
time-to-first-success, success@k, and per-instance cost ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from alphatamp.data.skeleton_dataset import SkeletonDataset, SkeletonItem
from alphatamp.evaluation.policy import SelectionPolicy

__all__ = ["EvalMetrics", "OfflineEvaluator", "RolloutResult"]


@dataclass(frozen=True)
class RolloutResult:
    """Result of one rollout on one instance."""

    seed_id: int
    ttfs: float | None
    """Cumulative refinement time to first success, or ``None`` if no success."""
    n_attempts: int
    """Number of skeletons attempted before success or exhaustion."""
    success: bool
    """Whether Y=1 was found."""
    attempt_indices: list[int] = field(default_factory=list)
    """Skeleton indices tried, in order."""


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate evaluation metrics across a dataset."""

    mean_ttfs: float
    """Mean TTFS over instances where the policy succeeded."""
    median_ttfs: float
    """Median TTFS over instances where the policy succeeded."""
    success_at_k: dict[int, float]
    """{k: fraction of instances with success in <= k attempts}."""
    n_instances: int
    n_succeeded: int
    mean_cost_ratio: float | None
    """Geometric mean of TTFS_policy / TTFS_baseline over paired instances.
    ``None`` if no baseline provided or no paired successes."""
    n_paired: int
    """Instances where both policy and baseline succeeded."""
    exclusive_success_policy: int
    """Instances where only the policy succeeded."""
    exclusive_success_baseline: int
    """Instances where only the baseline succeeded."""
    per_instance: list[RolloutResult]


class OfflineEvaluator:
    """Simulate sequential refinement rollouts on held-out instances.

    Rollout protocol:

    1. **Start**: inapplicable skeletons are "revealed" upfront (known failures).
    2. **Loop**: ``policy.select()`` → look up ground-truth (Y, F, T) → append
       to history.
    3. **Stop**: on Y=1 (success) or exhaustion of all applicable candidates.

    Metrics: mean TTFS (cumulative refinement time), success@k, and optional
    cost ratios vs a baseline policy.

    Parameters
    ----------
    dataset:
        :class:`SkeletonDataset` to evaluate on (held-out test set).
    success_at_k_values:
        List of *k* values for the success@k metric.  Default ``[1, 2, 3, 5]``.
    """

    def __init__(
        self,
        dataset: SkeletonDataset,
        success_at_k_values: list[int] | None = None,
    ) -> None:
        self._dataset = dataset
        self._k_values = success_at_k_values or [1, 2, 3, 5]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        policy: SelectionPolicy,
        *,
        baseline: SelectionPolicy | None = None,
    ) -> EvalMetrics:
        """Run rollouts for *policy* on all instances.

        Parameters
        ----------
        policy:
            The policy to evaluate.
        baseline:
            Optional baseline policy for cost-ratio computation.  When
            provided, per-instance ``TTFS_policy / TTFS_baseline`` ratios
            are aggregated via the geometric mean.
        """
        results = self._run_all(policy)
        baseline_results = self._run_all(baseline) if baseline is not None else None
        return self._compute_metrics(results, baseline_results)

    def rollout_single(
        self,
        policy: SelectionPolicy,
        instance_idx: int,
    ) -> RolloutResult:
        """Run a single rollout.  Useful for debugging."""
        item = self._dataset[instance_idx]
        return self._rollout(policy, item)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_all(self, policy: SelectionPolicy) -> list[RolloutResult]:
        results: list[RolloutResult] = []
        for i in range(len(self._dataset)):
            item = self._dataset[i]
            results.append(self._rollout(policy, item))
        return results

    def _rollout(self, policy: SelectionPolicy, item: SkeletonItem) -> RolloutResult:
        M = self._dataset.M
        applicable_mask = item.applicability > 0.5

        # Inapplicable skeletons are revealed upfront
        revealed_mask = ~applicable_mask  # (M,) bool
        revealed_y = torch.zeros(M)
        revealed_f = torch.zeros(M)
        revealed_t = torch.zeros(M)

        policy.reset(item, self._dataset)

        total_time = 0.0
        attempts: list[int] = []

        while True:
            candidate_mask = applicable_mask & ~revealed_mask
            if not candidate_mask.any():
                break  # exhaustion

            next_idx = policy.select(
                candidate_mask, revealed_mask,
                revealed_y, revealed_f, revealed_t,
            )
            attempts.append(next_idx)
            total_time += item.refinement_time[next_idx].item()

            # Reveal outcome
            revealed_mask = revealed_mask.clone()
            revealed_mask[next_idx] = True
            revealed_y = revealed_y.clone()
            revealed_y[next_idx] = item.success[next_idx]
            revealed_f = revealed_f.clone()
            revealed_f[next_idx] = item.steps_completed_fraction[next_idx]
            revealed_t = revealed_t.clone()
            revealed_t[next_idx] = item.refinement_time[next_idx]

            if item.success[next_idx] > 0.5:
                return RolloutResult(
                    seed_id=item.seed_id,
                    ttfs=total_time,
                    n_attempts=len(attempts),
                    success=True,
                    attempt_indices=attempts,
                )

        # Exhaustion — no success found
        return RolloutResult(
            seed_id=item.seed_id,
            ttfs=None,
            n_attempts=len(attempts),
            success=False,
            attempt_indices=attempts,
        )

    def _compute_metrics(
        self,
        results: list[RolloutResult],
        baseline_results: list[RolloutResult] | None,
    ) -> EvalMetrics:
        n = len(results)

        # -- success@k ------------------------------------------------
        success_at_k: dict[int, float] = {}
        for k in self._k_values:
            hits = sum(1 for r in results if r.success and r.n_attempts <= k)
            success_at_k[k] = hits / n if n > 0 else 0.0

        # -- TTFS stats (successful instances only) --------------------
        ttfs_values = [r.ttfs for r in results if r.success and r.ttfs is not None]
        mean_ttfs = float(np.mean(ttfs_values)) if ttfs_values else float("inf")
        median_ttfs = float(np.median(ttfs_values)) if ttfs_values else float("inf")

        # -- cost ratios -----------------------------------------------
        mean_cost_ratio: float | None = None
        n_paired = 0
        exclusive_policy = 0
        exclusive_baseline = 0

        if baseline_results is not None:
            log_ratios: list[float] = []
            for r_pol, r_base in zip(results, baseline_results):
                pol_ok = r_pol.success and r_pol.ttfs is not None
                base_ok = r_base.success and r_base.ttfs is not None
                if pol_ok and base_ok:
                    assert r_pol.ttfs is not None and r_base.ttfs is not None
                    ratio = r_pol.ttfs / max(r_base.ttfs, 1e-9)
                    log_ratios.append(math.log(ratio))
                    n_paired += 1
                elif pol_ok and not base_ok:
                    exclusive_policy += 1
                elif not pol_ok and base_ok:
                    exclusive_baseline += 1

            if log_ratios:
                mean_cost_ratio = math.exp(sum(log_ratios) / len(log_ratios))

        return EvalMetrics(
            mean_ttfs=mean_ttfs,
            median_ttfs=median_ttfs,
            success_at_k=success_at_k,
            n_instances=n,
            n_succeeded=sum(1 for r in results if r.success),
            mean_cost_ratio=mean_cost_ratio,
            n_paired=n_paired,
            exclusive_success_policy=exclusive_policy,
            exclusive_success_baseline=exclusive_baseline,
            per_instance=results,
        )
