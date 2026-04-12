"""Rollout-consistent prefix generator for training.

Converts a fully-labeled instance (all M candidate skeletons with known
outcomes) into a sequence of PrefixStep training examples that match the
test-time sequential reveal protocol:

1. Start with empty history (only inapplicable skeletons revealed).
2. Reveal one applicable skeleton per step.
3. Terminate on first Y=1 (success) or exhaustion of all applicable candidates.

Three reveal strategies are supported:

- **teacher_forced**: Reveal in ascending refinement-time order.
- **epsilon_random**: With probability epsilon, deviate to a uniform random
  reveal; otherwise follow the teacher order.
- **on_policy**: Use a provided scoring callable to select the next reveal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

__all__ = ["PrefixStep", "PrefixGenerator"]


@dataclass(frozen=True)
class PrefixStep:
    """One step of a rollout-consistent prefix sequence.

    Represents the belief state at a single point in the sequential reveal
    protocol, paired with supervision targets.

    Attributes
    ----------
    applicability:
        (M,) float32 binary {0,1}. Constant across steps of the same instance.
    lengths:
        (M,) float32. Skeleton lengths (number of operators). Constant across
        steps of the same instance.
    revealed_mask:
        (M,) bool. True where a skeleton has been revealed (includes
        inapplicable skeletons, which are revealed from the start).
    revealed_outcomes:
        Dict with keys ``'y'``, ``'f'``, ``'t'``, each (M,) float32.
        Observed values for revealed positions, 0.0 elsewhere.
    y_true:
        (M,) float32. Full ground-truth success labels.
    f_true:
        (M,) float32. Full ground-truth steps-completed fraction.
    t_true:
        (M,) float32. Full ground-truth refinement time.
    oracle_ranking:
        (M,) int64. Listwise rank over C_t (unrevealed applicable candidates).
        0-indexed, lower is better. Successes ranked by ascending T before
        failures. -1 for positions not in C_t.
    step_index:
        0-based step counter within this prefix sequence.
    """

    applicability: torch.Tensor
    lengths: torch.Tensor
    revealed_mask: torch.Tensor
    revealed_outcomes: dict[str, torch.Tensor]
    y_true: torch.Tensor
    f_true: torch.Tensor
    t_true: torch.Tensor
    oracle_ranking: torch.Tensor
    step_index: int


def _compute_oracle_ranking(
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute oracle ranking over candidate set C_t.

    Successes (Y=1) are ranked first in ascending T order, then failures
    (Y=0) in ascending T order. Ties broken by ascending index.

    Parameters
    ----------
    y_true:
        (M,) float32 ground-truth success.
    t_true:
        (M,) float32 ground-truth refinement time.
    candidate_mask:
        (M,) bool. True for unrevealed applicable candidates.

    Returns
    -------
    (M,) int64. Rank per position (0-indexed, lower = better).
    -1 for positions not in C_t.
    """
    M = y_true.shape[0]
    ranking = torch.full((M,), -1, dtype=torch.int64)

    cand_idx = torch.where(candidate_mask)[0]
    if len(cand_idx) == 0:
        return ranking

    y_cand = y_true[cand_idx]
    t_cand = t_true[cand_idx]

    # Compound sort key: failures (1-y=1) sort after successes (1-y=0),
    # then by ascending T, then by ascending index for determinism.
    is_failure = (y_cand < 0.5).to(torch.float64)
    sort_key = is_failure * 1e12 + t_cand.double() + cand_idx.double() * 1e-8
    order = torch.argsort(sort_key)

    for rank, pos in enumerate(order):
        ranking[cand_idx[pos]] = rank

    return ranking


class PrefixGenerator:
    """Convert a fully-labeled instance into rollout-consistent prefix examples.

    Parameters
    ----------
    mode:
        Reveal strategy.
    epsilon:
        Random deviation probability for ``"epsilon_random"`` mode. Ignored
        for other modes.
    """

    def __init__(
        self,
        mode: Literal["teacher_forced", "epsilon_random", "on_policy"],
        epsilon: float = 0.0,
    ) -> None:
        if mode not in ("teacher_forced", "epsilon_random", "on_policy"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if mode == "epsilon_random" and not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._mode = mode
        self._epsilon = epsilon

    def generate(
        self,
        applicability: torch.Tensor,
        success: torch.Tensor,
        steps_completed_fraction: torch.Tensor,
        refinement_time: torch.Tensor,
        lengths: torch.Tensor,
        *,
        score_fn: Callable[[PrefixStep], torch.Tensor] | None = None,
        rng: torch.Generator | None = None,
    ) -> list[PrefixStep]:
        """Generate a prefix sequence for one instance.

        Parameters
        ----------
        applicability:
            (M,) float32 binary.
        success:
            (M,) float32 binary.
        steps_completed_fraction:
            (M,) float32 in [0, 1].
        refinement_time:
            (M,) float32 >= 0.
        lengths:
            (M,) float32 >= 0. Number of operators per skeleton. Constant
            across steps (passed through to each PrefixStep unchanged).
        score_fn:
            Callable that takes a ``PrefixStep`` and returns (M,) scores.
            Required for ``"on_policy"`` mode.
        rng:
            ``torch.Generator`` for reproducible randomness in
            ``"epsilon_random"`` mode.

        Returns
        -------
        List of :class:`PrefixStep` objects, one per decision point.
        """
        if self._mode == "on_policy" and score_fn is None:
            raise ValueError("score_fn is required for on_policy mode")

        M = applicability.shape[0]
        applicable_mask = applicability > 0.5  # (M,) bool

        # Ground truth (constant across steps)
        y_true = success
        f_true = steps_completed_fraction
        t_true = refinement_time

        # Precompute teacher ordering: applicable indices sorted by ascending T
        applicable_indices = torch.where(applicable_mask)[0]
        if len(applicable_indices) > 0:
            t_for_sort = refinement_time[applicable_indices]
            sort_key = t_for_sort.double() + applicable_indices.double() * 1e-8
            sort_order = torch.argsort(sort_key)
            teacher_order = applicable_indices[sort_order].tolist()
        else:
            teacher_order = []

        # Mutable state
        revealed_mask = ~applicable_mask  # inapplicable always revealed
        revealed_y = torch.zeros(M, dtype=torch.float32)
        revealed_f = torch.zeros(M, dtype=torch.float32)
        revealed_t = torch.zeros(M, dtype=torch.float32)

        steps: list[PrefixStep] = []
        step_idx = 0

        while True:
            # Current candidate set
            candidate_mask = applicable_mask & ~revealed_mask

            # Oracle ranking over C_t
            oracle_ranking = _compute_oracle_ranking(y_true, t_true, candidate_mask)

            # Emit current state
            step = PrefixStep(
                applicability=applicability,
                lengths=lengths,
                revealed_mask=revealed_mask.clone(),
                revealed_outcomes={
                    "y": revealed_y.clone(),
                    "f": revealed_f.clone(),
                    "t": revealed_t.clone(),
                },
                y_true=y_true,
                f_true=f_true,
                t_true=t_true,
                oracle_ranking=oracle_ranking,
                step_index=step_idx,
            )
            steps.append(step)

            # Termination: no candidates left (exhaustion)
            if not candidate_mask.any():
                break

            # Pick next skeleton to reveal
            next_idx = self._pick_next(
                step, candidate_mask, teacher_order, revealed_mask,
                score_fn, rng,
            )

            # Reveal
            revealed_mask[next_idx] = True
            revealed_y[next_idx] = y_true[next_idx]
            revealed_f[next_idx] = f_true[next_idx]
            revealed_t[next_idx] = t_true[next_idx]

            # Termination: success found
            if y_true[next_idx] > 0.5:
                break

            step_idx += 1

        return steps

    def _pick_next(
        self,
        step: PrefixStep,
        candidate_mask: torch.Tensor,
        teacher_order: list[int],
        revealed_mask: torch.Tensor,
        score_fn: Callable[[PrefixStep], torch.Tensor] | None,
        rng: torch.Generator | None,
    ) -> int:
        """Select the next skeleton to reveal based on mode."""
        if self._mode == "teacher_forced":
            return self._teacher_next(teacher_order, revealed_mask)

        if self._mode == "epsilon_random":
            deviate = torch.rand(1, generator=rng).item() < self._epsilon
            if deviate:
                return self._random_candidate(candidate_mask, rng)
            return self._teacher_next(teacher_order, revealed_mask)

        # on_policy
        assert score_fn is not None
        scores = score_fn(step)
        masked_scores = scores.clone().detach()
        masked_scores[~candidate_mask] = float("-inf")
        return int(masked_scores.argmax().item())

    @staticmethod
    def _teacher_next(teacher_order: list[int], revealed_mask: torch.Tensor) -> int:
        """Return the first unrevealed entry in teacher_order."""
        for idx in teacher_order:
            if not revealed_mask[idx]:
                return idx
        raise RuntimeError("No unrevealed teacher candidate")  # pragma: no cover

    @staticmethod
    def _random_candidate(
        candidate_mask: torch.Tensor, rng: torch.Generator | None,
    ) -> int:
        """Return a uniformly random candidate from C_t."""
        cand_indices = torch.where(candidate_mask)[0]
        rand_pos = torch.randint(len(cand_indices), (1,), generator=rng).item()
        return int(cand_indices[rand_pos].item())
