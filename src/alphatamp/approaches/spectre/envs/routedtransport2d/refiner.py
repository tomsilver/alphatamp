"""Three-gate latent + tag refiner (spec §5.2).

Pure function of (skeleton, scene_latent, per-problem tags). No kinematic
simulation; no continuous-parameter sampling. Three gates evaluated per
operator in order:

  Gate 1 (blocked_color):  TraverseLoaded<X> fails iff X == scene_latent.blocked_color
  Gate 2 (size_width):     TraverseLoaded<X> fails iff size(item) > width(passage)
  Gate 3 (blocked_grasp):  PickItemTop fails iff blocked_grasp == "top"
                          PickItemSide fails iff blocked_grasp == "side"
  Plus residual noise: each op fails with probability ``base_op_fail_rate``.

Place ops are not gated by ``blocked_grasp`` — see spec §5.2 design rationale.

The class :class:`ThreeGateRefiner` is duck-compatible with
``BacktrackingRefiner`` so collect.py's per-skeleton refinement loop calls it
with the same interface ``__call__(x0, state_plan, action_plan, timeout_s, bpg)
-> Plan | None``. Unlike BacktrackingRefiner this returns either a stub-Plan
on success or None on fail, but additionally exposes ``last_outcome`` carrying
the structured failure cause; the dispatcher in collect.py reads that field
into ``OutcomeRecord.refiner_metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from bilevel_planning.structs import Plan, RelationalAbstractState
from relational_structs import GroundOperator

from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    ProblemInstance,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.tags import is_compatible


@dataclass(frozen=True)
class RefineOutcome:
    """Outcome of one three-gate refinement attempt.

    ``stuck_step_index`` is the 0-based index in the action plan where the gate
    fired (None on success). ``stuck_cause`` ∈ {None, "blocked_color",
    "size_width", "blocked_grasp", "noise"}.
    """

    success: bool
    stuck_step_index: Optional[int]
    stuck_cause: Optional[str]
    stuck_op_name: Optional[str]
    wall_clock_s: float


def _sample_fail_time(
    step_idx: int, plan_length: int, rng: np.random.Generator
) -> float:
    """Gamma(α=1.0 + 0.3·step_idx, β=1.0). Spec §5.2 wall-clock model.

    Earlier failures sample shorter wall-clock values; this matches "fail-fast" behavior
    in the diagnostic metrics (spec §7.4).
    """
    del plan_length
    alpha = 1.0 + 0.3 * step_idx
    return float(rng.gamma(alpha, 1.0))


def _sample_success_time(plan_length: int, rng: np.random.Generator) -> float:
    """Gamma(α=1.0 + 0.3·L, β=1.0).

    Successes scale with plan length.
    """
    alpha = 1.0 + 0.3 * plan_length
    return float(rng.gamma(alpha, 1.0))


def refine(
    operator_seq: Sequence[GroundOperator],
    *,
    blocked_color: str,
    blocked_grasp: str,
    passage_widths: dict[str, str],
    item_sizes: dict[str, str],
    rng: np.random.Generator,
    base_op_fail_rate: float = 0.02,
) -> RefineOutcome:
    """Apply the three gates in order; return the structured outcome.

    Looks up passage and item names directly from each ground op's parameters.
    The argument order conventions (set in operators.py):

    - PickItemTop/Side / PlaceItemTop/Side: (robot, item, zone)
    - TraverseEmpty: (robot, passage, src, dst)
    - TraverseLoadedColorX: (robot, passage, src, dst, item)
    """
    plan_length = len(operator_seq)
    for i, op in enumerate(operator_seq):
        op_name = op.name

        if op_name.startswith("TraverseLoadedColor"):
            color = op_name[-1]
            passage_obj = op.parameters[1]
            item_obj = op.parameters[4]
            # Gate 1: blocked color.
            if color == blocked_color:
                return RefineOutcome(
                    success=False,
                    stuck_step_index=i,
                    stuck_cause="blocked_color",
                    stuck_op_name=op_name,
                    wall_clock_s=_sample_fail_time(i, plan_length, rng),
                )
            # Gate 2: size-width compatibility.
            size = item_sizes[item_obj.name]
            width = passage_widths[passage_obj.name]
            if not is_compatible(size, width):
                return RefineOutcome(
                    success=False,
                    stuck_step_index=i,
                    stuck_cause="size_width",
                    stuck_op_name=op_name,
                    wall_clock_s=_sample_fail_time(i, plan_length, rng),
                )

        # Gate 3: blocked grasp (pick ops only; spec §5.2 design rationale).
        if op_name == "PickItemTop" and blocked_grasp == "top":
            return RefineOutcome(
                success=False,
                stuck_step_index=i,
                stuck_cause="blocked_grasp",
                stuck_op_name=op_name,
                wall_clock_s=_sample_fail_time(i, plan_length, rng),
            )
        if op_name == "PickItemSide" and blocked_grasp == "side":
            return RefineOutcome(
                success=False,
                stuck_step_index=i,
                stuck_cause="blocked_grasp",
                stuck_op_name=op_name,
                wall_clock_s=_sample_fail_time(i, plan_length, rng),
            )

        # Residual per-op noise.
        if rng.random() < base_op_fail_rate:
            return RefineOutcome(
                success=False,
                stuck_step_index=i,
                stuck_cause="noise",
                stuck_op_name=op_name,
                wall_clock_s=_sample_fail_time(i, plan_length, rng),
            )

    return RefineOutcome(
        success=True,
        stuck_step_index=None,
        stuck_cause=None,
        stuck_op_name=None,
        wall_clock_s=_sample_success_time(plan_length, rng),
    )


class ThreeGateRefiner:
    """Per-skeleton refiner; duck-compatible with ``BacktrackingRefiner``.

    Callers create one per refinement attempt (matches BacktrackingRefiner
    usage in collect.py:165) and read :attr:`last_outcome` after the call to
    extract structured failure metadata.
    """

    def __init__(
        self,
        problem: ProblemInstance,
        *,
        seed: int,
        base_op_fail_rate: float = 0.02,
    ) -> None:
        self._problem = problem
        self._rng = np.random.default_rng(seed)
        self._base_op_fail_rate = base_op_fail_rate
        self.last_outcome: Optional[RefineOutcome] = None

    def __call__(
        self,
        x0: object,
        state_plan: Sequence[RelationalAbstractState],
        action_plan: Sequence[GroundOperator],
        timeout_s: float,
        bpg: object,
    ) -> Optional[Plan]:
        """Run the three-gate model. Return a stub ``Plan`` on success, ``None`` on
        failure. Sets :attr:`last_outcome` for the caller to inspect.

        ``x0``, ``timeout_s``, ``bpg`` are accepted for interface compat. The
        returned :class:`Plan` carries the abstract trajectory in ``states``
        and an empty actions list — RT2D has no continuous actions, so we
        substitute a single dummy action to satisfy ``Plan.__post_init__``
        (which requires ``len(states) == len(actions) + 1``).
        """
        del timeout_s, bpg
        outcome = refine(
            action_plan,
            blocked_color=self._problem.blocked_color,
            blocked_grasp=self._problem.blocked_grasp,
            passage_widths=self._problem.passage_widths,
            item_sizes=self._problem.item_sizes,
            rng=self._rng,
            base_op_fail_rate=self._base_op_fail_rate,
        )
        self.last_outcome = outcome
        if not outcome.success:
            return None
        # Stub plan: one dummy action per state-transition. RT2D doesn't use
        # continuous actions, so a None list satisfies the structural contract.
        actions: list[object] = [None] * (len(state_plan) - 1)
        return Plan(states=list(state_plan), actions=actions)
