"""Refinement that reports *why* it failed, without changing what it decides.

The unified evidence construction (``unified_evidence.py``) needs, per failed candidate,
the failed step and the deviation between the abstract state the candidate predicted and
the one the sample actually produced. Upstream's :class:`BacktrackingRefiner` returns
``Plan | None`` and nothing else, and its sampler raises a payload-free
:class:`TrajectorySamplingFailure`.

**Observation-only is the hard invariant here, as it was for DD2D.** The deviation is not
recomputed: the acceptance check in
:class:`~alphatamp.approaches.spectre.envs.stickbutton2d.sampler.AcceptanceTrajectorySampler`
*already* abstracts the final state to decide accept-or-reject, and already stores both
sides. This module only keeps what that check threw away — no extra transition calls, no
extra abstractions, so labels are identical to an uninstrumented run. That is **measured,
not assumed**: ``tests/approaches/spectre/test_stickbutton2d_observational.py`` refines the
same pools under the same per-candidate seeds through upstream's sampler and this one and
requires the same labels back.

**On failure classes.** §2 of the design distinguishes class 1 (a validity check rejects
the sample before a successor state exists, and names the objects it rejected on) from
class 2 (the sample executes and the trace check finds a mismatch). StickButton2D produces
**only class 2**: kinder's motion model rejects a colliding transition by silently
declining to move (``base_env.py`` integrates, then discards on collision) rather than by
raising, and its collision check returns a bool without naming anything. There is no
object-naming check to read, so every SB2D record arrives through the deviation path. That
is not a gap in the instrumentation — it is what the environment affords.

The class-1 slot is emitted anyway, empty. Both channels stay wired on every environment,
so which class exists is a property of the *data* rather than a per-environment branch in
the consumer; an empty channel is provably inert downstream (``unified_evidence.waste``
abstains on an empty culprit pool, and ``covered`` skips blameless records for every
object).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.structs import RelationalAbstractState
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from relational_structs import GroundAtom, GroundOperator

from alphatamp.approaches.spectre.envs.stickbutton2d.sampler import (
    AcceptanceTrajectorySampler,
)
from alphatamp.approaches.spectre.unified_evidence import (
    Deviation,
    UnifiedRecord,
    blame,
)


@dataclass(frozen=True)
class _Rejection:
    """One rejected sample: the step attempted and how its result deviated."""

    step: GroundOperator
    expected: frozenset[GroundAtom]
    achieved: frozenset[GroundAtom]


class RecordingSampler(AcceptanceTrajectorySampler):
    """Acceptance sampler that keeps every rejection instead of discarding it.

    Accumulates into :attr:`rejections` across calls; the refiner backtracks, so one
    candidate produces many. :func:`refine_with_record` reduces them to the single
    record the evidence context wants.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.rejections: list[_Rejection] = []

    def __call__(  # type: ignore[override]
        self,
        x: object,
        s: RelationalAbstractState,
        a: GroundOperator,
        ns: RelationalAbstractState,
        bpg: BilevelPlanningGraph,
        rng: np.random.Generator,
    ) -> tuple[list[object], list[object]]:
        try:
            return super().__call__(x, s, a, ns, bpg, rng)  # type: ignore[arg-type]
        except TrajectorySamplingFailure:
            # `last_expected` / `last_actual` were computed by the acceptance test
            # itself.
            self.rejections.append(
                _Rejection(
                    step=a,
                    expected=frozenset(self.last_expected),  # type: ignore[arg-type]
                    achieved=frozenset(self.last_actual),  # type: ignore[arg-type]
                )
            )
            raise

    def clear(self) -> None:
        """Drop accumulated rejections — call between candidates."""
        self.rejections.clear()


def _deepest_rejection(
    rejections: Sequence[_Rejection], action_plan: Sequence[GroundOperator]
) -> Optional[tuple[int, _Rejection]]:
    """The rejection at the furthest step the refiner reached.

    Backtracking retries shallow steps after deep ones fail, so "last rejection" is the
    wrong reduction; the informative one is the deepest, which is the point the
    candidate actually got stuck at.
    """
    best: Optional[tuple[int, _Rejection]] = None
    for rej in rejections:
        index = next((j for j, op in enumerate(action_plan) if op == rej.step), None)
        if index is None:
            continue
        if best is None or index > best[0]:
            best = (index, rej)
    return best


def refine_with_record(
    refiner: BacktrackingRefiner,
    sampler: RecordingSampler,
    x0: object,
    state_plan: Sequence[RelationalAbstractState],
    action_plan: Sequence[GroundOperator],
    timeout_s: float,
    bpg: BilevelPlanningGraph,
) -> tuple[bool, Optional[UnifiedRecord]]:
    """Refine one candidate; on failure, return the record its own failure explains.

    Returns ``(succeeded, record)``. ``record`` is ``None`` on success, and also on the
    rare failure that produced no usable rejection (a timeout before any sample
    completed), where "no evidence" is the honest answer rather than a fabricated one.
    """
    sampler.clear()
    try:
        plan = refiner(x0, list(state_plan), list(action_plan), timeout_s, bpg)
        succeeded = plan is not None
    except BaseException:  # pylint: disable=broad-exception-caught
        succeeded = False

    if succeeded:
        return True, None

    deepest = _deepest_rejection(sampler.rejections, action_plan)
    if deepest is None:
        return False, None

    _, rej = deepest
    return False, UnifiedRecord(
        failed_step=rej.step,
        deviation=Deviation(
            added=rej.achieved - rej.expected,
            deleted=rej.expected - rej.achieved,
        ),
    )


def _atom_pairs(atoms: Iterable[GroundAtom]) -> list[list]:
    """``[[predicate, [arg, ...]], ...]`` — a picklable, canonicalisable atom list.

    Names rather than :class:`GroundAtom`s because this goes into ``refiner_metadata``,
    which ``canonicalize_episode`` rewrites: object identities have to be *visible as
    strings* there or the record's tags fail to resolve against the scene tags and the
    record silently degenerates to "some failure of some schema". Sorted so any
    downstream truncation is deterministic rather than a function of set iteration order.
    """
    return sorted(
        ([atom.predicate.name, [o.name for o in atom.objects]] for atom in atoms),
        key=repr,
    )


def failure_metadata(
    sampler: RecordingSampler,
    action_plan: Sequence[GroundOperator],
    num_sampling_attempts_per_step: int,
    budget_exhausted: bool,
) -> list[dict]:
    """The ``refiner_metadata["failures"]`` payload for one failed candidate.

    Same deepest-step reduction as :func:`refine_with_record`, so the serialized record
    and the in-memory one can never describe different failures. Returns ``[]`` when no
    sample completed (a timeout before the first rejection), because "no evidence" is the
    honest answer and a fabricated step index would be read as a real observation.

    ``exhausted`` is a genuine observation here, not an assumption: the sampler counted
    the rejections, so reaching ``num_sampling_attempts_per_step`` at the failing step
    means the refiner really did run that query to the end of its own retries.
    ``budget_exhausted`` is passed in because only the caller knows whether the wall-clock
    timeout fired, and it is exactly the case where ``exhausted`` proves nothing.
    """
    deepest = _deepest_rejection(sampler.rejections, action_plan)
    if deepest is None:
        return []
    index, rej = deepest
    n_step = sum(1 for r in sampler.rejections if r.step == rej.step)
    record = UnifiedRecord(
        failed_step=rej.step,
        deviation=Deviation(
            added=rej.achieved - rej.expected, deleted=rej.expected - rej.achieved
        ),
    )
    return [
        {
            "step_index": int(index),
            "schema": str(rej.step.name),
            "args": [p.name for p in rej.step.parameters],
            # Class 1 is structurally unavailable on this environment (see module
            # docstring). Emitted empty rather than omitted, so consumers see one schema.
            "culprits": [],
            "n_step": int(n_step),
            "exhausted": bool(n_step >= num_sampling_attempts_per_step),
            "budget_exhausted": bool(budget_exhausted),
            "dev_added": _atom_pairs(record.deviation.added),  # type: ignore[union-attr]
            "dev_deleted": _atom_pairs(
                record.deviation.deleted  # type: ignore[union-attr]
            ),
            # The objects the *collateral* part of the deviation names -- what the record
            # blames, once the failed step's own unachieved effects are stripped out.
            # Deliberately a separate key from `culprits`: a culprit was named by a check
            # the environment ran, this is inferred by us from the trace, and the record
            # token stream should not be able to confuse the two. Without it SB2D record
            # tokens would carry no object identity at all, and A17 measured that identity
            # at 1.28 FP on DD2D.
            "dev_blame": sorted(blame(record)),
        }
    ]
