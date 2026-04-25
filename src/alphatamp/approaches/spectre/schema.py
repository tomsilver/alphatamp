"""On-disk schema for SPECTRE episode records.

Every field is either a primitive Python type or a frozen dataclass from
``relational_structs`` / ``bilevel_planning`` whose constituents are also
frozen dataclasses (``RelationalAbstractState``, ``GroundOperator``,
``GroundAtom``, ``Object``, ``Type``, ``Predicate``). These pickle stably so
we do not need the plain-dicts discipline the pipeline spec §5.1 proposes.

We deliberately do *not* serialize:

- ``x0`` (``ObjectCentricState``) — SPECTRE's skeleton encoder Φ never sees
  continuous state; including it bloats records and pulls in numpy arrays
  whose pickle stability is weaker.
- ``RelationalAbstractGoal`` — carries a ``state_abstractor`` callable field
  that is not reliably pickleable across sessions. We store ``goal_atoms``
  instead; the callable can be reattached from env_models at load time if
  ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundAtom, GroundOperator

Outcome = Literal["success", "fail", "error"]


@dataclass(frozen=True)
class ProvenanceBlock:
    """Per-episode provenance, pipeline spec §5.3.

    ``scene_latent`` is populated only by environments whose refinement is
    governed by an externally-sampled per-episode latent (currently:
    RoutedTransport2D, where it carries ``(blocked_color, blocked_grasp)``).
    For the kinder envs it stays ``None`` and round-trips unchanged.
    """

    problem_id: int
    env_id: str
    env_variant: str
    split: str
    config_hash: str
    problem_seed: int
    git_sha: str
    collection_timestamp: str
    package_versions: dict[str, str]
    scene_latent: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class SummaryBlock:
    """Per-episode summary, pipeline spec §5.7."""

    num_skeletons: int
    num_success: int
    num_fail: int
    num_error: int
    first_success_idx: Optional[int]
    total_wall_clock_s: float
    pool_truncated: bool


@dataclass(frozen=True)
class SkeletonRecord:
    """One candidate skeleton, pipeline spec §5.5 (Substage A)."""

    skeleton_idx: int
    operator_seq: tuple[GroundOperator, ...]
    final_abstract_state: RelationalAbstractState


@dataclass(frozen=True)
class OutcomeRecord:
    """Outcome of one per-skeleton refinement attempt, pipeline spec §5.6."""

    skeleton_idx: int
    outcome: Outcome
    refinement_wall_clock_s: float
    refinement_seed: int
    stuck_step_index: Optional[int] = None
    sampler_retries: Optional[int] = None
    error_info: Optional[dict[str, str]] = None
    refiner_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    """One collected episode.

    Invariants (asserted in ``validate``):

    - I1. ``len(outcomes) == len(skeleton_pool)``
    - I2. every ``outcomes[i].skeleton_idx == skeleton_pool[i].skeleton_idx == i``
    - I3. summary counts sum to ``num_skeletons``
    - I4. if ``first_success_idx`` is set, that outcome is ``"success"``
    """

    provenance: ProvenanceBlock
    initial_abstract_state: RelationalAbstractState
    goal_atoms: frozenset[GroundAtom]
    object_registry: dict[str, str]  # obj_name -> type_name
    skeleton_pool: tuple[SkeletonRecord, ...]
    outcomes: tuple[OutcomeRecord, ...]
    summary: SummaryBlock

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check invariants I1–I4.

        Raises AssertionError on violation.
        """
        n = len(self.skeleton_pool)
        assert (
            len(self.outcomes) == n
        ), f"I1 violated: {len(self.outcomes)} outcomes vs {n} skeletons"
        for i, (s, o) in enumerate(zip(self.skeleton_pool, self.outcomes)):
            assert s.skeleton_idx == i, f"I2 violated: skeleton_pool[{i}].skeleton_idx"
            assert o.skeleton_idx == i, f"I2 violated: outcomes[{i}].skeleton_idx"
        c = self.summary
        assert c.num_success + c.num_fail + c.num_error == c.num_skeletons, (
            f"I3 violated: {c.num_success}+{c.num_fail}+{c.num_error}"
            f" != {c.num_skeletons}"
        )
        assert (
            c.num_skeletons == n
        ), f"I3 violated: summary.num_skeletons={c.num_skeletons} != {n}"
        if c.first_success_idx is not None:
            assert (
                self.outcomes[c.first_success_idx].outcome == "success"
            ), f"I4 violated: first_success_idx={c.first_success_idx} is not success"

    def success_indices(self) -> list[int]:
        """Skeleton indices whose outcome is ``"success"``."""
        return [o.skeleton_idx for o in self.outcomes if o.outcome == "success"]

    def fail_indices(self) -> list[int]:
        """Skeleton indices whose outcome is ``"fail"``."""
        return [o.skeleton_idx for o in self.outcomes if o.outcome == "fail"]

    def error_indices(self) -> list[int]:
        """Skeleton indices whose outcome is ``"error"`` (excluded from training)."""
        return [o.skeleton_idx for o in self.outcomes if o.outcome == "error"]
