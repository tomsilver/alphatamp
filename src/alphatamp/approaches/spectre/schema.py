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
from typing import Any, Literal, Optional

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

    ``gen_params`` (v3, trailing-nullable like the v2.2.1 geometry layer) records the
    generator/refiner arguments an episode was produced under. It exists so provenance
    is auditable rather than inferred -- omitting it is what forced the
    "reconstruct, don't regenerate" rule (``decisions.md`` 2026-07-19). It is an **audit
    trail, never a model input**: it carries ``stratum``, which is the answer, and
    nothing in the dataset/tensorizer path reads ``ProvenanceBlock``.
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
    gen_params: Optional[dict[str, Any]] = None


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


# --------------------------------------------------------------------------- #
# v2.2.1 geometry / evidence layer (all OPTIONAL and nullable; RT2D/kinder
# records leave these ``None`` and round-trip unchanged — see the migration shim
# in ``io.load_episode``). SPECTRE stays abstract-first; geometry is carried for
# the v2.2 geometry-aware model and the typed post-mortem evidence pathway.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObjectGeometry:
    """One object's ground-truth footprint (proposal §6.1 scene record).

    ``boundary`` is the exterior ring in the item frame (centroid at the origin),
    consumed by the footprint point-set encoder; ``pose`` places it in the world.
    """

    name: str
    pose: tuple[float, float, float]  # (x, y, theta), world frame
    boundary: tuple[
        tuple[float, float], ...
    ]  # exterior ring, item frame, centroid at 0
    family: str
    area: float
    concave: bool
    is_target: bool = False


@dataclass(frozen=True)
class ContainerGeometry:
    """A container / free-space region (drawer, buffer, wall band)."""

    kind: str  # e.g. "drawer" | "buffer" | "wall_band"
    bounds: tuple[float, float, float, float]  # axis-aligned (x0, y0, x1, y1)
    polygon: Optional[tuple[tuple[float, float], ...]] = None  # exact ring if non-rect


@dataclass(frozen=True)
class SceneGeometry:
    """Ground-truth object-centric scene geometry for one episode (proposal §6.1)."""

    objects: tuple[ObjectGeometry, ...]
    containers: tuple[ContainerGeometry, ...]
    units: str = "cm"
    frame: Optional[dict[str, float]] = None  # e.g. drawer_wh for normalization


@dataclass(frozen=True)
class Fact:
    """One typed post-mortem fact (proposal §6.4). ``args`` are pre-canonical object
    names (the witness set); ``tier`` is the proof/hint split."""

    fact_type: str
    args: tuple[str, ...]
    tier: str  # "proof" | "hint"
    schema: Optional[str] = None  # failing action schema, when applicable
    scalars: tuple[tuple[str, float], ...] = ()  # (name, value): depth, samples, ...


@dataclass(frozen=True)
class PostMortemRecord:
    """The typed record harvested from one failed refinement attempt (proposal §6.2)."""

    skeleton_idx: int
    refinement_seed: int
    failed_step_index: Optional[int] = None  # ℓ*+1
    failed_schema: Optional[str] = None
    failed_args: tuple[str, ...] = ()
    harvest_prefix: tuple[str, ...] = ()  # replayable bound-prefix action reprs
    harvest_state_hash: Optional[str] = None
    facts: tuple[Fact, ...] = ()
    harvest_cost_s: float = 0.0


@dataclass(frozen=True)
class AuxLabels:
    """Per-episode auxiliary supervision (proposal §8): ``necessary(o)`` (in every
    minimal feasible subset) and ``relevant(o)`` (in at least one)."""

    necessary: frozenset[str] = frozenset()
    relevant: frozenset[str] = frozenset()


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
    # v2.2.1: typed post-mortem evidence (populated for "fail" outcomes at collection).
    post_mortem: Optional[PostMortemRecord] = None


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
    # v2.2.1: optional geometry / evidence layer (None for RT2D/kinder; see io shim).
    scene_geometry: Optional[SceneGeometry] = None
    aux_labels: Optional[AuxLabels] = None

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
        # I5 (guarded): every registered object has ground-truth geometry.
        if self.scene_geometry is not None:
            geo_names = {o.name for o in self.scene_geometry.objects}
            missing = set(self.object_registry) - geo_names
            assert (
                not missing
            ), f"I5 violated: object_registry keys w/o geometry: {missing}"
        # I6 (guarded): a post_mortem indexes its own "fail" outcome.
        for o in self.outcomes:
            if o.post_mortem is not None:
                assert o.post_mortem.skeleton_idx == o.skeleton_idx, (
                    f"I6 violated: post_mortem.skeleton_idx={o.post_mortem.skeleton_idx}"
                    f" != outcome {o.skeleton_idx}"
                )
                assert (
                    o.outcome == "fail"
                ), f"I6 violated: post_mortem on non-fail outcome {o.skeleton_idx}"

    def success_indices(self) -> list[int]:
        """Skeleton indices whose outcome is ``"success"``."""
        return [o.skeleton_idx for o in self.outcomes if o.outcome == "success"]

    def fail_indices(self) -> list[int]:
        """Skeleton indices whose outcome is ``"fail"``."""
        return [o.skeleton_idx for o in self.outcomes if o.outcome == "fail"]

    def error_indices(self) -> list[int]:
        """Skeleton indices whose outcome is ``"error"`` (excluded from training)."""
        return [o.skeleton_idx for o in self.outcomes if o.outcome == "error"]
