"""The environment contract a VLMPlan baseline needs — everything else is shared.

A new environment implements :class:`EnvAdapter` (and, if it wants images, returns them
from :meth:`EnvAdapter.images`); `template.py`, `parsing.py`, `models.py` and `loop.py`
stay untouched. DD2D's implementation is `dd2d_adapter.py`, the only env-aware module in
this package.

Scoring is deliberately *not* on this interface. Turning a proposed plan into a
feasibility label is a heavyweight, env-specific, offline operation (DD2D live-refines
off-pool proposals against a reconstructed scene), and keeping it out of the generation
path is what lets a run be re-scored after a re-collection without re-querying the model.
See `score.py`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Sequence

from PIL.Image import Image

# One parsed skill invocation: ``(skill_name, (object_name, ...))``. Continuous
# parameters are parsed and validated but never carried — the VLM's job ends at the
# skeleton, and a downstream sampler chooses low-level poses.
Step = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class RawPlan:
    """One ``Plan N:`` block, parsed into steps but not yet checked for applicability."""

    steps: tuple[Step, ...]
    block_index: int
    text: str


@dataclass(frozen=True)
class SkillSpec:
    """A skill as the KinDER template presents it: name, argument types, param box."""

    name: str
    types: tuple[str, ...]
    num_params: int = 0


class EnvAdapter(abc.ABC):
    """Everything VLMPlan needs to know about one environment.

    ``problem`` is whatever the adapter's own runner hands it (for DD2D, a canonicalized
    ``EpisodeRecord``); the shared code treats it as opaque.
    """

    # --- vocabulary the parser validates against -------------------------------

    @abc.abstractmethod
    def skills(self, problem: object) -> dict[str, SkillSpec]:
        """Skill name -> spec, for parser validation."""

    @abc.abstractmethod
    def objects(self, problem: object) -> dict[str, str]:
        """Object name -> type name."""

    @abc.abstractmethod
    def type_ancestors(self, type_name: str) -> frozenset[str]:
        """``type_name`` plus all its ancestors, for the parser's subtype check."""

    # --- prompt content --------------------------------------------------------

    @abc.abstractmethod
    def controllers_str(self, problem: object) -> str:
        """The skills block, in the KinDER ``ParameterizedController`` format."""

    @abc.abstractmethod
    def typed_objects_str(self, problem: object) -> str:
        """``<object_name>: <type_name>`` lines."""

    @abc.abstractmethod
    def type_hierarchy_str(self, problem: object) -> str:
        """PDDL-style ``<child> ... - <parent>`` lines."""

    @abc.abstractmethod
    def goal_str(self, problem: object) -> str:
        """The goal expression, including any domain-semantics disclosure."""

    @abc.abstractmethod
    def init_state_str(self, problem: object) -> str:
        """The initial state as text (literals, and any geometry the arm discloses)."""

    @abc.abstractmethod
    def images(self, problem: object) -> list[Image]:
        """Images to attach. Empty list = the text-only (LLMPlan) arm."""

    # --- output handling -------------------------------------------------------

    @abc.abstractmethod
    def ground(self, raw: RawPlan, problem: object) -> tuple[Step, ...] | None:
        """Check symbolic applicability; return the canonical step sequence or ``None``.

        ``None`` means the plan is invalid (inapplicable, or does not reach the goal)
        and should be dropped without costing a refinement attempt.
        """

    @abc.abstractmethod
    def canonical_key(self, steps: Sequence[Step]) -> tuple[object, ...]:
        """Dedup key. Must distinguish plans that a refiner would treat differently."""

    @abc.abstractmethod
    def plan_str(self, steps: Sequence[Step]) -> str:
        """Render a plan back into the template's line format, for the repeat block."""

    @abc.abstractmethod
    def published_order(self, problem: object) -> list[tuple[Step, ...]]:
        """The planner's own ordering — the fallback an exhausted episode degrades to."""

    # --- scoring support -------------------------------------------------------
    # These were DD2D-only helpers that `score.py` reached for directly until
    # 2026-08-01; promoting them here is what lets the scorer be env-agnostic.

    @abc.abstractmethod
    def pool_index(self, problem: object) -> dict[tuple[object, ...], int]:
        """``canonical_key`` -> pool index, for matching proposals against stored labels.

        A proposal that hits this map is labelled from the stored ``OutcomeRecord``
        and is **never** re-refined, so VLMPlan sees byte-identical labels to every
        other method.
        """

    @abc.abstractmethod
    def discretionary_objects(self, steps: Sequence[Step]) -> list[str]:
        """The objects this plan chose to act on, in order — for diagnostics only.

        DD2D: the items staged to the buffer. StickButton2D: the press order. Never used
        to decide a label, only to make an attempt legible in the record.
        """


class Labeler(abc.ABC):
    """Feasibility for a proposal the candidate pool does not contain.

    Off-pool proposals are **refined for real** rather than dropped: silently discarding
    them would hand VLMPlan free attempts and flatter it. That refinement is heavyweight
    and env-specific, which is why it is a separate object from :class:`EnvAdapter` and
    off the generation path entirely — a run can be re-scored after a re-collection
    without re-querying the model.

    **The refiner settings must be the collection's own.** Off-pool labels are computed
    now; in-pool labels came off disk. They are drawn from the same distribution only if
    the refiner matches the one the collection ran, seed rule included.
    ``label_agreement``
    in ``score.py`` is the gate that measures whether that held.
    """

    n_refines: int = 0

    @abc.abstractmethod
    def label(self, episode: object, steps: Sequence[Step]) -> str:
        """``"success"`` or ``"fail"`` for one off-pool proposal."""

    def flush(self) -> None:
        """Persist any memo. Default: nothing to persist."""
