"""Plug-in static priors for SPECTRE.

Per ``SPECTRE_METHOD_SPEC.md`` §4.4, the prior is a per-skeleton scalar score
from any context-independent ranker. v0.1 ships only ``ZeroPrior``. A
``BasePrior`` ABC keeps the swap-in interface stable for future HSR / PIGINet.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphatamp.approaches.spectre.schema import EpisodeRecord, SkeletonRecord


class BasePrior(abc.ABC):
    """Interface for a static per-skeleton prior π(s)."""

    @abc.abstractmethod
    def score(
        self,
        problem_id: int,
        skeleton_idx: int,
        skeleton: "SkeletonRecord",
        episode: "EpisodeRecord",
    ) -> float:
        """Return the prior score for the given skeleton."""


class ZeroPrior(BasePrior):
    """Π ≡ 0.

    The "no prior" baseline per §4.4.1.
    """

    def score(
        self,
        problem_id: int,
        skeleton_idx: int,
        skeleton: "SkeletonRecord",
        episode: "EpisodeRecord",
    ) -> float:
        del problem_id, skeleton_idx, skeleton, episode
        return 0.0
