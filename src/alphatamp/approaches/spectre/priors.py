"""Plug-in static priors for SPECTRE.

Per ``docs/archive/SPECTRE_METHOD_SPEC.md`` §4.4, the prior is a per-skeleton
scalar score from any context-independent ranker. One implementation ships
today:

- :class:`ZeroPrior` — π ≡ 0 (the "no prior" reference baseline).

A ``BasePrior`` ABC keeps the swap-in interface stable for future HSR /
PIGINet priors.
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


# ---------------------------------------------------------------------------
# Factory — single source of truth for "name → BasePrior instance"
# ---------------------------------------------------------------------------


def make_prior(prior_type: str) -> BasePrior:
    """Construct a ``BasePrior`` from the string name training/inference saves.

    Used by ``train.py`` (to build the prior for a run) and by
    ``inference.load_prior_for_checkpoint`` (to reconstruct the matching
    prior at eval time). Keep these two callers in sync — adding a new
    prior class means adding it here.
    """
    if prior_type == "zero":
        return ZeroPrior()
    raise ValueError(f"Unknown prior_type={prior_type!r}; expected 'zero'")
