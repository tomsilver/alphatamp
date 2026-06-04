"""Plug-in static priors for SPECTRE.

Per ``docs/archive/SPECTRE_METHOD_SPEC.md`` §4.4, the prior is a per-skeleton
scalar score
from any context-independent ranker. Two implementations ship today:

- :class:`ZeroPrior` — π ≡ 0 (the "no prior" reference baseline).
- :class:`HeuristicPrior` — RT2D-specific. Sums the pyperplan FF heuristic
  along each stored skeleton's STRIPS trajectory, then per-episode z-scores
  the negated sum so π=0 ≈ "no opinion" and higher π = "FF prefers it".
  Mean-centering is intentional: it lines up with ``Scorer.prior_dropout_p``,
  which zeroes the prior 20% of the time to teach the model the
  no-information case.

A ``BasePrior`` ABC keeps the swap-in interface stable for future HSR /
PIGINet priors.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

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
# Shared FF-trajectory scoring (used by HeuristicPrior + eda.heuristic_search_baseline)
# ---------------------------------------------------------------------------


def ff_trajectory_scores(
    domain_gen: Any,
    episode: "EpisodeRecord",
) -> np.ndarray:
    """Per-skeleton ``Σᵢ h_FF(sᵢ)`` over the STRIPS trajectory.

    ``domain_gen`` is a ``RelationalHeuristicSearchAbstractPlanGenerator`` —
    we reuse its private ``_heuristic_factory`` to get a per-episode
    pyperplan FF heuristic. Lower is better; callers that want a "high =
    good" prior should negate before normalizing.

    Returns a ``(K,)`` float32 array aligned with ``episode.skeleton_pool``.
    """
    # pylint: disable=import-outside-toplevel
    from bilevel_planning.structs import RelationalAbstractGoal

    from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

    goal = RelationalAbstractGoal(
        atoms=set(episode.goal_atoms),
        # The FF factory only consults ``goal.atoms``; ``state_abstractor``
        # is never invoked here, so a sentinel is fine.
        state_abstractor=lambda x: x,
    )
    h_func = domain_gen._heuristic_factory(  # pylint: disable=protected-access
        episode.initial_abstract_state, goal
    )
    out = np.empty(len(episode.skeleton_pool), dtype=np.float32)
    for i, skel in enumerate(episode.skeleton_pool):
        trajectory = reconstruct_trajectory(
            episode.initial_abstract_state,
            skel.operator_seq,
            verify_preconditions=False,
        )
        out[i] = sum(float(h_func(s)) for s in trajectory)
    return out


def _build_rt2d_domain_gen(heuristic_name: str) -> Any:
    """Build a fresh ``RelationalHeuristicSearchAbstractPlanGenerator`` over RT2D.

    Heavy import lives here so non-RT2D callers don't pay it. The PDDL domain
    inside the generator is RT2D-fixed (``ALL_TYPES`` / ``ALL_PREDICATES`` /
    ``ALL_OPERATORS``) and is reused across every episode.
    """
    # pylint: disable=import-outside-toplevel
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )

    from alphatamp.approaches.spectre.envs.routedtransport2d.operators import (
        ALL_OPERATORS,
        ALL_PREDICATES,
        ALL_TYPES,
    )

    return RelationalHeuristicSearchAbstractPlanGenerator(
        ALL_TYPES,
        ALL_PREDICATES,
        ALL_OPERATORS,
        heuristic_name=heuristic_name,
        seed=0,
    )


class HeuristicPrior(BasePrior):
    """Π = per-episode z-score of negated FF trajectory cost (RT2D-only).

    For each problem (lazily on first ``score`` call): reconstruct each
    skeleton's STRIPS trajectory, sum the pyperplan FF heuristic along it,
    then z-score the *negated* sums within the episode so:

    - higher π ⟹ FF heuristic prefers this skeleton (lower trajectory cost)
    - π = 0 ⟹ episode-mean (i.e. "no opinion"), which matches the
      semantics of ``Scorer.prior_dropout`` zeroing the input.

    All K scores for a problem are computed on first touch and cached on
    ``problem_id``. FF h-values are invariant under bijective object
    renaming, so augmented vs. deterministic canonicalizations of the same
    problem produce identical π values — the cache is therefore correct
    regardless of which canonicalization arrives first.

    Degenerate pools (every skeleton ties on FF score) fall back to all
    zeros, which the model then sees as uniform "no opinion" rather than
    NaN-poisoned input.
    """

    def __init__(self, heuristic_name: str = "hff") -> None:
        self._heuristic_name = heuristic_name
        self._domain_gen = _build_rt2d_domain_gen(heuristic_name)
        self._cache: dict[int, np.ndarray] = {}

    def score(
        self,
        problem_id: int,
        skeleton_idx: int,
        skeleton: "SkeletonRecord",
        episode: "EpisodeRecord",
    ) -> float:
        del skeleton  # only the (problem_id, skeleton_idx) pair is load-bearing
        cached = self._cache.get(problem_id)
        if cached is None:
            cached = self._compute_episode_priors(episode)
            self._cache[problem_id] = cached
        return float(cached[skeleton_idx])

    def _compute_episode_priors(self, episode: "EpisodeRecord") -> np.ndarray:
        raw = ff_trajectory_scores(self._domain_gen, episode)
        neg = -raw
        std = float(neg.std())
        if std < 1e-8:
            return np.zeros_like(neg)
        return ((neg - float(neg.mean())) / std).astype(np.float32)

    # Diagnostic surface for tests / instrumentation.
    @property
    def num_cached_episodes(self) -> int:
        """How many problems have had their priors computed and cached so far."""
        return len(self._cache)


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
    if prior_type == "heuristic":
        return HeuristicPrior()
    raise ValueError(
        f"Unknown prior_type={prior_type!r}; expected 'zero' or 'heuristic'"
    )
