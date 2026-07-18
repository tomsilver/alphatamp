"""Tests for ``spectre.priors.BasePrior`` and ``ZeroPrior``."""

from __future__ import annotations

import pytest
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre.priors import BasePrior, ZeroPrior


def test_zero_prior_returns_zero() -> None:
    """``ZeroPrior.score`` returns exactly ``0.0`` for every skeleton."""
    ep = build_toy_episode()
    prior = ZeroPrior()
    for i, skel in enumerate(ep.skeleton_pool):
        assert prior.score(ep.provenance.problem_id, i, skel, ep) == 0.0


def test_base_prior_is_abstract() -> None:
    """``BasePrior`` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        BasePrior()  # type: ignore[abstract]
