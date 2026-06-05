"""Tests for ``eda.heuristic_search_baseline``.

Exercises the full flow on a real RT2D episode: collect once with the
deterministic closed-form generator, canonicalize, then re-rank via the
FF-heuristic trajectory scorer. We check determinism, alignment with the
existing ``BaselineResult`` schema, and that the produced order actually
differs from default-order on at least one episode.
"""

# Pytest injects fixtures by parameter name, so test params legitimately
# shadow the module-scoped ``split`` fixture.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.collect import collect_episode
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.eda import (
    BaselineResult,
    LoadedSplit,
    _skeleton_key,
    default_order_baseline,
    heuristic_search_baseline,
)


@pytest.fixture(scope="module")
def split() -> LoadedSplit:
    """Module-shared tiny RT2D split (collection is the slow part, not search)."""
    return _rt2d_test_split(num_problems=2, k_max=12)


def _rt2d_test_split(num_problems: int = 2, k_max: int = 12) -> LoadedSplit:
    """Collect a tiny RT2D split via the closed-form generator (no I/O)."""
    cfg = CollectionConfig(
        env_id="kinder/RoutedTransport2D-n3-v0",
        env_variant="routedtransport2d_n3_v1",
        model_name="routedtransport2d",
        model_kwargs={"num_items": 3, "variant": "v1"},
        split="test",
        num_problems=num_problems,
        problem_seed_start=0,
        problem_seed_end=num_problems,
        K_max=k_max,
        # ThreeGateRefiner is closed-form; these budgets only need to be
        # large enough to cover the trivial latency of refinement.
        abstract_plan_timeout_s=1.0,
        refinement_timeout_s=0.1,
    )
    episodes = []
    skeleton_keys = []
    pool_max = 0
    for pid in range(num_problems):
        ep = canonicalize_episode(collect_episode(cfg, problem_id=pid), rng=None)
        episodes.append(ep)
        skeleton_keys.append([_skeleton_key(s) for s in ep.skeleton_pool])
        pool_max = max(pool_max, len(ep.skeleton_pool))
    return LoadedSplit(episodes=episodes, skeleton_keys=skeleton_keys, k_max=pool_max)


def test_returns_baseline_result_aligned_with_default_order(split: LoadedSplit) -> None:
    """Output is a ``BaselineResult`` aligned 1:1 with B2_default_order."""
    b2_lex = default_order_baseline(split, attempt_budget=20)
    b2_hs = heuristic_search_baseline(split, attempt_budget=20, seed=0)

    assert isinstance(b2_hs, BaselineResult)
    assert b2_hs.name == "B2_heuristic_search"
    # Trainable filter is applied identically; problem_ids must match.
    assert (b2_hs.problem_ids == b2_lex.problem_ids).all()
    assert len(b2_hs.attempts) == len(b2_lex.attempts)
    # Wall-clock is also drawn from outcomes only (no abstract-search charge).
    assert (b2_hs.wall_clock >= 0).all()


def test_deterministic_under_fixed_seed(split: LoadedSplit) -> None:
    """Same seed → identical attempts/wall_clock arrays."""
    a = heuristic_search_baseline(split, attempt_budget=20, seed=0)
    b = heuristic_search_baseline(split, attempt_budget=20, seed=0)
    assert (a.attempts == b.attempts).all()
    assert (a.wall_clock == b.wall_clock).all()


def test_reorders_relative_to_default_order(split: LoadedSplit) -> None:
    """FF-score order must differ from lex order on at least one episode."""
    b2_lex = default_order_baseline(split, attempt_budget=20)
    b2_hs = heuristic_search_baseline(split, attempt_budget=20, seed=0)
    # Either the attempts or the wall-clocks must change; if both stay
    # bit-identical we are not exercising a real reorder.
    assert not (
        (b2_hs.attempts == b2_lex.attempts).all()
        and (b2_hs.wall_clock == b2_lex.wall_clock).all()
    )


def test_abstract_plan_timeout_is_a_no_op(split: LoadedSplit) -> None:
    """Legacy ``abstract_plan_timeout_s`` parameter is accepted and has no effect.

    The previous implementation ran A* with a wall-clock budget; the scorer is
    deterministic so the parameter is now a no-op kept only so existing call-sites do
    not break.
    """
    a = heuristic_search_baseline(split, attempt_budget=20, seed=0)
    b = heuristic_search_baseline(
        split, attempt_budget=20, seed=0, abstract_plan_timeout_s=0.0
    )
    c = heuristic_search_baseline(
        split, attempt_budget=20, seed=0, abstract_plan_timeout_s=999.0
    )
    assert (a.attempts == b.attempts).all()
    assert (a.attempts == c.attempts).all()
