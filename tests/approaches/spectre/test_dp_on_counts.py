"""Unit tests for the DP-on-counts baseline (B6) and its search module.

Covers the patched correctness requirements:

- ``h=1, attempts`` reproduces Adaptive Historical (B4) EXACTLY (arrays +
  per-step selection order).
- ``h=1, time`` is the cost-weighted greedy ``(1−q)/c`` order — checked against
  a direct index computation, and shown to differ from B4.
- The leaf ``G`` is the re-conditioning ``V^base`` rollout (NOT a frozen sum).
- Modeled-value monotonicity ``W_0 ≥ W_1 ≥ W_2`` holds, including a
  positive-co-failure-correlation instance (the case a frozen leaf breaks).
- Per-decision node expansion is ``O(K^{h−1})`` and ``≪ 2^K``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Hashable, Sequence

import numpy as np
import pytest
from _fixtures import write_toy_split

from alphatamp.approaches.spectre import dp_on_counts
from alphatamp.approaches.spectre.eda import (
    SkeletonKey,
    _adaptive_score,
    _build_dp_model,
    _fit_adaptive,
    adaptive_historical_baseline,
    dp_on_counts_baseline,
    load_split_episodes,
    solvability_at_cap,
)

POOL_CAP = 30  # RT2D-n3 candidate-pool cap (uncensored eval budget)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_dp_order(model: dp_on_counts.DPModel) -> list[int]:
    """Play the base greedy policy to completion, recording the pick order."""
    remaining = frozenset(range(len(model.key_of)))
    failed: tuple[Hashable, ...] = ()
    order: list[int] = []
    while remaining:
        idx = dp_on_counts.greedy_pick(model, remaining, failed)
        order.append(idx)
        failed = failed + (model.key_of[idx],)
        remaining = remaining - frozenset({idx})
    return order


def _b4_full_order(stats, keys: Sequence[SkeletonKey]) -> list[int]:
    """B4's greedy ranking played to completion (argmax S_succ, idx tie-break)."""
    remaining = set(range(len(keys)))
    failed: list[SkeletonKey] = []
    order: list[int] = []
    while remaining:
        best_score = -np.inf
        best_idx = min(remaining)
        for idx in remaining:
            score = _adaptive_score(stats, keys[idx], failed)
            if (score > best_score) or (score == best_score and idx < best_idx):
                best_score = score
                best_idx = idx
        order.append(best_idx)
        failed.append(keys[best_idx])
        remaining.remove(best_idx)
    return order


def _const_model(
    qs: Sequence[float],
    cs: Sequence[float],
    scores: Sequence[float] | None,
    objective: str,
) -> dp_on_counts.DPModel:
    """A failed-independent synthetic model over ``len(qs)`` pool indices."""
    keys: list[Hashable] = [(f"k{i}",) for i in range(len(qs))]

    def q_of(idx: int, _failed: tuple[Hashable, ...]) -> float:
        return qs[idx]

    def c_of(idx: int) -> float:
        return cs[idx]

    score_of: Callable[[int, tuple[Hashable, ...]], float] | None = None
    if scores is not None:
        scores_list = list(scores)

        def _score(idx: int, _failed: tuple[Hashable, ...]) -> float:
            return scores_list[idx]

        score_of = _score

    return dp_on_counts.DPModel(
        key_of=keys, q_of=q_of, c_of=c_of, objective=objective, score_of=score_of
    )


# ---------------------------------------------------------------------------
# h=1, attempts ≡ B4
# ---------------------------------------------------------------------------


def _toy_train_test(tmp_path: Path):
    train_outcomes: list[tuple[str, ...]] = []
    for _pattern, _count in [
        (("success", "fail", "fail"), 6),
        (("fail", "success", "fail"), 4),
        (("fail", "fail", "success"), 3),
        (("success", "success", "fail"), 2),
        (("fail", "success", "success"), 2),
    ]:
        train_outcomes.extend([_pattern] * _count)
    test_outcomes: list[tuple[str, ...]] = [
        ("fail", "success", "fail"),
        ("fail", "fail", "success"),
        ("success", "fail", "fail"),
        ("fail", "success", "success"),
    ]
    write_toy_split(tmp_path / "train", train_outcomes)
    write_toy_split(tmp_path / "test", test_outcomes)
    train = load_split_episodes(tmp_path / "train")
    test = load_split_episodes(tmp_path / "test")
    return train, test


def test_h1_attempts_arrays_match_b4_exactly(tmp_path: Path) -> None:
    """B6 ``h=1, attempts`` returns the same per-episode arrays as B4."""
    train, test = _toy_train_test(tmp_path)
    b4 = adaptive_historical_baseline(train, test, attempt_budget=20)
    b6 = dp_on_counts_baseline(
        train, test, attempt_budget=20, depth=1, objective="attempts"
    )
    assert np.array_equal(b6.attempts, b4.attempts)
    assert np.array_equal(b6.wall_clock, b4.wall_clock)
    assert np.array_equal(b6.censored, b4.censored)
    assert np.array_equal(b6.problem_ids, b4.problem_ids)


def test_solvability_at_cap_monotone_and_endpoints(tmp_path: Path) -> None:
    """solvability_at_cap is non-decreasing in k; endpoint = fraction-any-success."""
    write_toy_split(
        tmp_path / "s",
        [
            ("success", "fail", "fail"),  # solvable at k >= 1
            ("fail", "fail", "success"),  # solvable at k >= 3
            ("fail", "fail", "fail"),  # never solvable (no success)
        ],
    )
    split = load_split_episodes(tmp_path / "s")
    sol = solvability_at_cap(split, k_max=3)
    assert sol.shape == (3,)
    assert np.all(np.diff(sol) >= 0)  # non-decreasing in k
    assert sol[0] == pytest.approx(1 / 3)  # only episode 0 within first 1
    assert sol[2] == pytest.approx(2 / 3)  # episodes 0 and 1 within first 3
    frac_any = sum(1 for ep in split.episodes if ep.summary.num_success >= 1) / len(
        split.episodes
    )
    assert sol[2] == pytest.approx(frac_any)


def test_score_cache_keyed_in_failure_insertion_order(tmp_path: Path) -> None:
    """The NB score cache keys on ``failed`` in insertion order, not sorted.

    B4 sums the Naive-Bayes log-terms in failure-insertion order; the log-sum is
    mathematically order-independent but not bitwise so. Keying (and summing) in
    insertion order is what makes B6 ``h=1`` reproduce B4 *exactly* on real
    pools. A non-sorted ``failed`` whose key lands in the cache verbatim proves
    no reordering happens.
    """
    train, test = _toy_train_test(tmp_path)
    stats = _fit_adaptive(train)
    keys = test.skeleton_keys[0]
    cache: dict = {}
    model = _build_dp_model(stats, keys, "attempts", None, cache, {})
    failed = (keys[2], keys[0])  # deliberately NOT in sorted order
    assert list(failed) != sorted(failed)  # guard: the case is actually unsorted
    model.score_of(1, failed)  # type: ignore[misc]
    assert (keys[1], failed) in cache
    assert (keys[1], tuple(sorted(failed))) not in cache


def test_incremental_scores_equal_recompute(tmp_path: Path) -> None:
    """The incremental score path equals the closure-recompute path bitwise.

    The ``eda`` model carries both the incremental NB primitives (used by the
    search) and the ``q_of``/``score_of`` recompute closures. Stripping the
    primitives forces the closure backend; selections and modeled values must be
    identical for h ∈ {2,3,4} and several failure histories.
    """
    train, test = _toy_train_test(tmp_path)
    stats = _fit_adaptive(train)
    for ep_idx in range(len(test.episodes)):
        keys = test.skeleton_keys[ep_idx]
        inc = _build_dp_model(stats, keys, "attempts", None, {}, {}, {})
        closure = replace(inc, log_succ=None, log_fail=None, delta=None)
        assert inc.incremental and not closure.incremental
        rem = frozenset(range(len(keys)))
        for failed in [(), (keys[0],), (keys[2], keys[0])]:
            for depth in (2, 3, 4):
                assert dp_on_counts.select(
                    inc, rem, failed, depth
                ) == dp_on_counts.select(closure, rem, failed, depth)
            for level in (0, 1, 2):
                assert dp_on_counts.modeled_value(
                    inc, rem, failed, level
                ) == dp_on_counts.modeled_value(closure, rem, failed, level)


def test_h1_attempts_selection_order_matches_b4(tmp_path: Path) -> None:
    """Per-episode full selection order is identical to B4 (incl.

    tie-break).
    """
    train, test = _toy_train_test(tmp_path)
    stats = _fit_adaptive(train)
    for ep_idx in range(len(test.episodes)):
        keys = test.skeleton_keys[ep_idx]
        model = _build_dp_model(stats, keys, "attempts", None, {}, {})
        assert _full_dp_order(model) == _b4_full_order(stats, keys)


# ---------------------------------------------------------------------------
# h=1, time = cost-weighted greedy, ≠ B4
# ---------------------------------------------------------------------------


def test_h1_time_is_cost_weighted_greedy_not_attempts() -> None:
    """Time order = ``(1−q)/c`` desc; differs from attempts (S_succ) order."""
    qs = [0.5, 0.4, 0.3]
    cs = [1.0, 10.0, 1.0]
    scores = [1.0, 3.0, 2.0]  # attempts ranks idx1 > idx2 > idx0
    time_model = _const_model(qs, cs, None, "time")
    att_model = _const_model(qs, cs, scores, "attempts")

    # Direct index computation: (1−q)/c descending, idx tie-break.
    direct = sorted(range(3), key=lambda i: (-((1.0 - qs[i]) / cs[i]), i))
    assert direct == [2, 0, 1]
    assert _full_dp_order(time_model) == direct
    # Attempts (= B4) order is success-score descending: a different order.
    assert _full_dp_order(att_model) == [1, 2, 0]
    assert _full_dp_order(time_model) != _full_dp_order(att_model)


# ---------------------------------------------------------------------------
# Leaf G = re-conditioning V^base (NOT the frozen Σ c·Π q)
# ---------------------------------------------------------------------------


def _positive_correlation_model(objective: str = "attempts") -> dp_on_counts.DPModel:
    """3-skeleton model where a failure raises siblings' fail-probs.

    Order is always A(0), B(1), C(2) (failed-independent success scores). With a
    *frozen* leaf the value would be ``1 + 0.5(1 + 0.4·1) = 1.7``; with the
    re-conditioning rollout it is ``1 + 0.5(1 + 0.8·1) = 1.9``.
    """
    keys: list[Hashable] = [("A",), ("B",), ("C",)]

    def q_of(idx: int, failed: tuple[Hashable, ...]) -> float:
        a_failed = ("A",) in failed
        b_failed = ("B",) in failed
        if idx == 0:
            return 0.5
        if idx == 1:
            return 0.8 if a_failed else 0.4
        # idx == 2
        if a_failed and b_failed:
            return 0.95
        if a_failed:
            return 0.9
        return 0.3

    def c_of(_idx: int) -> float:
        return 1.0

    def score_of(idx: int, _failed: tuple[Hashable, ...]) -> float:
        return [3.0, 2.0, 1.0][idx]

    return dp_on_counts.DPModel(
        key_of=keys, q_of=q_of, c_of=c_of, objective=objective, score_of=score_of
    )


def test_leaf_value_is_reconditioning_vbase() -> None:
    """Leaf equals the re-conditioning greedy rollout, not the frozen sum."""
    model = _positive_correlation_model()
    remaining = frozenset({0, 1, 2})
    # Re-conditioning: 1 + 0.5·(1 + 0.8·(1 + 0.95·0)) = 1.9.
    assert dp_on_counts.leaf_value(model, remaining, ()) == pytest.approx(1.9)
    # The frozen Σ c·Π q value (using q at ∅) would be 1.7 — distinct.
    frozen = 1.0 + 0.5 * (1.0 + 0.4 * 1.0)
    assert frozen == pytest.approx(1.7)
    assert dp_on_counts.leaf_value(model, remaining, ()) != pytest.approx(frozen)


def test_leaf_value_equals_modeled_value_level_zero() -> None:
    """``modeled_value(level=0)`` is exactly the leaf ``V^base``."""
    model = _positive_correlation_model()
    remaining = frozenset({0, 1, 2})
    assert dp_on_counts.modeled_value(model, remaining, (), 0) == pytest.approx(
        dp_on_counts.leaf_value(model, remaining, ())
    )


# ---------------------------------------------------------------------------
# Modeled monotonicity W_0 ≥ W_1 ≥ W_2 (guarantee)
# ---------------------------------------------------------------------------


def test_modeled_value_monotone_under_positive_correlation() -> None:
    """Policy-improvement monotonicity holds on the frozen-leaf-breaking case."""
    model = _positive_correlation_model()
    remaining = frozenset({0, 1, 2})
    w0 = dp_on_counts.modeled_value(model, remaining, (), 0)
    w1 = dp_on_counts.modeled_value(model, remaining, (), 1)
    w2 = dp_on_counts.modeled_value(model, remaining, (), 2)
    assert w0 >= w1 - 1e-12
    assert w1 >= w2 - 1e-12


# ---------------------------------------------------------------------------
# Tractability: O(K^{h−1}), never 2^K
# ---------------------------------------------------------------------------


def test_node_expansion_is_polynomial_not_exponential() -> None:
    """Backup expansion is ``O(K^{h−1})`` and far below ``2^K``."""
    k = 10
    depth = 3
    assert depth <= 3  # h is small by construction
    assert k <= POOL_CAP  # K bounded by the pool cap
    rng = np.random.default_rng(0)
    qs = [float(x) for x in rng.uniform(0.2, 0.8, size=k)]
    scores = [float(x) for x in rng.uniform(0.0, 1.0, size=k)]
    model = _const_model(qs, [1.0] * k, scores, "attempts")
    stats = dp_on_counts.SearchStats()
    dp_on_counts.select(model, set(range(k)), (), depth, stats=stats)
    # Total child evaluations stay polynomial (root K + K·(K−1) memoized internal).
    assert stats.child_evals <= k ** (depth - 1) + k
    assert stats.child_evals < 2**k
    assert stats.expansions <= k ** (depth - 1)


def _random_model(
    k: int, seed: int, objective: str = "attempts"
) -> dp_on_counts.DPModel:
    rng = np.random.default_rng(seed)
    qs = [float(x) for x in rng.uniform(0.15, 0.85, size=k)]
    scores = [float(x) for x in rng.uniform(0.0, 1.0, size=k)]
    return _const_model(qs, [1.0] * k, scores, objective)


def test_pruning_m_ge_k_equals_unpruned() -> None:
    """Top-m pruning with m ≥ K is bitwise-identical to the unpruned search."""
    model = _random_model(k=6, seed=2)
    rem = frozenset(range(6))
    for depth in (2, 3, 4):
        unpruned = dp_on_counts.select(model, rem, (), depth, m=None)
        for m in (6, 7, 1000):
            assert dp_on_counts.select(model, rem, (), depth, m=m) == unpruned
        # Modeled value also matches at m ≥ K.
        assert dp_on_counts.modeled_value(
            model, rem, (), depth - 1, m=6
        ) == pytest.approx(dp_on_counts.modeled_value(model, rem, (), depth - 1))


def test_pruning_reduces_node_count() -> None:
    """Pruned expansion (m<K) evaluates strictly fewer backup children."""
    k, m = 10, 4
    model = _random_model(k=k, seed=3)
    rem = set(range(k))
    s_un = dp_on_counts.SearchStats()
    dp_on_counts.select(model, rem, (), 3, stats=s_un, m=None)
    s_pr = dp_on_counts.SearchStats()
    dp_on_counts.select(model, rem, (), 3, stats=s_pr, m=m)
    assert s_pr.child_evals < s_un.child_evals
    # Root K children, each a single level-1 node pruned to min(m, K-1) evals.
    assert s_pr.child_evals == k + k * min(m, k - 1)
    assert s_un.child_evals == k + k * (k - 1)


def test_pruning_does_not_touch_h1_or_root() -> None:
    """M never changes h=1 (no expansion) and the root stays full at h≥2."""
    model = _random_model(k=8, seed=4)
    rem = frozenset(range(8))
    # h=1 ignores m entirely.
    base = dp_on_counts.select(model, rem, (), 1)
    for m in (1, 2, None):
        assert dp_on_counts.select(model, rem, (), 1, m=m) == base
    # At h=2 there are no internal lookahead nodes, so any m == unpruned.
    h2 = dp_on_counts.select(model, rem, (), 2, m=None)
    assert dp_on_counts.select(model, rem, (), 2, m=1) == h2


def test_select_depth_one_equals_greedy_pick() -> None:
    """``select`` at ``depth=1`` is exactly the base greedy pick."""
    model = _positive_correlation_model()
    remaining = frozenset({0, 1, 2})
    assert dp_on_counts.select(model, remaining, (), 1) == dp_on_counts.greedy_pick(
        model, remaining, ()
    )
