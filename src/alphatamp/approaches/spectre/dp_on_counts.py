"""Receding-horizon expectimax search for the DP-on-counts baseline (B6).

B6 is an evaluation **baseline** for SPECTRE (it is not the candidate method):
a dynamic-programming / lookahead skeleton-selection policy that reuses B4's
(Adaptive Historical) count estimator as its ``q``-model and adds ``h−1`` levels
of lookahead over the cost-to-first-success Bellman recursion.

This module is deliberately env-free and probability-only: it operates on pool
indices plus injected model fields so it can be unit-tested with hand-built
numbers. ``eda.dp_on_counts_baseline`` builds the model from a fitted
``_AdaptiveStats`` and drives the per-episode rollout.

Depth indexing (base-policy convention, see ``docs/decisions.md``):

* ``h = 1`` — no lookahead; the base greedy policy ``π(F) = argmin index(σ|F)``.
  For ``objective="attempts"`` ``index`` ranks by B4's success score, so ``h=1``
  reproduces B4 **exactly** (including its pool-index tie-break).
* ``h ≥ 2`` — ``π(F) = argmin_σ [c(σ) + q(σ|F) · W_{h−2}(F ∪ {σ})]`` with the
  expectimax value iteration ::

      W_0(F) = G(F)                                          # re-conditioning leaf
      W_ℓ(F) = min_{σ∈R} [c(σ) + q(σ|F) · W_{ℓ−1}(F ∪ {σ})]

The leaf ``G`` is the **true re-conditioning value** ``V^base`` of the ``h=1``
base policy — a stationary-greedy rollout that re-selects ``σ*`` and re-evaluates
``q`` at every step, *not* a frozen ``Σ_k c·Π q``. Using ``V^base`` makes the
modeled-value monotonicity ``W_0 ≥ W_1 ≥ W_2`` a guarantee via policy
improvement, and ordering the leaf by the same ``index`` the base policy uses
keeps leaf-base ≡ ``h=1``-base exactly.

**Scoring context.** The Naive-Bayes ``S_succ``/``S_fail`` log-scores extend
additively as failures accumulate. When a model supplies the incremental
primitives (``log_succ``/``log_fail``/``delta``), the search threads a
:class:`_Ctx` that extends those per-candidate scores by **one term per failure
edge** (``O(K)``) instead of recomputing the ``Σ_{k'∈F}`` at every node — turning
the ``O(K³)`` leaf into ``O(K²)``. Models without the primitives (synthetic test
fixtures) fall back to the ``q_of``/``score_of`` recompute closures, so behaviour
is identical; ``eda`` verifies the two paths agree bitwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

Objective = str  # "attempts" | "time"


@dataclass(frozen=True)
class DPModel:
    """Injected model the search operates on (env-free).

    - ``key_of[idx]``: the canonical key for pool index ``idx`` (appended to the
      failed multiset when ``idx`` is hypothesised to fail).
    - ``q_of(idx, failed)``: calibrated ``P(idx fails | failed) ∈ (0, 1)``.
    - ``score_of(idx, failed)``: B4 success log-score ``S_succ`` (only required
      for ``objective="attempts"``; may be ``None`` otherwise).
    - ``c_of(idx)``: per-skeleton cost (``≡1`` for attempts; mean refine time for
      time).
    - ``objective``: ``"attempts"`` or ``"time"``.

    Optional **incremental** primitives — when all three are present the search
    avoids the per-node ``Σ_{k'∈F}`` recompute:

    - ``log_succ[idx]``: ``log p̂(key_of[idx])`` (empty-``F`` success log-score).
    - ``log_fail[idx]``: ``log(1 − p̂(key_of[idx]))`` (empty-``F`` fail log-score).
    - ``delta(idx, k')``: ``(Δ S_succ, Δ S_fail)`` contributed by one failure
      ``k'``, or ``None`` for "no update" (unseen pair). Equals
      ``(log[p̂(k|k')/p̂(k)], log[(1−p̂(k|k'))/(1−p̂(k))])``.

    ``failed`` is the ordered tuple of already-failed canonical keys; the
    closures treat it as a multiset (the Naive-Bayes sum is order-independent),
    while the incremental path sums ``delta`` in failure-insertion order so it
    matches B4 bitwise.
    """

    key_of: Sequence[Hashable]
    q_of: Callable[[int, tuple[Hashable, ...]], float]
    c_of: Callable[[int], float]
    objective: Objective
    score_of: Callable[[int, tuple[Hashable, ...]], float] | None = None
    log_succ: Sequence[float] | None = None
    log_fail: Sequence[float] | None = None
    delta: Callable[[int, Hashable], tuple[float, float] | None] | None = None

    @property
    def incremental(self) -> bool:
        """Whether the incremental scoring primitives are available."""
        return self.delta is not None


@dataclass
class SearchStats:
    """Diagnostics for tests: backup-node expansions and the root value."""

    expansions: int = 0  # distinct cache-miss ``_W`` evaluations at level ≥ 1
    child_evals: int = 0  # total ``c + q·W`` child evaluations (root + internal)
    root_value: float = math.nan  # W_{h−1}(F) of the chosen decision (h ≥ 2)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class _Ctx:
    """Scoring context for a fixed failed set ``F``.

    Provides ``index(idx)`` (lower = better, for ``argmin``) and ``q(idx)`` for
    candidates, and ``extend(k')`` → a child context for ``F ∪ {k'}``. Two
    backends: an **incremental** one carrying per-candidate ``ss``/``sf`` arrays
    (``O(K)`` per ``extend``), and a **closure** one that recomputes via the
    model's ``q_of``/``score_of`` (used by synthetic test fixtures).
    """

    __slots__ = ("model", "failed", "ss", "sf")

    def __init__(
        self,
        model: DPModel,
        failed: tuple[Hashable, ...],
        ss: list[float] | None,
        sf: list[float] | None,
    ) -> None:
        self.model = model
        self.failed = failed
        self.ss = ss  # None ⇒ closure backend
        self.sf = sf

    @classmethod
    def base(cls, model: DPModel, failed: tuple[Hashable, ...]) -> _Ctx:
        """Build the context for ``failed`` from scratch."""
        if not model.incremental:
            return cls(model, failed, None, None)
        assert model.log_succ is not None and model.log_fail is not None
        ss = list(model.log_succ)
        sf = list(model.log_fail)
        assert model.delta is not None
        for k_prime in failed:  # insertion order ⇒ bitwise match with B4
            for i in range(len(ss)):
                d = model.delta(i, k_prime)
                if d is not None:
                    ss[i] += d[0]
                    sf[i] += d[1]
        return cls(model, failed, ss, sf)

    def extend(self, k_prime: Hashable) -> _Ctx:
        """Child context for ``F ∪ {k'}`` (``O(K)`` in the incremental backend)."""
        if self.ss is None:  # closure backend
            return _Ctx(self.model, self.failed + (k_prime,), None, None)
        assert self.sf is not None and self.model.delta is not None
        ss = self.ss[:]
        sf = self.sf[:]
        for i in range(len(ss)):
            d = self.model.delta(i, k_prime)
            if d is not None:
                ss[i] += d[0]
                sf[i] += d[1]
        return _Ctx(self.model, self.failed + (k_prime,), ss, sf)

    def index(self, idx: int) -> float:
        """Base-policy ordering key (lower is better).

        ``attempts`` → ``−S_succ`` (success score desc ≡ B4 order).
        ``time``     → ``−(1−q)/c`` (success-prob-per-cost desc, Smith's rule).
        """
        if self.ss is None:  # closure backend
            if self.model.objective == "attempts":
                assert self.model.score_of is not None
                return -self.model.score_of(idx, self.failed)
            q = self.model.q_of(idx, self.failed)
            return -((1.0 - q) / self.model.c_of(idx))
        if self.model.objective == "attempts":
            return -self.ss[idx]
        assert self.sf is not None
        q = _sigmoid(self.sf[idx] - self.ss[idx])
        return -((1.0 - q) / self.model.c_of(idx))

    def q(self, idx: int) -> float:
        """Calibrated ``P(idx fails | F)``."""
        if self.ss is None:  # closure backend
            return self.model.q_of(idx, self.failed)
        assert self.sf is not None
        return _sigmoid(self.sf[idx] - self.ss[idx])


def _greedy_pick(remaining: frozenset[int], ctx: _Ctx) -> int:
    best_idx = -1
    best_key: tuple[float, int] | None = None
    for idx in remaining:
        cand = (ctx.index(idx), idx)
        if best_key is None or cand < best_key:
            best_key = cand
            best_idx = idx
    if best_idx < 0:
        raise ValueError("greedy_pick called on an empty remaining set")
    return best_idx


def greedy_pick(
    model: DPModel, remaining: frozenset[int], failed: tuple[Hashable, ...]
) -> int:
    """Base-policy pick: ``argmin index(σ|F)``, ties broken by smaller index."""
    return _greedy_pick(remaining, _Ctx.base(model, failed))


def _leaf(
    model: DPModel,
    remaining: frozenset[int],
    ctx: _Ctx,
    memo: dict[tuple[frozenset[int], int], float],
) -> float:
    """Memoized re-conditioning greedy rollout; see :func:`leaf_value`.

    Shares ``memo`` (keyed ``(remaining, 0)``) with :func:`_backup` so the leaf
    ``W_0`` is computed once per distinct remaining set across the whole search.
    ``remaining`` determines the failed multiset within a call, so keying on it
    alone is sound.
    """
    if not remaining:
        return 0.0
    cache_key = (remaining, 0)
    cached = memo.get(cache_key)
    if cached is not None:
        return cached
    sigma = _greedy_pick(remaining, ctx)
    cost = model.c_of(sigma)
    q = ctx.q(sigma)
    rest = remaining - frozenset({sigma})
    val = cost + q * _leaf(model, rest, ctx.extend(model.key_of[sigma]), memo)
    memo[cache_key] = val
    return val


def leaf_value(
    model: DPModel, remaining: frozenset[int], failed: tuple[Hashable, ...]
) -> float:
    """``G(F) = V^base``: re-conditioning stationary-greedy rollout value.

    At each step pick ``σ* = argmin index(·|F)`` (re-evaluated), pay ``c(σ*)``,
    and with probability ``q(σ*|F)`` (re-conditioned) continue from ``F∪{σ*}``.
    This is exactly the modeled cost-to-first-success of the base policy.
    """
    return _leaf(model, frozenset(remaining), _Ctx.base(model, failed), {})


def _top_m(remaining: frozenset[int], ctx: _Ctx, m: int | None) -> list[int]:
    """The ``m`` candidates of ``remaining`` with the best greedy index under the
    context (``S_succ`` desc for attempts, ``(1−q)/c`` desc for time), ties broken by
    smaller pool index.

    ``m is None`` or ``m ≥ |remaining|`` ⇒ no pruning. Used only inside the
    lookahead backup ``_backup`` (ℓ ≥ 1); the root argmin in :func:`select` and
    the leaf walk in :func:`_leaf` never prune.
    """
    if m is None or m >= len(remaining):
        return list(remaining)
    return sorted(remaining, key=lambda idx: (ctx.index(idx), idx))[:m]


def _backup(
    model: DPModel,
    remaining: frozenset[int],
    ctx: _Ctx,
    level: int,
    memo: dict[tuple[frozenset[int], int], float],
    stats: SearchStats | None,
    m: int | None,
) -> float:
    """Value-iteration value ``W_level(F)`` with a call-scoped memo.

    ``remaining`` uniquely determines the failed multiset within one call, so the
    memo keys on ``(remaining, level)`` only — but it MUST stay local to one
    ``select``/``modeled_value`` call: ``W`` depends on the episode's pool.

    ``m`` is the top-m lookahead pruning width: at each internal decision node the
    ``min`` ranges only over the ``m`` best candidates (see :func:`_top_m`).
    ``m=None`` ⇒ exact (unpruned) value iteration.
    """
    if not remaining:
        return 0.0
    if level == 0:
        return _leaf(model, remaining, ctx, memo)
    cache_key = (remaining, level)
    cached = memo.get(cache_key)
    if cached is not None:
        return cached
    if stats is not None:
        stats.expansions += 1
    best = math.inf
    for idx in _top_m(remaining, ctx, m):
        cost = model.c_of(idx)
        q = ctx.q(idx)
        child = _backup(
            model,
            remaining - frozenset({idx}),
            ctx.extend(model.key_of[idx]),
            level - 1,
            memo,
            stats,
            m,
        )
        if stats is not None:
            stats.child_evals += 1
        best = min(best, cost + q * child)
    memo[cache_key] = best
    return best


def modeled_value(
    model: DPModel,
    remaining: frozenset[int],
    failed: tuple[Hashable, ...],
    level: int,
    stats: SearchStats | None = None,
    m: int | None = None,
) -> float:
    """Public ``W_level(F)`` (fresh memo). ``level=0`` is the leaf ``V^base``.

    Default ``m=None`` is exact (unpruned); non-increasing in ``level`` by policy
    improvement — the property the monotonicity test asserts on the *modeled*
    value. (Top-m pruning yields an upper bound, so the guarantee is stated for
    the exact value.)
    """
    memo: dict[tuple[frozenset[int], int], float] = {}
    return _backup(
        model, frozenset(remaining), _Ctx.base(model, failed), level, memo, stats, m
    )


def select(
    model: DPModel,
    remaining: frozenset[int] | set[int],
    failed: tuple[Hashable, ...],
    depth: int,
    stats: SearchStats | None = None,
    m: int | None = None,
) -> int:
    """Return the next pool index under the depth-``h`` policy.

    ``h=1`` → base greedy (``= B4`` for attempts). ``h≥2`` →
    ``argmin_σ [c(σ) + q(σ|F)·W_{h−2}(F∪{σ})]``, ties broken by smaller index.

    ``m`` is the top-m lookahead pruning width applied **only** inside the backup
    ``W_{h−2}`` (see :func:`_backup`/:func:`_top_m`): the root argmin below ranges
    over the **full** remaining set, so pruning never restricts which skeleton
    may actually be attempted — only how the lookahead expands. ``m=None`` ⇒
    exact, per-decision ``O(K^{h−1}·K²)``; pruned ⇒ ``O(m^{h−1}·K²)``. ``h=1``
    ignores ``m`` (no expansion ⇒ B4 identity holds).
    """
    rem = frozenset(remaining)
    if not rem:
        raise ValueError("select called on an empty remaining set")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if m is not None and m < 1:
        raise ValueError(f"m must be >= 1 or None, got {m}")
    ctx = _Ctx.base(model, failed)
    if depth == 1:
        return _greedy_pick(rem, ctx)
    memo: dict[tuple[frozenset[int], int], float] = {}
    best_val = math.inf
    best_idx = -1
    for idx in sorted(rem):  # ROOT: full remaining (never pruned), ties by index
        cost = model.c_of(idx)
        q = ctx.q(idx)
        child = _backup(
            model,
            rem - frozenset({idx}),
            ctx.extend(model.key_of[idx]),
            depth - 2,
            memo,
            stats,
            m,
        )
        if stats is not None:
            stats.child_evals += 1
        val = cost + q * child
        if val < best_val:
            best_val = val
            best_idx = idx
    if stats is not None:
        stats.root_value = best_val
    return best_idx
