"""Exploratory data analysis for SPECTRE training collections.

Implements the full diagnostic battery defined in ``SPECTRE_EDA_SPEC.md``:

- **Group 1 — episode sanity** (§3): pool-cap confirmation, cross-problem
  skeleton diversity (rarefaction + Jaccard), episode success rate,
  default-order budget exhaustion. Computed on the *training* split.
- **Disjointness diagnostic**: train↔test canonical-key overlap
  (not in the spec proper but required to interpret Group 2 in the
  disjoint-pool regime).
- **Group 2 — baselines** (§4): B1 random floor, B2 default order,
  B3 static-historical, B4 adaptive-historical (Naive-Bayes), B5 oracle.
  All five **evaluate on the test split**. B3/B4 fit ``p̂`` on train.
- **Group 3 — scalars** (§5): adaptive premium ``Δ = mean(B3) − mean(B4)``
  and headroom ``H = mean(B2) − mean(B5)`` with paired bootstrap CIs.
- **Pass bar** (§6): boolean verdict with an interpretive qualifier for
  the disjoint-pool regime (§5.1 caveat).

Design principles:

- All functions are pure; they take ``LoadedSplit`` objects and return
  plain dataclasses or numpy arrays. The notebook is a thin presentation
  layer.
- Canonical skeleton keys are computed once at load time via
  ``canonicalize.canonicalize_episode(ep, rng=None)`` so every downstream
  comparison uses the same renumbering convention (spec §2, SPEC §4.1.4).
- Baseline simulation respects the spec's **censoring** discipline (§2):
  a traversal hitting the attempt budget without success contributes
  ``T=21`` with wall-clock summed to the 20th attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.schema import EpisodeRecord

SkeletonKey = tuple[tuple[str, tuple[str, ...]], ...]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedSplit:
    """Canonicalized episodes plus per-skeleton keys for one split.

    - ``episodes[i]``: the ``EpisodeRecord`` *after* deterministic
      canonicalization (``rng=None``). Object names are ``"{type}_{idx}"``.
    - ``skeleton_keys[i][j]``: a hashable tuple key for
      ``episodes[i].skeleton_pool[j]`` derived from the renumbered
      operator sequence. Equal keys ≡ equivalent skeletons per SPEC §4.1.4.
    - ``k_max``: the largest pool size observed; a proxy for the
      collection-config cap (we don't read the YAML to avoid coupling).
    """

    episodes: list[EpisodeRecord]
    skeleton_keys: list[list[SkeletonKey]]
    k_max: int


def _skeleton_key(skel) -> SkeletonKey:
    """Hashable canonical key derived from a canonicalized skeleton.

    After :func:`canonicalize_episode`, operator arg names are deterministic
    typed-local-ids. The tuple of ``(op_name, (arg_name, …))`` pairs is the
    canonical equivalence class.
    """
    return tuple(
        (op.name, tuple(arg.name for arg in op.parameters)) for op in skel.operator_seq
    )


def load_split_episodes(split_dir: Path) -> LoadedSplit:
    """Load every episode under ``<split_dir>/episodes/``, canonicalize, key.

    Canonicalization is deterministic (``rng=None``) so every call produces
    the same keys. Errors in reconstruction propagate — malformed episode
    files should be investigated, not silently skipped.
    """
    episodes: list[EpisodeRecord] = []
    skeleton_keys: list[list[SkeletonKey]] = []
    k_max = 0
    for path in list_episodes(split_dir):
        raw = load_episode(path)
        canon = canonicalize_episode(raw, rng=None)
        episodes.append(canon)
        skeleton_keys.append([_skeleton_key(skel) for skel in canon.skeleton_pool])
        k_max = max(k_max, len(canon.skeleton_pool))
    return LoadedSplit(episodes=episodes, skeleton_keys=skeleton_keys, k_max=k_max)


# ---------------------------------------------------------------------------
# Group 1 — episode sanity
# ---------------------------------------------------------------------------


def pool_cap_fraction(split: LoadedSplit) -> float:
    """Fraction of episodes whose pool equals the observed max ``k_max`` (§3.1).

    The spec expects planners to saturate the config's ``K_max`` on nearly
    every problem. We compare against the empirical max rather than reading
    the config YAML, which is fine as long as some episode hits the cap.
    """
    if not split.episodes:
        return 0.0
    return float(
        np.mean([len(ep.skeleton_pool) >= split.k_max for ep in split.episodes])
    )


def count_unique_canonical_keys(split: LoadedSplit) -> tuple[int, int]:
    """Returns ``(U, N_slots)`` (§3.2).

    ``U`` = number of distinct canonical keys across all episodes.
    ``N_slots`` = Σ |S_e| (total skeleton occurrences).
    """
    seen: set[SkeletonKey] = set()
    n_slots = 0
    for keys in split.skeleton_keys:
        n_slots += len(keys)
        seen.update(keys)
    return len(seen), n_slots


def rarefaction_curve(
    split: LoadedSplit,
    num_shuffles: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Cumulative unique-canonical-key curve averaged over ``num_shuffles`` (§3.2).

    Returns a float array of length ``len(split.episodes)``; entry ``i`` is
    the expected number of distinct keys after processing the first ``i+1``
    episodes in a uniformly random order.
    """
    n = len(split.episodes)
    if n == 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(seed)
    per_episode_keysets = [set(keys) for keys in split.skeleton_keys]
    accum = np.zeros(n, dtype=float)
    for _ in range(num_shuffles):
        order = rng.permutation(n)
        seen: set[SkeletonKey] = set()
        for step, ep_idx in enumerate(order):
            seen.update(per_episode_keysets[ep_idx])
            accum[step] += len(seen)
    return accum / num_shuffles


def jaccard_pair_sample(
    split: LoadedSplit,
    num_pairs: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Sample-based distribution of pairwise pool Jaccard similarities (§3.2).

    Returns a float array of length ``min(num_pairs, C(n, 2))`` with values
    in ``[0, 1]``. Sampling is with replacement over distinct ordered pairs;
    pairs with identical episode indices are rejected.
    """
    n = len(split.episodes)
    if n < 2:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(seed)
    per_episode_keysets = [set(keys) for keys in split.skeleton_keys]
    results = np.empty(num_pairs, dtype=float)
    drawn = 0
    while drawn < num_pairs:
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        a, b = per_episode_keysets[i], per_episode_keysets[j]
        union = a | b
        results[drawn] = len(a & b) / len(union) if union else 0.0
        drawn += 1
    return results


def success_rate_distribution(
    split: LoadedSplit,
) -> tuple[float, np.ndarray]:
    """Returns ``(fraction_with_success, n_succ_over_K_array)`` (§3.3)."""
    if not split.episodes:
        return 0.0, np.zeros(0, dtype=float)
    has_success = np.array(
        [ep.summary.num_success >= 1 for ep in split.episodes], dtype=bool
    )
    n_succ_over_k = np.array(
        [
            ep.summary.num_success / max(ep.summary.num_skeletons, 1)
            for ep in split.episodes
        ],
        dtype=float,
    )
    return float(has_success.mean()), n_succ_over_k


def default_order_budget_exhaustion(
    split: LoadedSplit, attempt_budget: int = 20
) -> float:
    """Fraction of episodes with ``T_default_e > attempt_budget`` (§3.4).

    ``T_default_e = 1 + ep.summary.first_success_idx`` if a success exists,
    else ∞. An episode with no success counts as exhausted regardless of
    budget.
    """
    if not split.episodes:
        return 0.0
    counts = 0
    for ep in split.episodes:
        fsi = ep.summary.first_success_idx
        t_default = float("inf") if fsi is None else 1 + fsi
        if t_default > attempt_budget:
            counts += 1
    return counts / len(split.episodes)


# ---------------------------------------------------------------------------
# Disjointness diagnostics (train ↔ test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyOverlapReport:
    """Train↔test canonical-key overlap; interpretive context for B3/B4.

    See the plan's decision-tree interpretation:
    - ``test_keys_seen_fraction ≥ 0.8``: overlapping-pool regime.
    - ``≤ 0.1``: disjoint-pool regime — B3/B4 degenerate to default order,
      Δ≈0 is mechanical not informative.
    - otherwise: partial overlap, attenuated signal.
    """

    num_unique_train_keys: int
    num_unique_test_keys: int
    test_keys_seen_in_train: int
    test_keys_seen_fraction: float
    median_per_episode_seen_fraction: float
    pairwise_cooccurrence_density: float

    def regime(self) -> Literal["overlapping", "partial", "disjoint"]:
        """Coarse label driving interpretation of Δ downstream."""
        if self.test_keys_seen_fraction >= 0.8:
            return "overlapping"
        if self.test_keys_seen_fraction <= 0.1:
            return "disjoint"
        return "partial"


def train_eval_key_overlap(
    train: LoadedSplit, test_split: LoadedSplit
) -> KeyOverlapReport:
    """Compute train↔test canonical-key overlap statistics.

    ``pairwise_cooccurrence_density`` is the fraction of *test* ordered
    key-pairs ``(k, k')`` for which the same pair co-occurred in any
    training episode. Drives the expected usefulness of B4's pairwise
    table.
    """
    train_key_set: set[SkeletonKey] = set()
    for keys in train.skeleton_keys:
        train_key_set.update(keys)

    test_key_set: set[SkeletonKey] = set()
    per_ep_seen_fractions: list[float] = []
    for keys in test_split.skeleton_keys:
        test_key_set.update(keys)
        if keys:
            seen = sum(1 for k in keys if k in train_key_set)
            per_ep_seen_fractions.append(seen / len(keys))

    # Pairwise co-occurrence (sampled over test pairs to stay tractable).
    train_pairs: set[tuple[SkeletonKey, SkeletonKey]] = set()
    for keys in train.skeleton_keys:
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1 :]:
                train_pairs.add((k1, k2))
                train_pairs.add((k2, k1))

    test_pair_count = 0
    test_pair_seen = 0
    for keys in test_split.skeleton_keys:
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1 :]:
                test_pair_count += 2  # both directions
                if (k1, k2) in train_pairs:
                    test_pair_seen += 1
                if (k2, k1) in train_pairs:
                    test_pair_seen += 1

    if test_key_set:
        test_seen_frac = len(test_key_set & train_key_set) / len(test_key_set)
    else:
        test_seen_frac = 0.0

    return KeyOverlapReport(
        num_unique_train_keys=len(train_key_set),
        num_unique_test_keys=len(test_key_set),
        test_keys_seen_in_train=len(test_key_set & train_key_set),
        test_keys_seen_fraction=test_seen_frac,
        median_per_episode_seen_fraction=(
            float(np.median(per_ep_seen_fractions)) if per_ep_seen_fractions else 0.0
        ),
        pairwise_cooccurrence_density=(
            test_pair_seen / test_pair_count if test_pair_count else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Group 2 — baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Per-episode outcome arrays produced by a baseline simulator.

    ``attempts[i]`` is the number of refinement attempts until first success
    on trainable episode ``i`` (or ``attempt_budget + 1 == 21`` if censored).
    ``wall_clock[i]`` is the cumulative refinement time through that step.
    """

    name: str
    attempts: np.ndarray
    wall_clock: np.ndarray
    censored: np.ndarray
    problem_ids: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.attempts)
        assert len(self.wall_clock) == n
        assert len(self.censored) == n
        assert len(self.problem_ids) == n


def _trainable_episodes(split: LoadedSplit) -> list[int]:
    """Indices of episodes with at least one success (Group 2 subset, §2)."""
    return [i for i, ep in enumerate(split.episodes) if ep.summary.num_success >= 1]


def _simulate_traversal(
    ep: EpisodeRecord,
    order: Sequence[int],
    attempt_budget: int,
) -> tuple[int, float, bool]:
    """Walk ``ep.skeleton_pool`` in the given order; stop at first success.

    ``order`` is a permutation of pool indices. Returns
    ``(attempts, wall_clock, censored)`` with censoring semantics per §2
    of the EDA spec.
    """
    wall = 0.0
    for step, idx in enumerate(order, start=1):
        outcome = ep.outcomes[idx]
        wall += outcome.refinement_wall_clock_s
        if outcome.outcome == "success":
            return step, wall, False
        if step >= attempt_budget:
            return attempt_budget + 1, wall, True
    # Pool exhausted without success before reaching the budget — reach
    # here only if |pool| < attempt_budget. Treat as censored with
    # ``attempt_budget + 1`` so all censored values share the same T.
    return attempt_budget + 1, wall, True


def random_floor_baseline(
    test: LoadedSplit,
    attempt_budget: int = 20,
    mc_permutations: int = 100,
    seed: int = 0,
) -> BaselineResult:
    """B1 (§4.1).

    Attempts via closed form; wall-clock via Monte Carlo.
    """
    trainable = _trainable_episodes(test)
    rng = np.random.default_rng(seed)
    attempts = np.zeros(len(trainable), dtype=float)
    wall_clock = np.zeros(len(trainable), dtype=float)
    censored = np.zeros(len(trainable), dtype=bool)
    problem_ids = np.zeros(len(trainable), dtype=np.int64)
    for out_idx, ep_idx in enumerate(trainable):
        ep = test.episodes[ep_idx]
        k = ep.summary.num_skeletons
        n_succ = ep.summary.num_success
        # Closed-form expected attempts under uniform permutation (§4.1).
        attempts[out_idx] = (k + 1) / (n_succ + 1)
        mc_walls = np.zeros(mc_permutations)
        mc_cens = 0
        for mc in range(mc_permutations):
            order = rng.permutation(k)
            _, w, c = _simulate_traversal(ep, order.tolist(), attempt_budget)
            mc_walls[mc] = w
            mc_cens += int(c)
        wall_clock[out_idx] = float(mc_walls.mean())
        censored[out_idx] = (mc_cens / mc_permutations) >= 0.5
        problem_ids[out_idx] = ep.provenance.problem_id
    return BaselineResult(
        name="B1_random_floor",
        attempts=attempts,
        wall_clock=wall_clock,
        censored=censored,
        problem_ids=problem_ids,
    )


def default_order_baseline(
    test: LoadedSplit, attempt_budget: int = 20
) -> BaselineResult:
    """B2 (§4.2).

    Planner's canonical order, simulated with censoring.
    """
    trainable = _trainable_episodes(test)
    attempts = np.zeros(len(trainable), dtype=float)
    wall_clock = np.zeros(len(trainable), dtype=float)
    censored = np.zeros(len(trainable), dtype=bool)
    problem_ids = np.zeros(len(trainable), dtype=np.int64)
    for out_idx, ep_idx in enumerate(trainable):
        ep = test.episodes[ep_idx]
        a, w, c = _simulate_traversal(
            ep, list(range(len(ep.skeleton_pool))), attempt_budget
        )
        attempts[out_idx] = a
        wall_clock[out_idx] = w
        censored[out_idx] = c
        problem_ids[out_idx] = ep.provenance.problem_id
    return BaselineResult(
        name="B2_default_order",
        attempts=attempts,
        wall_clock=wall_clock,
        censored=censored,
        problem_ids=problem_ids,
    )


def oracle_ceiling(test: LoadedSplit, attempt_budget: int = 20) -> BaselineResult:
    """B5 (§4.5).

    Attempts=1 per success; wall-clock = min refine time.
    """
    del attempt_budget  # unused; oracle never censors (only trainable eps)
    trainable = _trainable_episodes(test)
    attempts = np.ones(len(trainable), dtype=float)
    wall_clock = np.zeros(len(trainable), dtype=float)
    censored = np.zeros(len(trainable), dtype=bool)
    problem_ids = np.zeros(len(trainable), dtype=np.int64)
    for out_idx, ep_idx in enumerate(trainable):
        ep = test.episodes[ep_idx]
        succ_times = [
            ep.outcomes[i].refinement_wall_clock_s for i in ep.success_indices()
        ]
        wall_clock[out_idx] = min(succ_times)
        problem_ids[out_idx] = ep.provenance.problem_id
    return BaselineResult(
        name="B5_oracle",
        attempts=attempts,
        wall_clock=wall_clock,
        censored=censored,
        problem_ids=problem_ids,
    )


# ---- Historical baselines (fit on train, evaluate on test) ----------------


@dataclass(frozen=True)
class _MarginalStats:
    """Laplace-smoothed per-key marginals fit on training data."""

    successes: dict[SkeletonKey, int] = field(default_factory=dict)
    appearances: dict[SkeletonKey, int] = field(default_factory=dict)

    def p_hat(self, key: SkeletonKey) -> float:
        """Laplace-smoothed estimate ``(successes+1)/(appearances+2)``."""
        s = self.successes.get(key, 0)
        a = self.appearances.get(key, 0)
        return (s + 1.0) / (a + 2.0)


@dataclass(frozen=True)
class SkeletonKeyStats:
    """Per-canonical-key training statistics, ranked for presentation."""

    key: SkeletonKey
    successes: int
    appearances: int
    p_hat: float


def format_skeleton_key(key: SkeletonKey) -> str:
    """Pretty-print a canonical skeleton key as a numbered ``op(args)`` list."""
    lines = []
    for step, (op_name, args) in enumerate(key, start=1):
        lines.append(f"  {step:2d}. {op_name}({', '.join(args)})")
    return "\n".join(lines)


def top_successful_skeleton_keys(
    train: LoadedSplit,
    n: int = 10,
    rank_by: Literal["successes", "p_hat"] = "successes",
) -> list[SkeletonKeyStats]:
    """Top-``n`` canonical skeleton keys by historical success on ``train``.

    ``rank_by="successes"`` sorts by raw success count (how often the plan
    actually worked in training). ``rank_by="p_hat"`` uses the Laplace-smoothed
    rate ``(successes+1)/(appearances+2)`` — the same score B3 uses — which
    down-weights rare keys that happened to succeed once. Ties are broken
    first by appearances (desc), then lexicographically by the key itself.
    """
    stats = _fit_marginals(train)
    rows = [
        SkeletonKeyStats(
            key=k,
            successes=stats.successes.get(k, 0),
            appearances=stats.appearances[k],
            p_hat=stats.p_hat(k),
        )
        for k in stats.appearances
    ]
    if rank_by == "successes":
        rows.sort(key=lambda r: (-r.successes, -r.appearances, r.key))
    else:
        rows.sort(key=lambda r: (-r.p_hat, -r.appearances, r.key))
    return rows[:n]


@dataclass(frozen=True)
class SuccessOccurrence:
    """One successful (problem, skeleton_idx) pair in which a key appeared."""

    problem_id: int
    skeleton_idx: int
    refinement_seed: int
    refinement_wall_clock_s: float


def find_test_successes_for_key(
    target_key: SkeletonKey,
    test: LoadedSplit,
) -> list[SuccessOccurrence]:
    """Every test episode where ``target_key`` appeared *and* succeeded.

    The canonical key is matched after canonicalization; ``skeleton_idx`` is
    the pool index in the raw (and canonical) episode — both share it because
    :func:`canonicalize_episode` does not reorder the pool.
    """
    out: list[SuccessOccurrence] = []
    for ep_idx, ep in enumerate(test.episodes):
        keys = test.skeleton_keys[ep_idx]
        for pool_idx, k in enumerate(keys):
            if k != target_key:
                continue
            outcome = ep.outcomes[pool_idx]
            if outcome.outcome != "success":
                continue
            out.append(
                SuccessOccurrence(
                    problem_id=ep.provenance.problem_id,
                    skeleton_idx=pool_idx,
                    refinement_seed=outcome.refinement_seed,
                    refinement_wall_clock_s=outcome.refinement_wall_clock_s,
                )
            )
    return out


def render_successful_refinement_video(
    *,
    env_id: str,
    model_name: str,
    model_kwargs: dict[str, int | float | str],
    problem_id: int,
    skeleton_idx: int,
    K_max: int,
    heuristic_name: str,
    abstract_plan_timeout_s: float,
    refinement_timeout_s: float,
    num_sampling_attempts_per_step: int,
    max_trajectory_steps: int,
    refinement_seed_rule: str,
    video_dir: Path,
    video_name_prefix: str,
) -> Path:
    """Reproduce a stored success and save a video of its execution.

    Reconstructs the (``env``, ``env_models``, ``bpg``, ``plan_generator``,
    ``trajectory_sampler``, ``refiner``) tuple exactly as :func:`collect_episode`
    did, then executes the refined action sequence on a
    :class:`gymnasium.wrappers.RecordVideo`-wrapped env.

    Raises ``RuntimeError`` if refinement returns ``None`` (indicates non-
    determinism between collection and replay — usually a config mismatch).
    """
    # Heavy imports deferred — EDA functions that don't render videos shouldn't
    # pay the import cost of the full planning substrate.
    import itertools

    import kinder as _kinder
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
    from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
    from bilevel_planning.structs import RelationalAbstractGoal
    from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
        ParameterizedControllerTrajectorySampler,
    )
    from bilevel_planning.utils import RelationalControllerGenerator
    from gymnasium.wrappers import RecordVideo
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    from alphatamp.approaches.spectre.collect import _refinement_seed
    from alphatamp.approaches.spectre.env_registry import register_extra_envs

    register_extra_envs()
    video_dir.mkdir(parents=True, exist_ok=True)
    env = _kinder.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=video_name_prefix,
        episode_trigger=lambda i: i == 0,
    )
    try:
        obs, _ = env.reset(seed=problem_id)
        env_models = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            **model_kwargs,
        )
        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        assert isinstance(goal, RelationalAbstractGoal)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        plan_generator: RelationalHeuristicSearchAbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                env_models.types,
                env_models.predicates,
                env_models.operators,
                heuristic_name=heuristic_name,
                seed=problem_id,
            )
        )
        pool = list(
            itertools.islice(
                plan_generator(x0, s0, goal, abstract_plan_timeout_s, bpg), K_max
            )
        )
        if skeleton_idx >= len(pool):
            raise RuntimeError(
                f"skeleton_idx {skeleton_idx} >= regenerated pool size {len(pool)}"
            )
        state_plan, action_plan = pool[skeleton_idx]

        trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(env_models.skills),
            transition_function=env_models.transition_fn,
            state_abstractor=env_models.state_abstractor,
            max_trajectory_steps=max_trajectory_steps,
        )
        seed = _refinement_seed(refinement_seed_rule, problem_id, skeleton_idx)
        refiner = BacktrackingRefiner(
            trajectory_sampler=trajectory_sampler,
            num_sampling_attempts_per_step=num_sampling_attempts_per_step,
            seed=seed,
        )
        plan = refiner(x0, state_plan, action_plan, refinement_timeout_s, bpg)
        if plan is None:
            raise RuntimeError(
                f"Refinement returned None for problem_id={problem_id},"
                f" skeleton_idx={skeleton_idx}; replay diverged from collection"
            )

        done = False
        for action in plan.actions:
            _, _, done, _, _ = env.step(action)
            if done:
                break
        if not done:
            raise RuntimeError(
                f"Plan executed without signalling 'done' on problem {problem_id}"
            )
    finally:
        env.close()  # type: ignore[no-untyped-call]

    return video_dir / f"{video_name_prefix}-episode-0.mp4"


def _fit_marginals(train: LoadedSplit) -> _MarginalStats:
    stats = _MarginalStats()
    for ep_idx, ep in enumerate(train.episodes):
        keys = train.skeleton_keys[ep_idx]
        for pool_idx, key in enumerate(keys):
            stats.appearances[key] = stats.appearances.get(key, 0) + 1
            if ep.outcomes[pool_idx].outcome == "success":
                stats.successes[key] = stats.successes.get(key, 0) + 1
    return stats


def static_historical_baseline(
    train: LoadedSplit,
    test: LoadedSplit,
    attempt_budget: int = 20,
) -> BaselineResult:
    """B3 (§4.3). Fit Laplace marginals on ``train``, evaluate on ``test``.

    Rank each test episode's pool by ``p̂(key(s))`` descending, ties broken
    by the skeleton's original pool index (= planner canonical order). The
    per-episode signature separates fit data from eval data; evaluating on
    the fitting split would be in-sample and misleading.
    """
    stats = _fit_marginals(train)
    trainable = _trainable_episodes(test)
    attempts = np.zeros(len(trainable), dtype=float)
    wall_clock = np.zeros(len(trainable), dtype=float)
    censored = np.zeros(len(trainable), dtype=bool)
    problem_ids = np.zeros(len(trainable), dtype=np.int64)
    for out_idx, ep_idx in enumerate(trainable):
        ep = test.episodes[ep_idx]
        keys = test.skeleton_keys[ep_idx]
        # Sort by (-p_hat, pool_idx) for descending p_hat with canonical tie-break.
        scored = [(-stats.p_hat(k), i) for i, k in enumerate(keys)]
        scored.sort()
        order = [i for _, i in scored]
        a, w, c = _simulate_traversal(ep, order, attempt_budget)
        attempts[out_idx] = a
        wall_clock[out_idx] = w
        censored[out_idx] = c
        problem_ids[out_idx] = ep.provenance.problem_id
    return BaselineResult(
        name="B3_static_historical",
        attempts=attempts,
        wall_clock=wall_clock,
        censored=censored,
        problem_ids=problem_ids,
    )


@dataclass(frozen=True)
class _AdaptiveStats:
    """Marginals + pairwise conditionals for the adaptive baseline.

    ``pair_appearances[(k, k')]`` counts episodes where both ``k`` and
    ``k'`` appeared *and* ``k'`` failed. ``pair_successes[(k, k')]`` counts
    those episodes where additionally ``k`` succeeded. Add-one smoothing is
    applied at lookup time to handle unseen pairs.
    """

    marginals: _MarginalStats
    pair_appearances: dict[tuple[SkeletonKey, SkeletonKey], int] = field(
        default_factory=dict
    )
    pair_successes: dict[tuple[SkeletonKey, SkeletonKey], int] = field(
        default_factory=dict
    )

    def p_hat_cond(self, k: SkeletonKey, k_failed: SkeletonKey) -> float | None:
        """Return ``p̂(k succeeds | k_failed failed)`` or ``None`` if unseen.

        We use ``None`` as the sentinel for "no data at all" so the caller
        can fall back to the marginal (log-ratio = 0) rather than silently
        applying add-one smoothing on an empty cell (which would bias
        the score toward 0.5 even when marginal says otherwise).
        """
        a = self.pair_appearances.get((k, k_failed), 0)
        if a == 0:
            return None
        s = self.pair_successes.get((k, k_failed), 0)
        return (s + 1.0) / (a + 2.0)


def _fit_adaptive(train: LoadedSplit) -> _AdaptiveStats:
    stats = _AdaptiveStats(marginals=_fit_marginals(train))
    for ep_idx, ep in enumerate(train.episodes):
        keys = train.skeleton_keys[ep_idx]
        failed_mask = [o.outcome == "fail" for o in ep.outcomes]
        success_mask = [o.outcome == "success" for o in ep.outcomes]
        # For every (k, k') with k' failed, increment pair_appearances; also
        # pair_successes if k succeeded. Both k and k' must be in this
        # episode's pool by construction.
        for i, k in enumerate(keys):
            k_succ = success_mask[i]
            for j, k_prime in enumerate(keys):
                if i == j or not failed_mask[j]:
                    continue
                key = (k, k_prime)
                stats.pair_appearances[key] = stats.pair_appearances.get(key, 0) + 1
                if k_succ:
                    stats.pair_successes[key] = stats.pair_successes.get(key, 0) + 1
    return stats


def adaptive_historical_baseline(
    train: LoadedSplit,
    test: LoadedSplit,
    attempt_budget: int = 20,
) -> BaselineResult:
    """B4 (§4.4). Greedy Naive-Bayes log-odds traversal on the test split.

    Score per candidate ``k`` at step ``t`` with failed set ``F``:

        score(k, F) = log p̂(k) + Σ_{k' ∈ F} log[p̂(k|k' failed) / p̂(k)]

    Missing ``p̂(k|k' failed)`` entries contribute 0 (i.e. no update) per
    the plan's "log-ratio = 0 for unseen pairs" contract. Ties are broken
    by the skeleton's original pool index.
    """
    stats = _fit_adaptive(train)
    trainable = _trainable_episodes(test)
    attempts = np.zeros(len(trainable), dtype=float)
    wall_clock = np.zeros(len(trainable), dtype=float)
    censored = np.zeros(len(trainable), dtype=bool)
    problem_ids = np.zeros(len(trainable), dtype=np.int64)
    for out_idx, ep_idx in enumerate(trainable):
        ep = test.episodes[ep_idx]
        keys = test.skeleton_keys[ep_idx]
        remaining = set(range(len(keys)))
        failed_keys: list[SkeletonKey] = []
        steps = 0
        wall = 0.0
        attempts_i: float = attempt_budget + 1
        censored_i = True
        while remaining and steps < attempt_budget:
            best_score = -np.inf
            best_idx = min(remaining)  # tie-break fallback
            for idx in remaining:
                k = keys[idx]
                p_marginal = stats.marginals.p_hat(k)
                score = float(np.log(p_marginal))
                for k_prime in failed_keys:
                    cond = stats.p_hat_cond(k, k_prime)
                    if cond is None:
                        continue  # no update; log-ratio := 0
                    score += float(np.log(cond / p_marginal))
                if (score > best_score) or (score == best_score and idx < best_idx):
                    best_score = score
                    best_idx = idx
            chosen = best_idx
            steps += 1
            outcome = ep.outcomes[chosen]
            wall += outcome.refinement_wall_clock_s
            if outcome.outcome == "success":
                attempts_i = steps
                censored_i = False
                break
            failed_keys.append(keys[chosen])
            remaining.remove(chosen)
        attempts[out_idx] = attempts_i
        wall_clock[out_idx] = wall
        censored[out_idx] = censored_i
        problem_ids[out_idx] = ep.provenance.problem_id
    return BaselineResult(
        name="B4_adaptive_historical",
        attempts=attempts,
        wall_clock=wall_clock,
        censored=censored,
        problem_ids=problem_ids,
    )


# ---------------------------------------------------------------------------
# Group 3 — scalars with bootstrap CIs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarWithCI:
    """Point estimate plus 95% confidence interval (percentile bootstrap)."""

    point: float
    ci_low: float
    ci_high: float


def bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> ScalarWithCI:
    """Paired bootstrap of ``mean(a) - mean(b)``.

    ``a`` and ``b`` must have the same length and be indexed by episode
    (i.e. ``a[i]`` and ``b[i]`` come from the same test episode). The same
    resampled indices are used in both arms each iteration, preserving the
    pairing — correct for Δ on matched data.
    """
    assert len(a) == len(b), "paired arrays required"
    n = len(a)
    if n == 0:
        return ScalarWithCI(point=0.0, ci_low=0.0, ci_high=0.0)
    rng = np.random.default_rng(seed)
    diffs = np.empty(num_resamples, dtype=float)
    for r in range(num_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[r] = float(a[idx].mean() - b[idx].mean())
    point = float(a.mean() - b.mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    return ScalarWithCI(point=point, ci_low=float(low), ci_high=float(high))


def adaptive_premium(
    b3: BaselineResult,
    b4: BaselineResult,
    metric: Literal["attempts", "wall_clock"] = "attempts",
    num_resamples: int = 10_000,
    seed: int = 0,
) -> ScalarWithCI:
    """Δ = mean(B3) − mean(B4) with paired bootstrap CI (§5.1).

    Alignment: B3 and B4 must have been computed from the same trainable-episode
    filter (both take the same ``test`` split). Caller is responsible for
    passing compatible baselines; we sanity-check by comparing ``problem_ids``.
    """
    _assert_aligned(b3, b4)
    a = b3.attempts if metric == "attempts" else b3.wall_clock
    b = b4.attempts if metric == "attempts" else b4.wall_clock
    return bootstrap_mean_difference(a, b, num_resamples=num_resamples, seed=seed)


def headroom(
    b2: BaselineResult,
    b5: BaselineResult,
    metric: Literal["attempts", "wall_clock"] = "attempts",
    num_resamples: int = 10_000,
    seed: int = 0,
) -> ScalarWithCI:
    """H = mean(B2) − mean(B5) with paired bootstrap CI (§5.2)."""
    _assert_aligned(b2, b5)
    a = b2.attempts if metric == "attempts" else b2.wall_clock
    b = b5.attempts if metric == "attempts" else b5.wall_clock
    return bootstrap_mean_difference(a, b, num_resamples=num_resamples, seed=seed)


def _assert_aligned(x: BaselineResult, y: BaselineResult) -> None:
    if not np.array_equal(x.problem_ids, y.problem_ids):
        raise ValueError(
            f"Baseline {x.name} and {y.name} have mismatched problem_ids;"
            " cannot compute paired difference."
        )


# ---------------------------------------------------------------------------
# Pass bar (§6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassBarVerdict:
    """Pass/fail decision per SPECTRE_EDA_SPEC.md §6 with interpretation."""

    pool_cap_saturated: bool
    diversity_nontrivial: bool
    success_rate_adequate: bool
    default_budget_exhaustion: bool
    adaptive_premium_positive: bool
    headroom_meaningful: bool
    disjoint_pools_flag: bool
    details: dict[str, float]

    @property
    def primary_pass(self) -> bool:
        """All five primary §6 conditions hold."""
        return all(
            [
                self.pool_cap_saturated,
                self.diversity_nontrivial,
                self.success_rate_adequate,
                self.default_budget_exhaustion,
                self.adaptive_premium_positive,
            ]
        )

    def interpretive_note(self) -> str | None:
        """Caveat text when the strict bar misleads (disjoint-pool regime)."""
        if self.disjoint_pools_flag and not self.adaptive_premium_positive:
            return (
                "Disjoint-pool regime: B3/B4 degenerate to default order,"
                " so Δ≈0 is mechanical and does not imply SPECTRE cannot"
                " help. SPECTRE's learned Φ/Ψ may still exploit structure"
                " that discrete-key baselines cannot (SPEC §5.1 caveat)."
            )
        return None


def evaluate_pass_bar(
    *,
    pool_cap_fraction_value: float,
    diversity_U: int,
    k_max: int,
    success_fraction: float,
    budget_exhaustion_fraction: float,
    adaptive_premium_ci: ScalarWithCI,
    headroom_ci: ScalarWithCI,
    key_overlap: KeyOverlapReport,
    diversity_multiplier_threshold: float = 4.0,
    success_rate_threshold: float = 0.5,
    budget_exhaustion_threshold: float = 0.1,
    headroom_threshold: float = 2.0,
    pool_cap_threshold: float = 0.95,
) -> PassBarVerdict:
    """Combine the Group 1/3 scalars into a :class:`PassBarVerdict`.

    Thresholds match the primary conditions in SPECTRE_EDA_SPEC.md §6. Exposed as kwargs
    so the notebook can loosen them for pilot data without editing source.
    """
    return PassBarVerdict(
        pool_cap_saturated=pool_cap_fraction_value >= pool_cap_threshold,
        diversity_nontrivial=diversity_U >= diversity_multiplier_threshold * k_max,
        success_rate_adequate=success_fraction >= success_rate_threshold,
        default_budget_exhaustion=(
            budget_exhaustion_fraction >= budget_exhaustion_threshold
        ),
        adaptive_premium_positive=(
            adaptive_premium_ci.point > 0 and adaptive_premium_ci.ci_low > 0
        ),
        headroom_meaningful=headroom_ci.point >= headroom_threshold,
        disjoint_pools_flag=key_overlap.regime() == "disjoint",
        details={
            "pool_cap_fraction": pool_cap_fraction_value,
            "diversity_U": float(diversity_U),
            "k_max": float(k_max),
            "success_fraction": success_fraction,
            "budget_exhaustion_fraction": budget_exhaustion_fraction,
            "adaptive_premium": adaptive_premium_ci.point,
            "adaptive_premium_ci_low": adaptive_premium_ci.ci_low,
            "adaptive_premium_ci_high": adaptive_premium_ci.ci_high,
            "headroom": headroom_ci.point,
            "headroom_ci_low": headroom_ci.ci_low,
            "headroom_ci_high": headroom_ci.ci_high,
            "test_keys_seen_fraction": key_overlap.test_keys_seen_fraction,
            "pairwise_cooccurrence_density": (
                key_overlap.pairwise_cooccurrence_density
            ),
        },
    )
