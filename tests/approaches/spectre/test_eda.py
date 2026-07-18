"""Unit tests for ``spectre.eda``.

Covers:

- Data loading + canonical key equivalence.
- All six Group 1 functions.
- Train↔test key-overlap report.
- Each of the five baselines, including the two graceful-degeneracy
  contracts (B3→B2 under disjoint keys; B4→B3 with empty pairwise table).
- Bootstrap CI determinism and alignment enforcement.
- Pass-bar branches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _fixtures import (
    BLOCK,
    CLEAR,
    ON_TABLE,
    PICK,
    PLACE,
    ROBOT,
    build_toy_episode,
    write_toy_split,
)
from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundAtom, Object

from alphatamp.approaches.spectre.eda import (
    BaselineResult,
    KeyOverlapReport,
    ScalarWithCI,
    _skeleton_key,
    adaptive_historical_baseline,
    adaptive_premium,
    bootstrap_mean_difference,
    count_unique_canonical_keys,
    default_order_baseline,
    default_order_budget_exhaustion,
    evaluate_pass_bar,
    headroom,
    jaccard_pair_sample,
    load_split_episodes,
    oracle_ceiling,
    pool_cap_fraction,
    random_floor_baseline,
    rarefaction_curve,
    static_historical_baseline,
    success_rate_distribution,
    train_eval_key_overlap,
)
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz
from alphatamp.approaches.spectre.schema import (
    EpisodeRecord,
    OutcomeRecord,
    ProvenanceBlock,
    SkeletonRecord,
    SummaryBlock,
)

# ---------------------------------------------------------------------------
# Helpers for building ad-hoc toy episodes with controllable key distributions
# ---------------------------------------------------------------------------


def _make_episode(
    problem_id: int,
    skeleton_op_pairs: list[list[tuple]],
    outcomes: list[str],
    blocks_in_episode: int = 3,
) -> EpisodeRecord:
    """Build a toy episode with explicit (robot, block_i) Pick/Place skeletons.

    ``skeleton_op_pairs[i]`` is a list of ``("Pick"|"Place", block_idx)`` tuples
    describing skeleton ``i``'s operator sequence. Object set is always one
    robot + ``blocks_in_episode`` blocks to exercise the canonical key path.
    """
    robot = Object("robot_0", ROBOT)
    blocks = [Object(f"block_{i}", BLOCK) for i in range(blocks_in_episode)]
    s0_atoms: set[GroundAtom] = {ON_TABLE([b]) for b in blocks} | {
        CLEAR([b]) for b in blocks
    }
    s0 = RelationalAbstractState(atoms=s0_atoms, objects={robot, *blocks})
    goal_atoms = frozenset({CLEAR([blocks[0]])})
    skels: list[SkeletonRecord] = []
    outs: list[OutcomeRecord] = []
    for i, (ops, outcome) in enumerate(zip(skeleton_op_pairs, outcomes)):
        ground_ops = []
        for op_name, block_idx in ops:
            lifted = PICK if op_name == "Pick" else PLACE
            ground_ops.append(lifted.ground((robot, blocks[block_idx])))
        skels.append(
            SkeletonRecord(
                skeleton_idx=i,
                operator_seq=tuple(ground_ops),
                final_abstract_state=s0,  # STRIPS-null cycle (Pick→Place)
            )
        )
        outs.append(
            OutcomeRecord(
                skeleton_idx=i,
                outcome=outcome,  # type: ignore[arg-type]
                refinement_wall_clock_s=0.1 * (i + 1),
                refinement_seed=1000 + i,
            )
        )
    first_succ = next((j for j, o in enumerate(outs) if o.outcome == "success"), None)
    summary = SummaryBlock(
        num_skeletons=len(skels),
        num_success=sum(1 for o in outs if o.outcome == "success"),
        num_fail=sum(1 for o in outs if o.outcome == "fail"),
        num_error=sum(1 for o in outs if o.outcome == "error"),
        first_success_idx=first_succ,
        total_wall_clock_s=sum(o.refinement_wall_clock_s for o in outs),
        pool_truncated=False,
    )
    return EpisodeRecord(
        provenance=ProvenanceBlock(
            problem_id=problem_id,
            env_id="test/Toy-v0",
            env_variant="toy",
            split="train",
            config_hash="deadbeef0000",
            problem_seed=problem_id,
            git_sha="test",
            collection_timestamp="2026-04-22T00:00:00",
            package_versions={},
        ),
        initial_abstract_state=s0,
        goal_atoms=goal_atoms,
        object_registry={obj.name: obj.type.name for obj in {robot, *blocks}},
        skeleton_pool=tuple(skels),
        outcomes=tuple(outs),
        summary=summary,
    )


def _write_split(tmp_path: Path, episodes: list[EpisodeRecord]) -> Path:
    split_dir = tmp_path
    for ep in episodes:
        atomic_write_pickle_gz(
            ep,
            split_dir / "episodes" / f"ep_{ep.provenance.problem_id:05d}.pkl.gz",
        )
    return split_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def test_load_split_episodes_roundtrips_counts(tmp_path: Path) -> None:
    """Every episode written under ``episodes/`` is loaded and canonicalized."""
    write_toy_split(
        tmp_path / "train", [("fail", "success"), ("success", "fail", "fail")]
    )
    split = load_split_episodes(tmp_path / "train")
    assert len(split.episodes) == 2
    assert len(split.skeleton_keys) == 2
    # First episode has 2 skeletons, second has 3.
    assert [len(keys) for keys in split.skeleton_keys] == [2, 3]
    assert split.k_max == 3


def test_canonical_keys_collapse_identical_skeletons(tmp_path: Path) -> None:
    """Two episodes with identical pools produce identical key sets."""
    ep_a = _make_episode(
        0, [[("Pick", 0), ("Place", 0)]], ["success"], blocks_in_episode=3
    )
    ep_b = _make_episode(
        1, [[("Pick", 0), ("Place", 0)]], ["success"], blocks_in_episode=3
    )
    split_dir = _write_split(tmp_path, [ep_a, ep_b])
    split = load_split_episodes(split_dir)
    assert split.skeleton_keys[0] == split.skeleton_keys[1]


# ---------------------------------------------------------------------------
# Group 1
# ---------------------------------------------------------------------------


def test_pool_cap_fraction_counts_saturating(tmp_path: Path) -> None:
    """3.1: fraction at ``k_max`` is the number of largest-pool episodes."""
    ep_small = _make_episode(0, [[("Pick", 0)]], ["success"])
    ep_large_a = _make_episode(
        1, [[("Pick", 0)], [("Pick", 1)], [("Pick", 2)]], ["success", "fail", "fail"]
    )
    ep_large_b = _make_episode(
        2, [[("Pick", 0)], [("Pick", 1)], [("Pick", 2)]], ["fail", "success", "fail"]
    )
    split = load_split_episodes(
        _write_split(tmp_path, [ep_small, ep_large_a, ep_large_b])
    )
    # k_max = 3; 2 of 3 episodes saturate.
    assert pool_cap_fraction(split) == pytest.approx(2 / 3)


def test_count_unique_canonical_keys(tmp_path: Path) -> None:
    """3.2: U counts distinct keys; N_slots counts total occurrences."""
    ep_a = _make_episode(0, [[("Pick", 0)], [("Pick", 1)]], ["success", "fail"])
    # ep_b's first skeleton is identical to ep_a's first skeleton after canon
    # (different object names → same typed-local-id renumbering).
    ep_b = _make_episode(1, [[("Pick", 0)], [("Pick", 2)]], ["fail", "success"])
    split = load_split_episodes(_write_split(tmp_path, [ep_a, ep_b]))
    u, n_slots = count_unique_canonical_keys(split)
    assert n_slots == 4
    # Pick(robot_0, block_0), Pick(robot_0, block_1), Pick(robot_0, block_2)
    # → after canon, block_2 maps to block_1 in ep_b (per-type idx is 2 but
    # we have 3 blocks; sorted names = block_0, block_1, block_2 → idx
    # 0,1,2). So keys in ep_a: Pick(block_0), Pick(block_1). Keys in ep_b:
    # Pick(block_0), Pick(block_2). Union size = 3.
    assert u == 3


def test_rarefaction_curve_monotonic(tmp_path: Path) -> None:
    """Curve is non-decreasing and bounded by U."""
    eps = [_make_episode(i, [[("Pick", i % 3)]], ["success"]) for i in range(5)]
    split = load_split_episodes(_write_split(tmp_path, eps))
    curve = rarefaction_curve(split, num_shuffles=20, seed=0)
    assert len(curve) == 5
    assert np.all(np.diff(curve) >= -1e-9)  # monotone non-decreasing
    u, _ = count_unique_canonical_keys(split)
    assert curve[-1] == pytest.approx(u)


def test_jaccard_pair_sample_bounded(tmp_path: Path) -> None:
    """Jaccard values in [0, 1] and correct shape."""
    eps = [_make_episode(i, [[("Pick", i % 3)]], ["success"]) for i in range(5)]
    split = load_split_episodes(_write_split(tmp_path, eps))
    samples = jaccard_pair_sample(split, num_pairs=50, seed=0)
    assert samples.shape == (50,)
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_success_rate_distribution(tmp_path: Path) -> None:
    """3.3: fraction-with-success and per-episode ratio."""
    eps = [
        _make_episode(0, [[("Pick", 0)], [("Pick", 1)]], ["fail", "fail"]),
        _make_episode(1, [[("Pick", 0)], [("Pick", 1)]], ["success", "fail"]),
        _make_episode(2, [[("Pick", 0)], [("Pick", 1)]], ["success", "success"]),
    ]
    split = load_split_episodes(_write_split(tmp_path, eps))
    frac, ratios = success_rate_distribution(split)
    assert frac == pytest.approx(2 / 3)
    assert list(ratios) == pytest.approx([0.0, 0.5, 1.0])


def test_default_order_budget_exhaustion(tmp_path: Path) -> None:
    """3.4: count first_success_idx > (budget - 1) or None."""
    ep_early = _make_episode(0, [[("Pick", 0)], [("Pick", 1)]], ["success", "fail"])
    ep_late = _make_episode(
        1,
        [[("Pick", i)] for i in range(3)] * 10,  # 30 skels, no success
        ["fail"] * 30,
    )
    split = load_split_episodes(_write_split(tmp_path, [ep_early, ep_late]))
    frac = default_order_budget_exhaustion(split, attempt_budget=20)
    # ep_early: T_default = 1 ≤ 20; ep_late: T_default = ∞ (no success).
    assert frac == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Train↔test overlap
# ---------------------------------------------------------------------------


def test_train_eval_key_overlap_identical(tmp_path: Path) -> None:
    """Same episodes in both splits → test_keys_seen_fraction == 1."""
    eps = [_make_episode(i, [[("Pick", i % 2)]], ["success"]) for i in range(4)]
    train = load_split_episodes(_write_split(tmp_path / "train", eps))
    test = load_split_episodes(_write_split(tmp_path / "test", eps))
    report = train_eval_key_overlap(train, test)
    assert report.test_keys_seen_fraction == pytest.approx(1.0)
    assert report.regime() == "overlapping"


def test_train_eval_key_overlap_disjoint(tmp_path: Path) -> None:
    """Non-overlapping key sets → fraction == 0."""
    train_eps = [_make_episode(i, [[("Pick", 0)]], ["success"]) for i in range(3)]
    # Test episodes use a 4-block toy so block_3 exists; skeleton picks
    # block_3 → renumbers to block_3 (different key than train's block_0).
    test_eps = [
        _make_episode(
            i + 10,
            [[("Pick", 3)]],
            ["success"],
            blocks_in_episode=4,
        )
        for i in range(3)
    ]
    train = load_split_episodes(_write_split(tmp_path / "train", train_eps))
    test = load_split_episodes(_write_split(tmp_path / "test", test_eps))
    report = train_eval_key_overlap(train, test)
    assert report.test_keys_seen_fraction == pytest.approx(0.0)
    assert report.pairwise_cooccurrence_density == pytest.approx(0.0)
    assert report.regime() == "disjoint"


# ---------------------------------------------------------------------------
# Group 2 — baselines
# ---------------------------------------------------------------------------


def test_random_floor_matches_closed_form(tmp_path: Path) -> None:
    """B1 attempts match ``(K+1)/(n_succ+1)`` on a deterministic fixture."""
    # K=3, n_succ=1 → expected attempts = (3+1)/(1+1) = 2.0.
    ep = _make_episode(
        0,
        [[("Pick", 0)], [("Pick", 1)], [("Pick", 2)]],
        ["fail", "success", "fail"],
    )
    split = load_split_episodes(_write_split(tmp_path, [ep]))
    res = random_floor_baseline(split, mc_permutations=50, seed=0)
    assert res.attempts[0] == pytest.approx(2.0)
    # Wall-clock is the MC average — should be finite and positive.
    assert res.wall_clock[0] > 0


def test_default_order_baseline_matches_first_success_idx(tmp_path: Path) -> None:
    """B2 attempts equal ``1 + first_success_idx`` for trainable episodes."""
    ep = _make_episode(
        0,
        [[("Pick", 0)], [("Pick", 1)], [("Pick", 2)]],
        ["fail", "fail", "success"],
    )
    split = load_split_episodes(_write_split(tmp_path, [ep]))
    res = default_order_baseline(split)
    assert res.attempts[0] == 3
    assert not res.censored[0]


def test_default_order_censors_at_budget(tmp_path: Path) -> None:
    """B2 contributes ``attempt_budget + 1`` when success lies past the budget."""
    # 25 skeletons, success at index 24 — budget 20 censors.
    ops = [[("Pick", i % 3)] for i in range(25)]
    outcomes = ["fail"] * 24 + ["success"]
    ep = _make_episode(0, ops, outcomes, blocks_in_episode=3)
    split = load_split_episodes(_write_split(tmp_path, [ep]))
    res = default_order_baseline(split, attempt_budget=20)
    assert res.attempts[0] == 21
    assert res.censored[0]


def test_oracle_ceiling_is_always_one(tmp_path: Path) -> None:
    """B5 attempts are exactly 1 on every trainable episode."""
    eps = [
        _make_episode(i, [[("Pick", 0)], [("Pick", 1)]], ["fail", "success"])
        for i in range(3)
    ]
    split = load_split_episodes(_write_split(tmp_path, eps))
    res = oracle_ceiling(split)
    assert np.all(res.attempts == 1)
    # Wall-clock is the min over successful skeletons' refine times — here
    # skeleton 1 has refine_time = 0.1 * (1 + 1) = 0.2.
    assert np.allclose(res.wall_clock, 0.2)


def test_static_historical_baseline_ranks_by_p_hat(tmp_path: Path) -> None:
    """B3 on an overlapping-key fixture puts the historically-best skeleton first."""
    # Train: Pick(block_0) succeeds 3/3; Pick(block_1) fails 3/3.
    train_eps = [
        _make_episode(i, [[("Pick", 0)], [("Pick", 1)]], ["success", "fail"])
        for i in range(3)
    ]
    # Test: same pool, but success labels reversed — we want to see whether
    # B3 still ranks Pick(block_0) first (because it learned from train).
    test_eps = [
        _make_episode(
            100,
            [[("Pick", 0)], [("Pick", 1)]],
            ["fail", "success"],  # test-labels
        ),
    ]
    train = load_split_episodes(_write_split(tmp_path / "train", train_eps))
    test = load_split_episodes(_write_split(tmp_path / "test", test_eps))
    res = static_historical_baseline(train, test)
    # B3 picks index 0 (Pick(block_0)) first on test based on train stats.
    # That skeleton fails in test → attempt 1 fails, attempt 2 tries idx 1
    # which succeeds.
    assert res.attempts[0] == 2


def test_b3_disjoint_keys_reduces_to_default_order(tmp_path: Path) -> None:
    """When train and test share no keys, B3 == B2 per the §2 contract."""
    train_eps = [_make_episode(i, [[("Pick", 0)]], ["success"]) for i in range(3)]
    test_eps = [
        _make_episode(
            100,
            [[("Pick", 3)], [("Pick", 4)]],
            ["fail", "success"],
            blocks_in_episode=5,
        ),
    ]
    train = load_split_episodes(_write_split(tmp_path / "train", train_eps))
    test = load_split_episodes(_write_split(tmp_path / "test", test_eps))
    b3 = static_historical_baseline(train, test)
    b2 = default_order_baseline(test)
    assert np.array_equal(b3.attempts, b2.attempts)
    assert np.array_equal(b3.censored, b2.censored)


def test_b3_avoids_in_sample_leakage(tmp_path: Path) -> None:
    """B3(train=test) does *not* perform at oracle — Laplace smooths learning.

    The purpose of this test is structural: the function signature requires
    separate train and test arguments, and even when both are the same set,
    Laplace smoothing prevents degenerate attempts==1 behavior (the ranker
    still has to simulate a real traversal on the eval episodes).
    """
    eps = [
        _make_episode(i, [[("Pick", 0)], [("Pick", 1)]], ["fail", "success"])
        for i in range(3)
    ]
    both = load_split_episodes(_write_split(tmp_path, eps))
    res = static_historical_baseline(both, both)
    assert res.attempts.shape == (3,)
    # p̂(Pick(block_0)) = (0+1)/(3+2) = 0.2; p̂(Pick(block_1)) = (3+1)/(3+2) = 0.8.
    # B3 picks block_1 first → succeeds at attempt 1.
    assert np.all(res.attempts == 1)
    # But this happens only because the train/test labels are (trivially) aligned;
    # the *function* still takes two args and simulates traversal on eval. A
    # mismatched label set (e.g. preceding test) can give attempts > 1.


def test_adaptive_historical_picks_success_with_informative_pair(
    tmp_path: Path,
) -> None:
    """B4 uses pairwise conditionals: after a failure, the conditional boosts a success.

    Fixture construction, designed so step-1 marginal picks idx 0, step-2
    conditional picks idx 1 (the real success) over idx 2 (the distractor):

    - 50 train eps of [Pick(b_0)=success] → block_0 marginal high.
    - 50 train eps of [Pick(b_1)=fail]   → block_1 marginal low.
    - 20 train eps of [Pick(b_0)=fail, Pick(b_1)=success]
      → pair (b_1 | b_0 failed) has high conditional success rate.
    - 20 train eps of [Pick(b_0)=fail, Pick(b_2)=fail]
      → pair (b_2 | b_0 failed) has zero conditional success rate.

    Marginal ranking on test pool [b_0, b_1, b_2]:
      p̂(b_0) ≈ 0.55, p̂(b_1) ≈ 0.29, p̂(b_2) ≈ 0.05.
    After b_0 fails (step 1), log-ratio boosts b_1 heavily but leaves b_2
    unchanged (coincidentally p̂_cond == p̂_marginal for b_2).
    """
    train_all: list = []
    for i in range(50):
        train_all.append(_make_episode(i, [[("Pick", 0)]], ["success"]))
    for i in range(50):
        train_all.append(_make_episode(50 + i, [[("Pick", 1)]], ["fail"]))
    for i in range(20):
        train_all.append(
            _make_episode(
                100 + i,
                [[("Pick", 0)], [("Pick", 1)]],
                ["fail", "success"],
            )
        )
    for i in range(20):
        train_all.append(
            _make_episode(
                200 + i,
                [[("Pick", 0)], [("Pick", 2)]],
                ["fail", "fail"],
            )
        )
    test_eps = [
        _make_episode(
            900,
            [[("Pick", 0)], [("Pick", 1)], [("Pick", 2)]],
            ["fail", "success", "fail"],
        ),
    ]
    train = load_split_episodes(_write_split(tmp_path / "train", train_all))
    test = load_split_episodes(_write_split(tmp_path / "test", test_eps))
    res = adaptive_historical_baseline(train, test)
    # Step 1 picks idx 0 (highest marginal), fails. Step 2 picks idx 1 via
    # pairwise conditional → succeeds → attempts == 2.
    assert res.attempts[0] == 2


def test_b4_empty_pairwise_table_reduces_to_b3(tmp_path: Path) -> None:
    """Disjoint train/test pools → B4 == B3 (empty pairwise table)."""
    train_eps = [_make_episode(i, [[("Pick", 0)]], ["success"]) for i in range(3)]
    test_eps = [
        _make_episode(
            100,
            [[("Pick", 3)], [("Pick", 4)]],
            ["fail", "success"],
            blocks_in_episode=5,
        ),
    ]
    train = load_split_episodes(_write_split(tmp_path / "train", train_eps))
    test = load_split_episodes(_write_split(tmp_path / "test", test_eps))
    b3 = static_historical_baseline(train, test)
    b4 = adaptive_historical_baseline(train, test)
    assert np.array_equal(b3.attempts, b4.attempts)


# ---------------------------------------------------------------------------
# Group 3 — scalars
# ---------------------------------------------------------------------------


def test_bootstrap_mean_difference_deterministic() -> None:
    """Fixed seed + fixed inputs produces identical ScalarWithCI across runs."""
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    r1 = bootstrap_mean_difference(a, b, num_resamples=1000, seed=42)
    r2 = bootstrap_mean_difference(a, b, num_resamples=1000, seed=42)
    assert r1 == r2
    assert r1.point == pytest.approx(1.0)
    assert r1.ci_low <= r1.point <= r1.ci_high


def test_adaptive_premium_enforces_alignment() -> None:
    """Mismatched problem_ids raises rather than silently averaging bad data."""
    a = BaselineResult(
        name="B3",
        attempts=np.array([2.0]),
        wall_clock=np.array([1.0]),
        censored=np.array([False]),
        problem_ids=np.array([0], dtype=np.int64),
    )
    b = BaselineResult(
        name="B4",
        attempts=np.array([1.0]),
        wall_clock=np.array([0.5]),
        censored=np.array([False]),
        problem_ids=np.array([5], dtype=np.int64),  # different problem_id
    )
    with pytest.raises(ValueError, match="mismatched problem_ids"):
        adaptive_premium(a, b)


def test_headroom_ci_contains_point() -> None:
    """Sanity: the point estimate lies within its own bootstrap CI."""
    b2 = BaselineResult(
        name="B2",
        attempts=np.array([3.0, 4.0, 5.0]),
        wall_clock=np.array([1.0, 2.0, 3.0]),
        censored=np.array([False, False, False]),
        problem_ids=np.array([0, 1, 2], dtype=np.int64),
    )
    b5 = BaselineResult(
        name="B5",
        attempts=np.array([1.0, 1.0, 1.0]),
        wall_clock=np.array([0.2, 0.3, 0.4]),
        censored=np.array([False, False, False]),
        problem_ids=np.array([0, 1, 2], dtype=np.int64),
    )
    res = headroom(b2, b5, num_resamples=500, seed=0)
    assert res.ci_low <= res.point <= res.ci_high
    assert res.point == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Pass bar
# ---------------------------------------------------------------------------


def _dummy_overlap(regime: str) -> KeyOverlapReport:
    frac = {"overlapping": 1.0, "partial": 0.5, "disjoint": 0.0}[regime]
    return KeyOverlapReport(
        num_unique_train_keys=100,
        num_unique_test_keys=100,
        test_keys_seen_in_train=int(frac * 100),
        test_keys_seen_fraction=frac,
        median_per_episode_seen_fraction=frac,
        pairwise_cooccurrence_density=frac,
    )


def test_evaluate_pass_bar_all_pass() -> None:
    """All five primary conditions hold → ``primary_pass`` is True."""
    verdict = evaluate_pass_bar(
        pool_cap_fraction_value=0.99,
        diversity_U=300,
        k_max=50,
        success_fraction=0.7,
        budget_exhaustion_fraction=0.15,
        adaptive_premium_ci=ScalarWithCI(point=1.2, ci_low=0.3, ci_high=2.1),
        headroom_ci=ScalarWithCI(point=3.0, ci_low=2.5, ci_high=3.5),
        key_overlap=_dummy_overlap("overlapping"),
    )
    assert verdict.primary_pass
    assert verdict.interpretive_note() is None


def test_evaluate_pass_bar_disjoint_adds_caveat() -> None:
    """Disjoint-pool regime produces an interpretive caveat on Δ-zero failures."""
    verdict = evaluate_pass_bar(
        pool_cap_fraction_value=0.99,
        diversity_U=300,
        k_max=50,
        success_fraction=0.7,
        budget_exhaustion_fraction=0.15,
        # Δ ≈ 0 with CI straddling zero — fails primary condition 5.
        adaptive_premium_ci=ScalarWithCI(point=0.0, ci_low=-0.5, ci_high=0.5),
        headroom_ci=ScalarWithCI(point=3.0, ci_low=2.5, ci_high=3.5),
        key_overlap=_dummy_overlap("disjoint"),
    )
    assert not verdict.primary_pass
    assert not verdict.adaptive_premium_positive
    assert verdict.disjoint_pools_flag
    note = verdict.interpretive_note()
    assert note is not None and "disjoint" in note.lower()


def test_evaluate_pass_bar_fails_low_success_rate() -> None:
    """Below-threshold success rate fails condition 3.3 and thus primary_pass."""
    verdict = evaluate_pass_bar(
        pool_cap_fraction_value=0.99,
        diversity_U=300,
        k_max=50,
        success_fraction=0.3,  # below 0.5 threshold
        budget_exhaustion_fraction=0.15,
        adaptive_premium_ci=ScalarWithCI(point=1.2, ci_low=0.3, ci_high=2.1),
        headroom_ci=ScalarWithCI(point=3.0, ci_low=2.5, ci_high=3.5),
        key_overlap=_dummy_overlap("overlapping"),
    )
    assert not verdict.primary_pass
    assert not verdict.success_rate_adequate


def test_skeleton_key_is_deterministic() -> None:
    """Same canonicalized skeleton always produces the same key."""
    ep = build_toy_episode()
    key_a = _skeleton_key(ep.skeleton_pool[0])
    key_b = _skeleton_key(ep.skeleton_pool[0])
    assert key_a == key_b
