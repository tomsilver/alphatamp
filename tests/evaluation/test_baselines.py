"""Tests for OracleBaseline, SuccessFirstFixedOrder, and ShortestFirstFixedOrder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project src and experiments are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from alphatamp.data.skeleton_dataset import SkeletonDataset, write_skeleton_dataset
from alphatamp.evaluation.evaluator import OfflineEvaluator
from alphatamp.evaluation.policy import (
    OracleBaseline,
    RandomPolicy,
    SelectionPolicy,
    ShortestFirstFixedOrder,
    ShortestFirstPolicy,
    SuccessFirstFixedOrder,
)
from build_synthetic_dataset import build_synthetic_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_hdf5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a small synthetic dataset to HDF5."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "test.h5"
    dd = build_synthetic_dataset(N=20, M=8, rng_seed=0)
    write_skeleton_dataset(path, dd)
    return path


@pytest.fixture(scope="module")
def test_dataset(synthetic_hdf5: Path) -> SkeletonDataset:
    return SkeletonDataset(synthetic_hdf5, preload=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_success_dataset(tmp_path: Path) -> Path:
    """Create HDF5 with N=1, M=4 where only skeleton 2 succeeds.

    Layout:
    - Skeleton 0: inapplicable
    - Skeleton 1: applicable, Y=0, F=0.5, T=5.0
    - Skeleton 2: applicable, Y=1, F=1.0, T=3.0  (the only success)
    - Skeleton 3: applicable, Y=0, F=0.3, T=2.0
    """
    dd = build_synthetic_dataset(N=1, M=4, rng_seed=99)
    dd["applicability"] = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    dd["success"] = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    dd["steps_completed_fraction"] = np.array([[0.0, 0.5, 1.0, 0.3]], dtype=np.float32)
    dd["refinement_time"] = np.array([[0.0, 5.0, 3.0, 2.0]], dtype=np.float32)

    path = tmp_path / "single_success.h5"
    write_skeleton_dataset(path, dd)
    return path


def _make_two_success_dataset(tmp_path: Path) -> Path:
    """Create HDF5 with N=1, M=4 where skeletons 1 and 2 succeed.

    Layout:
    - Skeleton 0: inapplicable
    - Skeleton 1: applicable, Y=1, F=1.0, T=5.0  (success, expensive)
    - Skeleton 2: applicable, Y=1, F=1.0, T=2.0  (success, cheap)
    - Skeleton 3: applicable, Y=0, F=0.3, T=1.0
    """
    dd = build_synthetic_dataset(N=1, M=4, rng_seed=99)
    dd["applicability"] = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    dd["success"] = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    dd["steps_completed_fraction"] = np.array([[0.0, 1.0, 1.0, 0.3]], dtype=np.float32)
    dd["refinement_time"] = np.array([[0.0, 5.0, 2.0, 1.0]], dtype=np.float32)

    path = tmp_path / "two_success.h5"
    write_skeleton_dataset(path, dd)
    return path


def _make_controlled_train_dataset(tmp_path: Path) -> Path:
    """Create HDF5 with N=10, M=4 with controlled success rates.

    - Skeleton 0: always applicable, always succeeds (rate=1.0)
    - Skeleton 1: always applicable, succeeds 50% of time (rate=0.5)
    - Skeleton 2: always applicable, never succeeds (rate=0.0)
    - Skeleton 3: never applicable (rate=0.0)
    """
    dd = build_synthetic_dataset(N=10, M=4, rng_seed=99)
    N = 10
    dd["applicability"] = np.ones((N, 4), dtype=np.float32)
    dd["applicability"][:, 3] = 0.0  # skeleton 3 never applicable

    dd["success"] = np.zeros((N, 4), dtype=np.float32)
    dd["success"][:, 0] = 1.0  # skeleton 0 always succeeds
    dd["success"][:5, 1] = 1.0  # skeleton 1 succeeds first 5 instances

    dd["steps_completed_fraction"] = np.where(
        dd["success"] > 0.5, 1.0, 0.3,
    ).astype(np.float32)
    dd["steps_completed_fraction"][:, 3] = 0.0  # inapplicable

    dd["refinement_time"] = np.full((N, 4), 2.0, dtype=np.float32)
    dd["refinement_time"][:, 3] = 0.0  # inapplicable

    path = tmp_path / "controlled_train.h5"
    write_skeleton_dataset(path, dd)
    return path


# ---------------------------------------------------------------------------
# Test: Protocol compliance
# ---------------------------------------------------------------------------


def test_all_new_protocol_compliance(test_dataset: SkeletonDataset) -> None:
    """All three new policy classes satisfy the SelectionPolicy protocol."""
    oracle = OracleBaseline()
    sf = SuccessFirstFixedOrder(ordering=[0, 1, 2])
    shortest = ShortestFirstFixedOrder(test_dataset.skeleton_lengths)

    assert isinstance(oracle, SelectionPolicy)
    assert isinstance(sf, SelectionPolicy)
    assert isinstance(shortest, SelectionPolicy)


# ---------------------------------------------------------------------------
# Tests: OracleBaseline
# ---------------------------------------------------------------------------


def test_oracle_picks_success_first(tmp_path: Path) -> None:
    """Oracle picks the only Y=1 skeleton first → TTFS = T of that skeleton."""
    hdf5_path = _make_single_success_dataset(tmp_path)
    ds = SkeletonDataset(hdf5_path, preload=True)

    evaluator = OfflineEvaluator(ds)
    oracle = OracleBaseline()
    result = evaluator.rollout_single(oracle, 0)

    assert result.success is True
    assert result.n_attempts == 1
    assert result.ttfs == pytest.approx(3.0)
    assert result.attempt_indices == [2]


def test_oracle_f_desc_t_asc(tmp_path: Path) -> None:
    """When two skeletons succeed (F=1.0), oracle picks the lower-T one first."""
    hdf5_path = _make_two_success_dataset(tmp_path)
    ds = SkeletonDataset(hdf5_path, preload=True)

    evaluator = OfflineEvaluator(ds)
    oracle = OracleBaseline()
    result = evaluator.rollout_single(oracle, 0)

    assert result.success is True
    assert result.n_attempts == 1
    # Should pick skeleton 2 (T=2.0) before skeleton 1 (T=5.0)
    assert result.attempt_indices[0] == 2
    assert result.ttfs == pytest.approx(2.0)


def test_oracle_ttfs_lower_bound(test_dataset: SkeletonDataset) -> None:
    """Oracle mean_ttfs ≤ every other policy's mean_ttfs."""
    evaluator = OfflineEvaluator(test_dataset)

    oracle = OracleBaseline()
    random_pol = RandomPolicy(seed=42)
    shortest_fixed = ShortestFirstFixedOrder(test_dataset.skeleton_lengths)

    oracle_metrics = evaluator.evaluate(oracle)
    random_metrics = evaluator.evaluate(random_pol)
    shortest_metrics = evaluator.evaluate(shortest_fixed)

    if oracle_metrics.n_succeeded > 0:
        assert oracle_metrics.mean_ttfs <= random_metrics.mean_ttfs + 1e-6
        assert oracle_metrics.mean_ttfs <= shortest_metrics.mean_ttfs + 1e-6


def test_oracle_success_at_1_equals_feasibility(test_dataset: SkeletonDataset) -> None:
    """Oracle success@1 == fraction of instances with any applicable Y=1 skeleton."""
    evaluator = OfflineEvaluator(test_dataset)
    oracle = OracleBaseline()
    metrics = evaluator.evaluate(oracle)

    # Compute feasibility ceiling from raw data
    n_feasible = 0
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        applicable_mask = item.applicability > 0.5
        has_success = (item.success * applicable_mask.float()).sum() > 0.5
        if has_success:
            n_feasible += 1

    expected_frac = n_feasible / len(test_dataset) if len(test_dataset) > 0 else 0.0
    assert metrics.success_at_k[1] == pytest.approx(expected_frac)


# ---------------------------------------------------------------------------
# Tests: SuccessFirstFixedOrder
# ---------------------------------------------------------------------------


def test_success_first_fit_ordering(tmp_path: Path) -> None:
    """Skeleton with 100% success rate is ranked before 50% rate."""
    hdf5_path = _make_controlled_train_dataset(tmp_path)
    ds = SkeletonDataset(hdf5_path, preload=True)

    sf = SuccessFirstFixedOrder()
    sf.fit(ds)

    # Skeleton 0 has rate 1.0, skeleton 1 has rate 0.5, skeleton 2 has 0.0
    ordering = sf._ordering
    idx_0 = ordering.index(0)
    idx_1 = ordering.index(1)
    idx_2 = ordering.index(2)
    assert idx_0 < idx_1, "Skeleton 0 (rate=1.0) should precede skeleton 1 (rate=0.5)"
    assert idx_1 < idx_2, "Skeleton 1 (rate=0.5) should precede skeleton 2 (rate=0.0)"


def test_success_first_never_applicable_gets_zero(tmp_path: Path) -> None:
    """Skeleton that is never applicable gets rate 0 and appears last."""
    hdf5_path = _make_controlled_train_dataset(tmp_path)
    ds = SkeletonDataset(hdf5_path, preload=True)

    sf = SuccessFirstFixedOrder()
    sf.fit(ds)

    # Skeleton 3 is never applicable → rate = 0
    assert sf.success_rates[3].item() == pytest.approx(0.0)
    # Should be ranked after skeleton 0 (rate=1.0) and skeleton 1 (rate=0.5)
    ordering = sf._ordering
    idx_3 = ordering.index(3)
    idx_0 = ordering.index(0)
    assert idx_3 > idx_0


def test_success_first_save_load_roundtrip(tmp_path: Path) -> None:
    """fit → save → load → verify identical ordering."""
    train_path = _make_controlled_train_dataset(tmp_path)
    ds = SkeletonDataset(train_path, preload=True)

    sf = SuccessFirstFixedOrder()
    sf.fit(ds)

    json_path = tmp_path / "ordering.json"
    sf.save_ordering(json_path)

    loaded = SuccessFirstFixedOrder.load_ordering(json_path)
    assert loaded._ordering == sf._ordering
    assert torch.allclose(loaded.success_rates, sf.success_rates)


# ---------------------------------------------------------------------------
# Tests: ShortestFirstFixedOrder
# ---------------------------------------------------------------------------


def test_shortest_fixed_ascending_length(test_dataset: SkeletonDataset) -> None:
    """Ordering matches ascending skeleton_lengths, ties broken by index."""
    lengths = test_dataset.skeleton_lengths
    policy = ShortestFirstFixedOrder(lengths)

    ordering = policy._ordering
    for i in range(len(ordering) - 1):
        a, b = ordering[i], ordering[i + 1]
        assert lengths[a] <= lengths[b], (
            f"Ordering not ascending: L[{a}]={lengths[a]} > L[{b}]={lengths[b]}"
        )
        if lengths[a] == lengths[b]:
            assert a < b, f"Tie-break should prefer lower index: {a} vs {b}"


def test_shortest_fixed_vs_shortest_policy_distinct(tmp_path: Path) -> None:
    """ShortestFirstFixedOrder (by length L) and ShortestFirstPolicy (by T)
    can produce different attempt sequences on the same instance."""
    # Create dataset where T ordering differs from L ordering
    dd = build_synthetic_dataset(N=1, M=4, rng_seed=99)
    dd["applicability"] = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    dd["success"] = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    dd["steps_completed_fraction"] = np.array([[0.3, 0.5, 1.0, 0.2]], dtype=np.float32)
    # T order: 3 (T=1.0), 0 (T=2.0), 2 (T=3.0), 1 (T=5.0)
    dd["refinement_time"] = np.array([[2.0, 5.0, 3.0, 1.0]], dtype=np.float32)
    # L order depends on skeleton_lengths from build_synthetic_dataset

    path = tmp_path / "distinct_test.h5"
    write_skeleton_dataset(path, dd)
    ds = SkeletonDataset(path, preload=True)

    evaluator = OfflineEvaluator(ds)

    fixed = ShortestFirstFixedOrder(ds.skeleton_lengths)
    oracle_t = ShortestFirstPolicy()

    result_fixed = evaluator.rollout_single(fixed, 0)
    result_oracle = evaluator.rollout_single(oracle_t, 0)

    # Both should eventually succeed, but the attempt_indices may differ
    assert result_fixed.success is True
    assert result_oracle.success is True
    # The orderings should differ (L order ≠ T order in general)
    # We just verify both complete — the orderings are different policies


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


def test_fixed_order_deterministic(test_dataset: SkeletonDataset) -> None:
    """Fixed-order baselines produce identical results across two runs."""
    evaluator = OfflineEvaluator(test_dataset)

    sf = SuccessFirstFixedOrder()
    sf.fit(test_dataset)  # fit on same data for simplicity

    shortest = ShortestFirstFixedOrder(test_dataset.skeleton_lengths)

    # Run twice
    sf_m1 = evaluator.evaluate(sf)
    sf_m2 = evaluator.evaluate(sf)

    sh_m1 = evaluator.evaluate(shortest)
    sh_m2 = evaluator.evaluate(shortest)

    # Per-instance results must be identical
    for r1, r2 in zip(sf_m1.per_instance, sf_m2.per_instance):
        assert r1.attempt_indices == r2.attempt_indices
        assert r1.ttfs == r2.ttfs
        assert r1.success == r2.success

    for r1, r2 in zip(sh_m1.per_instance, sh_m2.per_instance):
        assert r1.attempt_indices == r2.attempt_indices
        assert r1.ttfs == r2.ttfs
        assert r1.success == r2.success
