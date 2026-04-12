"""Tests for IndexPolicy, baseline policies, and OfflineEvaluator."""

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

from alphatamp.data.skeleton_dataset import SkeletonDataset, SkeletonItem, write_skeleton_dataset
from alphatamp.evaluation.evaluator import EvalMetrics, OfflineEvaluator, RolloutResult
from alphatamp.evaluation.policy import (
    IndexPolicy,
    RandomPolicy,
    SelectionPolicy,
    ShortestFirstPolicy,
)
from alphatamp.models.belief_encoder import BeliefEncoder
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead
from alphatamp.models.skeleton_encoder import SkeletonEncoder
from alphatamp.models.token_builder import TokenBuilder
from build_synthetic_dataset import build_synthetic_dataset


# ---------------------------------------------------------------------------
# Constants (small model dims for fast tests)
# ---------------------------------------------------------------------------

D_SKEL = 16
D_OUT = 8
D_TOKEN = D_SKEL + D_OUT  # 24
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 1


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


@pytest.fixture
def small_models(test_dataset: SkeletonDataset) -> dict:
    """Build small model components with random weights."""
    num_op_types = len(test_dataset.op_type_vocab)
    num_objects = len(test_dataset.obj_vocab)

    return {
        "skeleton_encoder": SkeletonEncoder(
            num_op_types=num_op_types,
            num_objects=num_objects,
            d_model=D_SKEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
        ),
        "token_builder": TokenBuilder(d_skel=D_SKEL, d_out=D_OUT, dropout=0.0),
        "belief_encoder": BeliefEncoder(
            d_token=D_TOKEN, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS,
            ffn_dim=D_MODEL * 2, dropout=0.0,
        ),
        "y_head": YHead(D_MODEL, dropout=0.0),
        "t_head": THead(D_MODEL, dropout=0.0),
        "joint_y_head": JointYHead(D_MODEL, n_heads=N_HEADS, rank=4, dropout=0.0),
    }


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
    # Build a tiny synthetic vocab (M=4) using build_synthetic_dataset
    # to get properly structured op_sequence_vocab
    dd = build_synthetic_dataset(N=1, M=4, rng_seed=99)

    # Override the instance matrices
    dd["applicability"] = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    dd["success"] = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    dd["steps_completed_fraction"] = np.array([[0.0, 0.5, 1.0, 0.3]], dtype=np.float32)
    dd["refinement_time"] = np.array([[0.0, 5.0, 3.0, 2.0]], dtype=np.float32)

    path = tmp_path / "single_success.h5"
    write_skeleton_dataset(path, dd)
    return path


class _OraclePolicy:
    """Mock policy that always picks the target skeleton first."""

    def __init__(self, target_idx: int) -> None:
        self._target = target_idx

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        pass

    def select(self, candidate_mask, revealed_mask, revealed_y, revealed_f, revealed_t) -> int:
        if candidate_mask[self._target]:
            return self._target
        # Fallback: first available candidate
        return int(torch.where(candidate_mask)[0][0].item())


class _LongestFirstPolicy:
    """Oracle baseline: try skeletons in descending ground-truth T order."""

    def __init__(self) -> None:
        self._sorted_order: list[int] = []
        self._ptr: int = 0

    def reset(self, item: SkeletonItem, dataset: SkeletonDataset) -> None:
        applicable_mask = item.applicability > 0.5
        applicable_indices = torch.where(applicable_mask)[0]
        if len(applicable_indices) == 0:
            self._sorted_order = []
            self._ptr = 0
            return
        t_app = item.refinement_time[applicable_indices]
        order = torch.argsort(t_app, descending=True, stable=True)
        self._sorted_order = applicable_indices[order].tolist()
        self._ptr = 0

    def select(self, candidate_mask, revealed_mask, revealed_y, revealed_f, revealed_t) -> int:
        while self._ptr < len(self._sorted_order):
            idx = self._sorted_order[self._ptr]
            self._ptr += 1
            if candidate_mask[idx]:
                return idx
        raise RuntimeError("No candidate available")


# ---------------------------------------------------------------------------
# Test 1: Sanity test — single success instance
# ---------------------------------------------------------------------------


def test_single_success_ttfs(tmp_path: Path) -> None:
    """When only one skeleton succeeds and the policy picks it first,
    TTFS equals T of that skeleton."""
    hdf5_path = _make_single_success_dataset(tmp_path)
    ds = SkeletonDataset(hdf5_path, preload=True)

    evaluator = OfflineEvaluator(ds)

    # Oracle policy that picks skeleton 2 (the only success) first
    oracle = _OraclePolicy(target_idx=2)
    result = evaluator.rollout_single(oracle, 0)

    assert result.success is True
    assert result.n_attempts == 1
    assert result.ttfs == pytest.approx(3.0)
    assert result.attempt_indices == [2]

    # ShortestFirstPolicy: sorts by T → tries 3 (T=2.0), then 2 (T=3.0), then 1 (T=5.0)
    shortest = ShortestFirstPolicy()
    result_sf = evaluator.rollout_single(shortest, 0)

    assert result_sf.success is True
    assert result_sf.n_attempts == 2  # skeleton 3 fails, then skeleton 2 succeeds
    assert result_sf.ttfs == pytest.approx(5.0)  # 2.0 + 3.0
    assert result_sf.attempt_indices == [3, 2]


# ---------------------------------------------------------------------------
# Test 2: Exhaustion — no success
# ---------------------------------------------------------------------------


def test_exhaustion_no_success(tmp_path: Path) -> None:
    """Instance with all applicable skeletons Y=0 → success=False, ttfs=None."""
    dd = build_synthetic_dataset(N=1, M=4, rng_seed=99)
    dd["applicability"] = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    dd["success"] = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dd["steps_completed_fraction"] = np.array([[0.0, 0.3, 0.5, 0.2]], dtype=np.float32)
    dd["refinement_time"] = np.array([[0.0, 2.0, 3.0, 1.0]], dtype=np.float32)

    path = tmp_path / "no_success.h5"
    write_skeleton_dataset(path, dd)
    ds = SkeletonDataset(path, preload=True)

    evaluator = OfflineEvaluator(ds)
    random_pol = RandomPolicy(seed=42)
    result = evaluator.rollout_single(random_pol, 0)

    assert result.success is False
    assert result.ttfs is None
    assert result.n_attempts == 3  # tried all 3 applicable skeletons


# ---------------------------------------------------------------------------
# Test 3: success@k values
# ---------------------------------------------------------------------------


def test_success_at_k(test_dataset: SkeletonDataset) -> None:
    """success@k is monotonically non-decreasing and in [0, 1]."""
    evaluator = OfflineEvaluator(test_dataset)
    random_pol = RandomPolicy(seed=42)
    metrics = evaluator.evaluate(random_pol)

    assert metrics.n_instances == len(test_dataset)
    prev = 0.0
    for k in [1, 2, 3, 5]:
        val = metrics.success_at_k[k]
        assert 0.0 <= val <= 1.0, f"success@{k}={val} not in [0,1]"
        assert val >= prev, f"success@{k}={val} < success@{k-1}={prev}"
        prev = val


# ---------------------------------------------------------------------------
# Test 4: IndexPolicy returns valid candidate
# ---------------------------------------------------------------------------


def test_index_policy_returns_valid_candidate(
    test_dataset: SkeletonDataset,
    small_models: dict,
) -> None:
    """IndexPolicy.select() returns an index in the candidate set."""
    policy = IndexPolicy(
        **small_models,
        dataset=test_dataset,
        device=torch.device("cpu"),
    )

    item = test_dataset[0]
    policy.reset(item, test_dataset)

    applicable_mask = item.applicability > 0.5
    revealed_mask = ~applicable_mask  # only inapplicable revealed initially
    M = test_dataset.M

    candidate_mask = applicable_mask & ~revealed_mask
    if not candidate_mask.any():
        pytest.skip("No candidates in first instance")

    idx = policy.select(
        candidate_mask,
        revealed_mask,
        torch.zeros(M),
        torch.zeros(M),
        torch.zeros(M),
    )

    assert candidate_mask[idx], f"IndexPolicy selected {idx} which is not a candidate"
    assert 0 <= idx < M


# ---------------------------------------------------------------------------
# Test 5: Cost ratio — shortest-first vs longest-first
# ---------------------------------------------------------------------------


def test_cost_ratio_shortest_vs_longest(test_dataset: SkeletonDataset) -> None:
    """ShortestFirstPolicy should generally have lower TTFS than LongestFirst,
    so cost ratio (shortest/longest) should be < 1 for most paired instances."""
    evaluator = OfflineEvaluator(test_dataset)

    shortest = ShortestFirstPolicy()
    longest = _LongestFirstPolicy()

    metrics = evaluator.evaluate(shortest, baseline=longest)

    # If there are paired successes, cost ratio should be <= 1
    # (ShortestFirst spends less cumulative time than LongestFirst)
    if metrics.n_paired > 0:
        assert metrics.mean_cost_ratio is not None
        assert metrics.mean_cost_ratio <= 1.0 + 1e-6, (
            f"ShortestFirst should be <= LongestFirst, got ratio={metrics.mean_cost_ratio}"
        )


# ---------------------------------------------------------------------------
# Test 6: Protocol compliance
# ---------------------------------------------------------------------------


def test_protocol_compliance(
    test_dataset: SkeletonDataset,
    small_models: dict,
) -> None:
    """All policy classes satisfy the SelectionPolicy protocol."""
    index_pol = IndexPolicy(
        **small_models,
        dataset=test_dataset,
        device=torch.device("cpu"),
    )
    random_pol = RandomPolicy(seed=0)
    shortest_pol = ShortestFirstPolicy()

    assert isinstance(index_pol, SelectionPolicy)
    assert isinstance(random_pol, SelectionPolicy)
    assert isinstance(shortest_pol, SelectionPolicy)


# ---------------------------------------------------------------------------
# Test 7: IndexPolicy full evaluator integration
# ---------------------------------------------------------------------------


def test_index_policy_evaluator_integration(
    test_dataset: SkeletonDataset,
    small_models: dict,
) -> None:
    """Full evaluate() with IndexPolicy completes without errors."""
    policy = IndexPolicy(
        **small_models,
        dataset=test_dataset,
        device=torch.device("cpu"),
    )
    shortest = ShortestFirstPolicy()

    evaluator = OfflineEvaluator(test_dataset)
    metrics = evaluator.evaluate(policy, baseline=shortest)

    assert metrics.n_instances == len(test_dataset)
    assert metrics.n_succeeded >= 0
    assert metrics.n_succeeded <= metrics.n_instances
    # TTFS should be finite for successful instances
    if metrics.n_succeeded > 0:
        assert metrics.mean_ttfs < float("inf")
        assert metrics.median_ttfs < float("inf")
