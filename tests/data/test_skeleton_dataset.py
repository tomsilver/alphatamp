"""Tests for src/alphatamp/data/skeleton_dataset.py.

Covers:
- Round-trip: build_synthetic_dataset → write → SkeletonDataset → verify values
- Dataset attributes (skeleton_lengths, op_sequences, vocabs)
- __getitem__ shape, dtype, type
- Collate function and DataLoader integration
- preload=True vs preload=False produce identical results
- Edge cases: L=0 skeleton, all-inapplicable row, single instance/skeleton
- Invariant violation detection via validate_skeleton_dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.utils.data

# Ensure project src is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from build_synthetic_dataset import (
    _SyntheticObj,
    _SyntheticOp,
    _SyntheticType,
    build_synthetic_dataset,
)
from validate_skeleton_dataset import validate_skeleton_dataset

from alphatamp.data.skeleton_dataset import (
    OpSequenceTokens,
    SkeletonBatch,
    SkeletonDataset,
    SkeletonItem,
    skeleton_collate_fn,
    write_skeleton_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_dict():
    """Small synthetic dataset dict (N=10, M=5) for fast round-trip tests."""
    return build_synthetic_dataset(N=10, M=5, rng_seed=0)


@pytest.fixture
def hdf5_path(tmp_path, synthetic_dict):
    """Write synthetic_dict to a temp HDF5 file and return the path."""
    path = tmp_path / "test.h5"
    write_skeleton_dataset(path, synthetic_dict)
    return path


# ---------------------------------------------------------------------------
# Round-trip tests: writer → reader
# ---------------------------------------------------------------------------


def test_write_and_load_shapes(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    N, M = synthetic_dict["applicability"].shape
    assert ds.N == N
    assert ds.M == M
    assert len(ds) == N
    assert ds.skeleton_lengths.shape == (M,)
    assert len(ds.op_sequences) == M


def test_write_and_load_applicability(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    expected = torch.from_numpy(synthetic_dict["applicability"])
    for i in range(ds.N):
        torch.testing.assert_close(ds[i].applicability, expected[i])


def test_write_and_load_success(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    expected = torch.from_numpy(synthetic_dict["success"])
    for i in range(ds.N):
        torch.testing.assert_close(ds[i].success, expected[i])


def test_write_and_load_steps(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    expected = torch.from_numpy(synthetic_dict["steps_completed_fraction"])
    for i in range(ds.N):
        torch.testing.assert_close(ds[i].steps_completed_fraction, expected[i])


def test_write_and_load_time(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    expected = torch.from_numpy(synthetic_dict["refinement_time"])
    for i in range(ds.N):
        torch.testing.assert_close(ds[i].refinement_time, expected[i])


def test_write_and_load_seed_ids(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    expected_seeds = list(synthetic_dict["seed_ids"])
    loaded_seeds = [ds[i].seed_id for i in range(ds.N)]
    assert loaded_seeds == expected_seeds


# ---------------------------------------------------------------------------
# Dataset attribute tests
# ---------------------------------------------------------------------------


def test_skeleton_lengths_attribute(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    vocab = synthetic_dict["op_sequence_vocab"]
    expected = torch.tensor([len(seq) for seq in vocab], dtype=torch.int32)
    torch.testing.assert_close(ds.skeleton_lengths, expected)


def test_op_sequences_decoded_correctly(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    vocab = synthetic_dict["op_sequence_vocab"]

    # Check each skeleton's sequence
    for j, seq in enumerate(vocab):
        tok = ds.op_sequences[j]
        assert tok.length == len(seq)
        assert tok.op_type_ids.shape == (tok.length,)
        if tok.length > 0:
            assert tok.obj_ids.shape[0] == tok.length
            assert tok.type_ids.shape[0] == tok.length

        # Check op type names map back correctly
        for k, op in enumerate(seq):
            op_name = ds.op_type_vocab[int(tok.op_type_ids[k])]
            assert (
                op_name == op.name
            ), f"skeleton {j} op {k}: expected {op.name}, got {op_name}"

        # Check object names for first op (if any)
        if tok.length > 0 and len(seq[0].parameters) > 0:
            first_op = seq[0]
            for p_idx, param in enumerate(first_op.parameters):
                obj_id = int(tok.obj_ids[0, p_idx])
                assert obj_id >= 0
                assert ds.obj_vocab[obj_id] == param.name


def test_vocabs_are_sorted_strings(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    assert ds.op_type_vocab == sorted(ds.op_type_vocab)
    assert ds.obj_vocab == sorted(ds.obj_vocab)
    assert ds.type_vocab == sorted(ds.type_vocab)


# ---------------------------------------------------------------------------
# __getitem__ tests
# ---------------------------------------------------------------------------


def test_getitem_returns_skeleton_item(hdf5_path):
    ds = SkeletonDataset(hdf5_path)
    item = ds[0]
    assert isinstance(item, SkeletonItem)


def test_getitem_tensor_shapes(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    M = ds.M
    item = ds[0]
    assert item.applicability.shape == (M,)
    assert item.success.shape == (M,)
    assert item.steps_completed_fraction.shape == (M,)
    assert item.refinement_time.shape == (M,)


def test_getitem_dtype(hdf5_path):
    ds = SkeletonDataset(hdf5_path)
    item = ds[0]
    assert item.applicability.dtype == torch.float32
    assert item.success.dtype == torch.float32
    assert item.steps_completed_fraction.dtype == torch.float32
    assert item.refinement_time.dtype == torch.float32


def test_getitem_seed_id_is_int(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    item = ds[0]
    assert isinstance(item.seed_id, int)
    assert item.seed_id == synthetic_dict["seed_ids"][0]


# ---------------------------------------------------------------------------
# Collate and DataLoader tests
# ---------------------------------------------------------------------------


def test_collate_fn_shapes(hdf5_path, synthetic_dict):
    ds = SkeletonDataset(hdf5_path)
    M = ds.M
    batch = [ds[i] for i in range(min(3, ds.N))]
    result = skeleton_collate_fn(batch)
    B = len(batch)
    assert isinstance(result, SkeletonBatch)
    assert result.applicability.shape == (B, M)
    assert result.success.shape == (B, M)
    assert result.steps_completed_fraction.shape == (B, M)
    assert result.refinement_time.shape == (B, M)
    assert len(result.seed_ids) == B


def test_collate_fn_seed_ids_is_list(hdf5_path):
    ds = SkeletonDataset(hdf5_path)
    batch = [ds[0], ds[1]]
    result = skeleton_collate_fn(batch)
    assert isinstance(result.seed_ids, list)


def test_dataloader_iteration(hdf5_path):
    ds = SkeletonDataset(hdf5_path)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=3,
        collate_fn=skeleton_collate_fn,
        num_workers=0,
    )
    batches = list(loader)
    assert len(batches) > 0
    total_items = sum(len(b.seed_ids) for b in batches)
    assert total_items == ds.N


# ---------------------------------------------------------------------------
# Preload mode tests
# ---------------------------------------------------------------------------


def test_preload_same_values(hdf5_path, synthetic_dict):
    ds_lazy = SkeletonDataset(hdf5_path, preload=False)
    ds_preload = SkeletonDataset(hdf5_path, preload=True)

    for i in range(ds_lazy.N):
        item_lazy = ds_lazy[i]
        item_pre = ds_preload[i]
        torch.testing.assert_close(item_lazy.applicability, item_pre.applicability)
        torch.testing.assert_close(item_lazy.success, item_pre.success)
        torch.testing.assert_close(
            item_lazy.steps_completed_fraction, item_pre.steps_completed_fraction
        )
        torch.testing.assert_close(item_lazy.refinement_time, item_pre.refinement_time)
        assert item_lazy.seed_id == item_pre.seed_id


def test_preload_skeleton_lengths_same(hdf5_path):
    ds_lazy = SkeletonDataset(hdf5_path, preload=False)
    ds_preload = SkeletonDataset(hdf5_path, preload=True)
    torch.testing.assert_close(ds_lazy.skeleton_lengths, ds_preload.skeleton_lengths)


# ---------------------------------------------------------------------------
# Edge case: L=0 skeleton
# ---------------------------------------------------------------------------


def _make_minimal_dict(N=2, M=3):
    """Create a minimal valid dataset dict with all-zero matrices."""
    _BLOCK = _SyntheticType("block")
    _OBJ0 = _SyntheticObj("obj0", _BLOCK)

    op_sequence_vocab = [
        (
            _SyntheticOp("Pick", (_OBJ0,)),
            _SyntheticOp("Place", (_OBJ0, _OBJ0)),
        ),
        (_SyntheticOp("Stack", (_OBJ0, _OBJ0)),),
        (_SyntheticOp("Pick", (_OBJ0,)),),
    ][:M]

    return {
        "seed_ids": list(range(N)),
        "op_sequence_vocab": op_sequence_vocab,
        "applicability": np.zeros((N, M), dtype=np.float32),
        "success": np.zeros((N, M), dtype=np.float32),
        "steps_completed_fraction": np.zeros((N, M), dtype=np.float32),
        "refinement_time": np.zeros((N, M), dtype=np.float32),
        "skeleton_lengths": np.array(
            [len(s) for s in op_sequence_vocab], dtype=np.int16
        ),
    }


def _make_dict_with_L0_skeleton(tmp_path):
    """Dataset with one L=0 skeleton (no operators)."""
    _BLOCK = _SyntheticType("block")
    _OBJ0 = _SyntheticObj("obj0", _BLOCK)

    op_sequence_vocab = [
        (),  # L=0
        (_SyntheticOp("Pick", (_OBJ0,)),),  # L=1
    ]

    N, M = 4, 2
    return {
        "seed_ids": list(range(N)),
        "op_sequence_vocab": op_sequence_vocab,
        "applicability": np.ones((N, M), dtype=np.float32),
        "success": np.zeros((N, M), dtype=np.float32),
        "steps_completed_fraction": np.zeros((N, M), dtype=np.float32),
        "refinement_time": np.array([[0.1, 0.2]] * N, dtype=np.float32),
        "skeleton_lengths": np.array([0, 1], dtype=np.int16),
    }


def test_empty_skeleton_L0_writes_and_loads(tmp_path):
    d = _make_dict_with_L0_skeleton(tmp_path)
    path = tmp_path / "l0.h5"
    write_skeleton_dataset(path, d)

    ds = SkeletonDataset(path)
    assert ds.skeleton_lengths[0] == 0
    assert ds.op_sequences[0].length == 0
    assert ds.op_sequences[0].op_type_ids.shape == (0,)


def test_empty_skeleton_L0_validator_passes(tmp_path):
    d = _make_dict_with_L0_skeleton(tmp_path)
    path = tmp_path / "l0.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert len(summary["violations"]) == 0


# ---------------------------------------------------------------------------
# Edge case: all inapplicable row
# ---------------------------------------------------------------------------


def test_all_inapplicable_row(tmp_path):
    d = _make_minimal_dict(N=3, M=3)
    # Row 1: all inapplicable — all zeros already; valid
    path = tmp_path / "all_inapp.h5"
    write_skeleton_dataset(path, d)

    ds = SkeletonDataset(path)
    item = ds[1]
    assert torch.all(item.applicability == 0.0)
    assert torch.all(item.success == 0.0)
    assert torch.all(item.steps_completed_fraction == 0.0)
    assert torch.all(item.refinement_time == 0.0)


def test_single_skeleton_M1(tmp_path):
    d = build_synthetic_dataset(N=5, M=1, rng_seed=7)
    path = tmp_path / "m1.h5"
    write_skeleton_dataset(path, d)
    ds = SkeletonDataset(path)
    assert ds.M == 1
    item = ds[0]
    assert item.applicability.shape == (1,)


def test_single_instance_N1(tmp_path):
    d = build_synthetic_dataset(N=1, M=4, rng_seed=8)
    path = tmp_path / "n1.h5"
    write_skeleton_dataset(path, d)
    ds = SkeletonDataset(path)
    assert ds.N == 1
    assert len(ds) == 1
    item = ds[0]
    assert item.applicability.shape == (4,)


# ---------------------------------------------------------------------------
# Invariant violation detection
# ---------------------------------------------------------------------------


def test_validator_passes_valid_synthetic_data(hdf5_path):
    summary = validate_skeleton_dataset(hdf5_path, strict=False)
    assert len(summary["violations"]) == 0, summary["violations"]


def test_validator_passes_larger_synthetic_data(tmp_path):
    d = build_synthetic_dataset(N=500, M=20, rng_seed=42)
    path = tmp_path / "large.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert len(summary["violations"]) == 0, summary["violations"]


def test_validator_catches_Y_gt_A(tmp_path):
    d = build_synthetic_dataset(N=10, M=5, rng_seed=1)
    # Introduce Y=1 where A=0
    d["success"][0, 0] = 1.0
    d["applicability"][0, 0] = 0.0
    d["steps_completed_fraction"][0, 0] = 0.0  # keep F consistent
    path = tmp_path / "bad_y_gt_a.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert any("Y" in v or "success" in v.lower() for v in summary["violations"])


def test_validator_catches_F_gt_0_when_inapplicable(tmp_path):
    d = build_synthetic_dataset(N=10, M=5, rng_seed=2)
    # Find an inapplicable entry and set F > 0
    inapplicable = d["applicability"] < 0.5
    if not np.any(inapplicable):
        d["applicability"][0, 0] = 0.0
    row, col = np.argwhere(d["applicability"] < 0.5)[0]
    d["steps_completed_fraction"][row, col] = 0.5
    path = tmp_path / "bad_f_inapplicable.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert any(
        "F" in v or "step" in v.lower() or "inapplicable" in v.lower()
        for v in summary["violations"]
    )


def test_validator_catches_F_not_K_over_L(tmp_path):
    d = build_synthetic_dataset(N=10, M=5, rng_seed=3)
    # Find an applicable entry and set F to a value not representable as K/L
    app = d["applicability"] > 0.5
    L_arr = d["skeleton_lengths"]
    # Find a skeleton with L > 1 that has an applicable entry
    for j in range(d["applicability"].shape[1]):
        L_j = int(L_arr[j])
        if L_j > 1 and np.any(d["applicability"][:, j] > 0.5):
            row = np.argwhere(d["applicability"][:, j] > 0.5)[0, 0]
            # Set F to something between 0 and 1 but not K/L
            d["steps_completed_fraction"][row, j] = 0.7 / L_j  # not an integer multiple
            d["success"][row, j] = 0.0  # ensure not Y=1
            break

    path = tmp_path / "bad_f_kl.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert any(
        "K/L" in v or "F=" in v or "error" in v.lower() for v in summary["violations"]
    )


def test_validator_catches_F_not_1_when_Y_1(tmp_path):
    d = build_synthetic_dataset(N=10, M=5, rng_seed=4)
    # Find a successful entry and set F < 1
    suc = d["success"] > 0.5
    if not np.any(suc):
        # Force one success
        d["applicability"][0, 0] = 1.0
        d["success"][0, 0] = 1.0
        d["steps_completed_fraction"][0, 0] = 1.0
        d["refinement_time"][0, 0] = 0.5
        suc = d["success"] > 0.5

    row, col = np.argwhere(suc)[0]
    d["steps_completed_fraction"][row, col] = 0.5  # Y=1 but F=0.5 → violation
    path = tmp_path / "bad_y1_f_lt_1.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert any(
        "F" in v or "Y=1" in v or "success" in v.lower() for v in summary["violations"]
    )


def test_validator_catches_T_nonzero_inapplicable(tmp_path):
    d = build_synthetic_dataset(N=10, M=5, rng_seed=5)
    # Find inapplicable entry and set T > 0
    inapplicable = d["applicability"] < 0.5
    if not np.any(inapplicable):
        d["applicability"][0, 0] = 0.0
    row, col = np.argwhere(d["applicability"] < 0.5)[0]
    d["refinement_time"][row, col] = 1.5
    path = tmp_path / "bad_t_inapplicable.h5"
    write_skeleton_dataset(path, d)
    summary = validate_skeleton_dataset(path, strict=False)
    assert any(
        "T" in v or "time" in v.lower() or "inapplicable" in v.lower()
        for v in summary["violations"]
    )


def test_validator_strict_raises_on_violation(tmp_path):
    d = build_synthetic_dataset(N=10, M=5, rng_seed=6)
    d["success"][0, 0] = 1.0
    d["applicability"][0, 0] = 0.0
    d["steps_completed_fraction"][0, 0] = 0.0
    path = tmp_path / "strict_bad.h5"
    write_skeleton_dataset(path, d)
    with pytest.raises(AssertionError):
        validate_skeleton_dataset(path, strict=True)


# ---------------------------------------------------------------------------
# Compression / no-compression round-trip
# ---------------------------------------------------------------------------


def test_write_no_compression(tmp_path, synthetic_dict):
    path = tmp_path / "nocomp.h5"
    write_skeleton_dataset(path, synthetic_dict, compression=None)
    ds = SkeletonDataset(path)
    assert ds.N == 10
    assert ds.M == 5


# ---------------------------------------------------------------------------
# skeleton_lengths computed from vocab when absent in dict
# ---------------------------------------------------------------------------


def test_writer_computes_skeleton_lengths_if_absent(tmp_path, synthetic_dict):
    d = {k: v for k, v in synthetic_dict.items() if k != "skeleton_lengths"}
    assert "skeleton_lengths" not in d
    path = tmp_path / "no_lengths.h5"
    write_skeleton_dataset(path, d)
    ds = SkeletonDataset(path)
    vocab = synthetic_dict["op_sequence_vocab"]
    expected = torch.tensor([len(seq) for seq in vocab], dtype=torch.int32)
    torch.testing.assert_close(ds.skeleton_lengths, expected)
