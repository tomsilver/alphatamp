"""Tests for ``collate_spectre_batch``: shapes, masks, vocab lookup."""

from __future__ import annotations

from pathlib import Path

import torch
from _fixtures import write_toy_split

from alphatamp.approaches.spectre.dataset import (
    SpectreDataset,
    collate_spectre_batch,
)
from alphatamp.approaches.spectre.priors import ZeroPrior
from alphatamp.approaches.spectre.vocab import extract_vocab


def _fixture(tmp_path: Path):
    train = tmp_path / "train"
    write_toy_split(
        train,
        outcomes_per_problem=[
            ("fail", "fail", "success"),
            ("success", "fail", "fail"),
            ("fail", "success", "fail", "fail"),
        ],
    )
    vocab = extract_vocab(train, config_hash="abc")
    ds = SpectreDataset(
        split_dir=train,
        prior=ZeroPrior(),
        seed=1234,
        augment=False,
    )
    return ds, vocab


def test_collate_shapes(tmp_path: Path) -> None:
    """Batch dimensions match spec."""
    ds, vocab = _fixture(tmp_path)
    batch = [ds[i] for i in range(len(ds))]
    b = len(batch)
    batched = collate_spectre_batch(batch, vocab)

    assert batched.r_op_ids.dim() == 3  # (B, R, L)
    assert batched.r_op_ids.shape[0] == b
    assert batched.r_op_arg_type_ids.shape[:3] == batched.r_op_ids.shape
    assert (
        batched.r_mask.shape == batched.r_priors.shape == batched.r_success_mask.shape
    )
    assert batched.f_mask.shape[0] == b
    assert batched.problem_ids.shape == (b,)


def test_mask_true_where_token_present(tmp_path: Path) -> None:
    """``r_op_mask`` is True for real operators and False for padding."""
    ds, vocab = _fixture(tmp_path)
    batch = [ds[0]]
    out = collate_spectre_batch(batch, vocab)
    # Skeleton 0 in example 0 has 2 operators (Pick, Place).
    # All real ops should have op_ids != 0.
    real_mask = out.r_op_ids[0, 0] != 0
    # Mask aligns with real_mask.
    assert torch.equal(out.r_op_mask[0, 0], real_mask)


def test_type_histogram_sums_to_num_objects(tmp_path: Path) -> None:
    """``s0_type_histogram`` rows sum to the number of objects in the problem."""
    ds, vocab = _fixture(tmp_path)
    batch = [ds[i] for i in range(len(ds))]
    out = collate_spectre_batch(batch, vocab)
    for i, ex in enumerate(batch):
        assert int(out.s0_type_histogram[i].sum()) == len(ex.object_registry)


def test_empty_f_still_collates(tmp_path: Path) -> None:
    """Examples with empty F produce width-1 F tensors (no crash)."""
    ds, vocab = _fixture(tmp_path)
    # Force an F=∅ example by picking an RNG that yields no selections.
    # We do this by just taking many samples and looking for one with empty F.
    found = False
    for _ in range(50):
        for i in range(len(ds)):
            ex = ds[i]
            if len(ex.f_skeletons) == 0:
                out = collate_spectre_batch([ex], vocab)
                assert out.f_op_ids.shape[1] == 1
                assert not out.f_mask[0, 0].item()
                found = True
                break
        if found:
            break
    assert found, "Expected to observe at least one empty-F sample in 50 draws"
