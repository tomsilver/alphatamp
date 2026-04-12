"""Tests for SkeletonEncoder."""

from __future__ import annotations

import torch

from alphatamp.models.skeleton_encoder import SkeletonEncoder


def test_output_shape() -> None:
    """Forward pass produces (B, d_model) output with correct dtype."""
    num_op_types = 5
    num_objects = 8
    d_model = 64
    B, L, P = 3, 6, 3

    model = SkeletonEncoder(
        num_op_types=num_op_types,
        num_objects=num_objects,
        d_model=d_model,
    )
    model.eval()

    op_type_ids = torch.randint(0, num_op_types, (B, L))
    obj_ids = torch.randint(0, num_objects, (B, L, P))
    # Introduce some -1 padding in parameter slots
    obj_ids[:, :, -1] = -1
    lengths = torch.tensor([4, 6, 5])

    with torch.no_grad():
        out = model(op_type_ids, obj_ids, lengths)

    assert out.shape == (B, d_model)
    assert out.dtype == torch.float32


def test_object_relabeling_equivariance() -> None:
    """Consistently permuting object IDs + embedding rows gives same output."""
    num_op_types = 4
    num_objects = 6
    d_model = 32

    torch.manual_seed(42)
    model = SkeletonEncoder(
        num_op_types=num_op_types,
        num_objects=num_objects,
        d_model=d_model,
        dropout=0.0,
    )
    model.eval()

    # Deterministic input
    B, L, P = 2, 5, 3
    op_type_ids = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 0, 1, 3]]
    )
    obj_ids = torch.tensor(
        [
            [[0, 1, -1], [2, 3, -1], [4, 5, 0], [1, 2, -1], [3, 4, -1]],
            [[5, 0, -1], [1, 2, 3], [4, 5, -1], [0, 1, -1], [2, 3, 4]],
        ]
    )
    lengths = torch.tensor([5, 5])

    # (a) Original output
    with torch.no_grad():
        out_orig = model(op_type_ids, obj_ids, lengths).clone()

    # (b) Random permutation of object IDs
    perm = torch.randperm(num_objects)  # π: old_id → new_id

    # Relabel obj_ids
    new_obj_ids = obj_ids.clone()
    valid = obj_ids >= 0
    new_obj_ids[valid] = perm[obj_ids[valid]]

    # Permute embedding table rows accordingly:
    # new_weight[π(i)+1] = old_weight[i+1] for i in 0..num_objects-1
    old_weight = model.obj_embed.weight.data.clone()
    new_weight = torch.zeros_like(old_weight)
    # Row 0 stays zero (padding_idx)
    for i in range(num_objects):
        new_weight[perm[i] + 1] = old_weight[i + 1]
    model.obj_embed.weight.data.copy_(new_weight)

    with torch.no_grad():
        out_relabeled = model(op_type_ids, new_obj_ids, lengths)

    # Restore original weights
    model.obj_embed.weight.data.copy_(old_weight)

    assert torch.allclose(out_orig, out_relabeled, atol=1e-5), (
        f"Max diff: {(out_orig - out_relabeled).abs().max().item():.2e}"
    )


def test_operator_order_sensitivity() -> None:
    """Shuffling operator order must change the output embedding."""
    num_op_types = 6
    num_objects = 5
    d_model = 32

    torch.manual_seed(99)
    model = SkeletonEncoder(
        num_op_types=num_op_types,
        num_objects=num_objects,
        d_model=d_model,
        dropout=0.0,
    )
    model.eval()

    # Single sequence with distinct operators at each position
    L, P = 5, 2
    op_type_ids = torch.tensor([[0, 1, 2, 3, 4]])
    obj_ids = torch.tensor([[[0, 1], [2, 3], [4, 0], [1, 2], [3, 4]]])
    lengths = torch.tensor([L])

    with torch.no_grad():
        out_orig = model(op_type_ids, obj_ids, lengths)

    # Shuffle: reverse the operator order
    shuffle_perm = torch.tensor([4, 3, 2, 1, 0])
    shuffled_op_type_ids = op_type_ids[:, shuffle_perm]
    shuffled_obj_ids = obj_ids[:, shuffle_perm]

    with torch.no_grad():
        out_shuffled = model(shuffled_op_type_ids, shuffled_obj_ids, lengths)

    assert not torch.allclose(out_orig, out_shuffled, atol=1e-4), (
        "Shuffling operators should produce a different embedding"
    )
