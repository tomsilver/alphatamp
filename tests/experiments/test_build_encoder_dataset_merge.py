"""Unit tests for dataset merge utilities in build_encoder_dataset.py."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from experiments.build_encoder_dataset import _merge_partial_datasets
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"Skipping build_encoder_dataset merge tests due to import error: {exc}",
        allow_module_level=True,
    )


def test_merge_partial_datasets_preserves_all_encoder_matrices() -> None:
    """Merged output should retain all MAE-relevant matrix fields."""
    vocab = [("op0",), ("op1",), ("op2",)]
    skeleton_lengths = np.array([2, 3, 1], dtype=np.int16)

    part0 = {
        "seed_ids": [10, 11],
        "op_sequence_vocab": vocab,
        "applicability": np.array([[1, 0, 1], [1, 1, 0]], dtype=np.float32),
        "success": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "refinement_time": np.array(
            [[0.3, 5.0, 0.9], [0.6, 1.2, 5.0]], dtype=np.float32
        ),
        "steps_completed_fraction": np.array(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        "skeleton_lengths": skeleton_lengths,
        "initial_low_level_states": ["x10", "x11"],
        "initial_abstract_states": ["s10", "s11"],
        "problem_goals": ["g10", "g11"],
    }

    part1 = {
        "seed_ids": [12],
        "op_sequence_vocab": vocab,
        "applicability": np.array([[0, 1, 1]], dtype=np.float32),
        "success": np.array([[0, 0, 1]], dtype=np.float32),
        "refinement_time": np.array([[5.0, 0.4, 1.5]], dtype=np.float32),
        "steps_completed_fraction": np.array([[0.0, 0.25, 1.0]], dtype=np.float32),
        "skeleton_lengths": skeleton_lengths,
        "initial_low_level_states": ["x12"],
        "initial_abstract_states": ["s12"],
        "problem_goals": ["g12"],
    }

    merged = _merge_partial_datasets([part0, part1])

    assert merged["seed_ids"] == [10, 11, 12]
    assert merged["op_sequence_vocab"] == vocab
    np.testing.assert_array_equal(merged["skeleton_lengths"], skeleton_lengths)

    np.testing.assert_array_equal(
        merged["applicability"],
        np.concatenate([part0["applicability"], part1["applicability"]], axis=0),
    )
    np.testing.assert_array_equal(
        merged["success"],
        np.concatenate([part0["success"], part1["success"]], axis=0),
    )
    np.testing.assert_array_equal(
        merged["refinement_time"],
        np.concatenate([part0["refinement_time"], part1["refinement_time"]], axis=0),
    )
    np.testing.assert_array_equal(
        merged["steps_completed_fraction"],
        np.concatenate(
            [part0["steps_completed_fraction"], part1["steps_completed_fraction"]],
            axis=0,
        ),
    )

    assert merged["initial_low_level_states"] == ["x10", "x11", "x12"]
    assert merged["initial_abstract_states"] == ["s10", "s11", "s12"]
    assert merged["problem_goals"] == ["g10", "g11", "g12"]
