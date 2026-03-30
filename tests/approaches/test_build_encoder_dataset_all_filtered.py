"""Smoke / integration tests for the all_filtered vocabulary-building pipeline.

These tests exercise the full Stage A → B → C pipeline by calling the same
helper functions that build_encoder_dataset.py uses, but with a tiny o1
environment and minimal seed counts so they complete in < 60 s.

All tests are marked ``slow`` and are therefore skipped by default; run with::

    pytest -m slow tests/approaches/test_build_encoder_dataset_all_filtered.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dill
import kinder
import numpy as np
import pytest
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.encoder_approach import EncoderApproach
from alphatamp.structs import FrozenGroundOpSequence

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_ENV_ID = "kinder/Obstruction2D-o1-v0"
_MODEL_NAME = "obstruction2d"
_NUM_OBSTRUCTIONS = 1

# Tiny parameters tuned for speed (< 60 s total for both tests).
_APPROACH_KWARGS: dict[str, Any] = {
    "env_id": _ENV_ID,
    "model_name": _MODEL_NAME,
    "num_obstructions": _NUM_OBSTRUCTIONS,
    "max_abstract_plans": 5,
    "samples_per_step": 2,
    "max_skill_horizon": 50,
    "num_training_skeletons_per_problem": 5,
    "training_planning_timeout": 3.0,
    "vocabulary_size": 5,
}

_VOCAB_SEEDS = list(range(0, 4))  # 4 seeds for vocab collection
_FILTER_SEEDS = list(range(4, 7))  # 3 seeds for filter stage (separate range)


def _build_approach() -> EncoderApproach:  # type: ignore[type-arg]
    """Construct a fresh EncoderApproach for the o1 environment."""
    kinder.register_all_environments()
    env = kinder.make(_ENV_ID)
    try:
        obs, _ = env.reset(seed=0)
        del obs
        env_models = create_bilevel_planning_models(
            _MODEL_NAME,
            env.observation_space,
            env.action_space,
            num_obstructions=_NUM_OBSTRUCTIONS,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]

    return EncoderApproach(
        env_models=env_models,
        seed=0,
        **{
            k: v
            for k, v in _APPROACH_KWARGS.items()
            if k not in ("model_name", "num_obstructions")
        },
    )


# ---------------------------------------------------------------------------
# Stage A: full vocab collection (uncapped)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_filtered_stage_a_full_vocab() -> None:
    """Stage A: build_full_vocab collects counts for all observed sequences."""
    approach = _build_approach()
    approach.build_full_vocab(_VOCAB_SEEDS)
    counts = approach.get_op_sequence_counts()

    assert len(counts) > 0, "Expected at least one op-sequence to be observed"
    assert all(v > 0 for v in counts.values())

    # Sort descending by frequency — this is what all_filtered does.
    full_vocab: list[FrozenGroundOpSequence] = sorted(
        counts, key=lambda seq: -counts[seq]
    )
    approach.set_vocab(full_vocab)
    assert approach.get_op_sequence_vocabulary() == full_vocab


# ---------------------------------------------------------------------------
# Stage B: filter-seed reference dataset (simulator run)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_filtered_stage_b_filter_dataset_shape() -> None:
    """Stage B: filter-seed dataset has shape (n_filter_seeds, n_full_vocab)."""
    approach = _build_approach()

    # Stage A
    approach.build_full_vocab(_VOCAB_SEEDS)
    counts = approach.get_op_sequence_counts()
    full_vocab: list[FrozenGroundOpSequence] = sorted(
        counts, key=lambda seq: -counts[seq]
    )
    approach.set_vocab(full_vocab)

    # Stage B
    filter_dataset = approach.build_dataset(_FILTER_SEEDS, show_progress=False)
    applicability = cast(np.ndarray[Any, Any], filter_dataset["applicability"])
    success = cast(np.ndarray[Any, Any], filter_dataset["success"])
    refinement_time = cast(np.ndarray[Any, Any], filter_dataset["refinement_time"])
    steps = cast(np.ndarray[Any, Any], filter_dataset["steps_completed_fraction"])
    skeleton_lengths = cast(np.ndarray[Any, Any], filter_dataset["skeleton_lengths"])
    seed_ids = cast(list[int], filter_dataset["seed_ids"])
    op_sequence_vocab = cast(
        list[FrozenGroundOpSequence], filter_dataset["op_sequence_vocab"]
    )

    n_filter = len(_FILTER_SEEDS)
    n_vocab = len(full_vocab)

    assert applicability.shape == (n_filter, n_vocab)
    assert success.shape == (n_filter, n_vocab)
    assert refinement_time.shape == (n_filter, n_vocab)
    assert steps.shape == (n_filter, n_vocab)
    assert skeleton_lengths.shape == (n_vocab,)
    assert applicability.dtype == np.float32
    assert success.dtype == np.float32
    assert steps.dtype == np.float32
    assert skeleton_lengths.dtype == np.int16
    assert seed_ids == _FILTER_SEEDS
    assert op_sequence_vocab == full_vocab


# ---------------------------------------------------------------------------
# End-to-end: Stages A + B + C with output validation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_filtered_end_to_end(tmp_path: Path) -> None:
    """Full A → B → C pipeline: outputs have correct shapes, types, and alignment."""
    approach = _build_approach()

    # ---- Stage A ----
    approach.build_full_vocab(_VOCAB_SEEDS)
    counts = approach.get_op_sequence_counts()
    full_vocab: list[FrozenGroundOpSequence] = sorted(
        counts, key=lambda seq: -counts[seq]
    )
    approach.set_vocab(full_vocab)
    n_full = len(full_vocab)

    # ---- Stage B ----
    filter_dataset = approach.build_dataset(_FILTER_SEEDS, show_progress=False)
    applicability = cast(np.ndarray[Any, Any], filter_dataset["applicability"])
    success = cast(np.ndarray[Any, Any], filter_dataset["success"])
    assert applicability.shape == (len(_FILTER_SEEDS), n_full)

    # ---- Stage C ----
    threshold = 0.0
    filtered_vocab, keep_indices, stats = EncoderApproach.filter_vocab_by_success_rate(
        filter_dataset, threshold
    )
    filtered_dataset = EncoderApproach.apply_vocab_filter_to_dataset(
        filter_dataset, keep_indices
    )

    n_filtered = len(filtered_vocab)

    # --- stats sanity ---
    assert stats["original_size"] == n_full
    assert stats["filtered_size"] == n_filtered
    assert stats["removed_count"] == n_full - n_filtered
    assert stats["threshold"] == threshold
    assert len(stats["success_rates"]) == n_full

    # --- filtered vocab must be a strict subset, in success_rate desc order ---
    assert set(filtered_vocab).issubset(set(full_vocab))
    assert len(keep_indices) == n_filtered
    assert all(0 <= i < n_full for i in keep_indices)

    # success_rate must be > threshold (> 0.0 means at least one success)
    app = np.asarray(applicability)
    suc = np.asarray(success)
    for col_idx in keep_indices:
        applicable_count = int(app[:, col_idx].sum())
        success_count = int(suc[:, col_idx].sum())
        assert applicable_count > 0, f"kept col {col_idx} was never applicable"
        assert success_count > 0, f"kept col {col_idx} had zero successes"

    # --- filtered dataset shapes ---
    filtered_app = cast(np.ndarray[Any, Any], filtered_dataset["applicability"])
    filtered_success = cast(np.ndarray[Any, Any], filtered_dataset["success"])
    filtered_time = cast(np.ndarray[Any, Any], filtered_dataset["refinement_time"])
    filtered_steps = cast(
        np.ndarray[Any, Any], filtered_dataset["steps_completed_fraction"]
    )
    filtered_lengths = cast(np.ndarray[Any, Any], filtered_dataset["skeleton_lengths"])
    filtered_vocab_cast = cast(
        list[FrozenGroundOpSequence], filtered_dataset["op_sequence_vocab"]
    )
    filtered_seed_ids = cast(list[int], filtered_dataset["seed_ids"])

    assert filtered_app.shape == (len(_FILTER_SEEDS), n_filtered)
    assert filtered_success.shape == (len(_FILTER_SEEDS), n_filtered)
    assert filtered_time.shape == (len(_FILTER_SEEDS), n_filtered)
    assert filtered_steps.shape == (len(_FILTER_SEEDS), n_filtered)
    assert filtered_lengths.shape == (n_filtered,)

    # --- filtered dataset columns must equal the sliced original columns ---
    np.testing.assert_array_equal(
        filtered_app,
        applicability[:, keep_indices],
    )
    np.testing.assert_array_equal(
        filtered_success,
        success[:, keep_indices],
    )
    np.testing.assert_array_equal(
        filtered_steps,
        np.asarray(filter_dataset["steps_completed_fraction"])[:, keep_indices],
    )
    np.testing.assert_array_equal(
        filtered_lengths,
        np.asarray(filter_dataset["skeleton_lengths"])[keep_indices],
    )

    # --- vocab alignment: filtered_dataset vocab must match filtered_vocab ---
    assert filtered_vocab_cast == filtered_vocab

    # --- seed_ids preserved ---
    assert filtered_seed_ids == _FILTER_SEEDS

    # --- success_rate ordering: must be non-increasing ---
    if n_filtered > 1:
        rates = [
            float(suc[:, keep_indices[i]].sum())
            / float(max(app[:, keep_indices[i]].sum(), 1))
            for i in range(n_filtered)
        ]
        assert rates == sorted(
            rates, reverse=True
        ), f"keep_indices not sorted by success_rate descending: {rates}"

    # --- optionally snapshot outputs to tmp_path (exercises _save_pickle path) ---

    out_filter = tmp_path / "encoder_filter_dataset.pkl"
    out_filtered_vocab = tmp_path / "encoder_vocab_filtered_train.pkl"
    out_filtered_dataset = tmp_path / "encoder_filter_dataset_filtered.pkl"

    with open(out_filter, "wb") as f:
        dill.dump({"split": "filter", "dataset": filter_dataset}, f)
    with open(out_filtered_vocab, "wb") as f:
        dill.dump(
            {
                "vocabulary": filtered_vocab,
                "keep_indices": keep_indices,
                "filter_stats": stats,
            },
            f,
        )
    with open(out_filtered_dataset, "wb") as f:
        dill.dump({"split": "filter_filtered", "dataset": filtered_dataset}, f)

    # Reload and verify round-trip.
    with open(out_filtered_vocab, "rb") as f:
        reloaded = dill.load(f)
    assert reloaded["vocabulary"] == filtered_vocab
    assert reloaded["keep_indices"] == keep_indices

    assert out_filter.stat().st_size > 0
    assert out_filtered_vocab.stat().st_size > 0
    assert out_filtered_dataset.stat().st_size > 0
