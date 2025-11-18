"""Tests for saving and loading parameter dataset pickles."""

import pickle
from pathlib import Path

import pytest

from alphatamp.approaches.simfree_param_policy_approach import (
    SimFreeParamPolicyApproach,
)


FIXTURE_PATH = Path("tests/fixtures") / "parameter_dataset.pkl"


def test_write_parameter_dataset_pickle():
    """Write a small synthetic parameter dataset to a fixed fixture path."""
    sample = {"test_action": [((0.1, 0.2, 0.3), "success")]}

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("wb") as f:
        pickle.dump(sample, f)

    assert FIXTURE_PATH.exists()


def test_read_parameter_dataset_pickle():
    """Read the previously written pickle using the approach loader."""
    if not FIXTURE_PATH.exists():
        pytest.skip("parameter_dataset pickle not found; run write test first")

    loaded = SimFreeParamPolicyApproach.load_parameter_dataset(FIXTURE_PATH)

    assert loaded, "Loaded dataset should not be empty"
    assert "test_action" in loaded