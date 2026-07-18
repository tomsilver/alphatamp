"""Tests for saving and loading parameter dataset pickles for the
SimFreeParamPolicyApproach."""

import pickle

from alphatamp.approaches.simfree_param_policy_approach import (
    SimFreeParamPolicyApproach,
)


def test_parameter_dataset_pickle_roundtrip(tmp_path):
    """Write a small synthetic parameter dataset and read it back.

    Uses a per-test temp dir rather than a fixed path under tests/datasets/ so the test
    neither depends on execution order nor dirties the tracked fixture directory.
    """
    sample = {"test_action": [((0.1, 0.2, 0.3), "success")]}
    path = tmp_path / "test_parameter_dataset.pkl"
    with path.open("wb") as f:
        pickle.dump(sample, f)
    assert path.exists()

    loaded = SimFreeParamPolicyApproach.load_abstract_action_level_dataset(path)

    assert loaded, "Loaded dataset should not be empty"
    assert "test_action" in loaded
