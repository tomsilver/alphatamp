"""Unit tests for run_box_matrix_experiment helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from experiments import run_box_matrix_experiment as matrix_exp


def test_build_score_matrix_uses_accessor_and_returns_copy() -> None:
    """Experiment helper should consume BoxApproach accessor copy of D."""

    source = np.array([[0.25, 0.75], [0.0, 0.50]], dtype=float)

    approach = SimpleNamespace(
        get_score_matrix_copy=lambda: np.array(source, copy=True)
    )

    matrix = matrix_exp._build_score_matrix(approach)
    assert np.allclose(matrix, source)

    matrix[0, 0] = 99.0
    assert source[0, 0] == 0.25


def test_build_box_approach_for_level_passes_effort_config(monkeypatch) -> None:
    """Hydra effort fields should be passed through to BoxApproach constructor."""

    captured: dict[str, object] = {}

    def _fake_box_approach(env_models, **kwargs):  # noqa: ANN001
        captured["env_models"] = env_models
        captured.update(kwargs)
        return SimpleNamespace(**captured)

    monkeypatch.setattr(matrix_exp, "BoxApproach", _fake_box_approach)

    cfg = OmegaConf.create(
        {
            "seed": 123,
            "approach": {
                "samples_per_step": 10,
                "max_skill_horizon": 100,
                "heuristic_name": "hff",
                "skeleton_batch_size": 100,
                "num_training_skeletons_per_problem": 10,
                "exploration_constant": 1.41421356237,
                "training_label_mode": "effort",
                "failure_penalty_multiplier": 2.5,
            },
        }
    )

    env_models = object()
    built = matrix_exp._build_box_approach_for_level(
        cfg,
        env_models,
        level_timeout=42.0,
        level_max_abstract_plans=17,
    )

    assert built.env_models is env_models
    assert built.max_abstract_plans == 17
    assert built.training_planning_timeout == 42.0
    assert built.training_label_mode == "effort"
    assert built.failure_penalty_multiplier == 2.5


def test_compute_matrix_summary_effort_mode_uses_continuous_metrics() -> None:
    """Effort mode should report continuous diagnostics instead of binary entropy."""

    matrix = np.array(
        [
            [0.25, 0.90, 0.50],
            [0.75, 0.20, 0.10],
            [0.10, 0.85, 0.60],
        ],
        dtype=float,
    )

    summary = matrix_exp._compute_matrix_summary(
        level="o1",
        num_obstructions=1,
        num_train_seeds=3,
        training_label_mode="effort",
        matrix=matrix,
    )

    assert summary.diagnostics_mode == "effort"
    assert summary.mean_row_l1_distance > 0.0
    assert summary.mean_row_l2_distance > 0.0
    assert summary.col_variance_mean > 0.0
    assert summary.mean_row_hamming_distance == 0.0
    assert summary.col_entropy_mean == 0.0


def test_compute_matrix_summary_binary_mode_keeps_binary_metrics() -> None:
    """Binary mode should keep hamming distance and Bernoulli entropy metrics."""

    matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    summary = matrix_exp._compute_matrix_summary(
        level="o1",
        num_obstructions=1,
        num_train_seeds=3,
        training_label_mode="binary",
        matrix=matrix,
    )

    assert summary.diagnostics_mode == "binary"
    assert summary.mean_row_hamming_distance > 0.0
    assert summary.col_entropy_mean > 0.0
