"""Run BOX score-matrix diagnostics across Obstruction2D complexity levels.

This script trains one BOX model per complexity level, then extracts the
backfilled score matrix D and computes diagnostics over D.

Example:
    python experiments/run_box_matrix_experiment.py
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import hydra
import numpy as np
import prbench
from omegaconf import DictConfig
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.box_approach import BoxApproach


@dataclass(frozen=True)
class MatrixSummary:
    """Summary statistics for one trained BOX score matrix."""

    level: str
    num_obstructions: int
    num_train_seeds: int
    training_label_mode: str
    diagnostics_mode: str
    num_rows: int
    vocab_size: int
    rank: int
    effective_rank: float
    stable_rank: float
    duplicate_row_ratio: float
    mean_row_hamming_distance: float
    mean_row_l1_distance: float
    mean_row_l2_distance: float
    avg_abs_column_correlation: float
    col_mean_mean: float
    col_mean_std: float
    col_mean_min: float
    col_mean_max: float
    col_entropy_mean: float
    col_entropy_std: float
    col_entropy_min: float
    col_entropy_max: float
    col_variance_mean: float
    col_variance_std: float
    col_variance_min: float
    col_variance_max: float
    row_mean_mean: float
    row_mean_std: float
    row_mean_min: float
    row_mean_max: float


def _get_complexity_config(cfg: DictConfig, level: str) -> tuple[float, int]:
    """Get per-level timeout and max_abstract_plans (with approach defaults)."""
    timeout = float(cfg.approach.training_planning_timeout)
    max_abstract_plans = int(cfg.approach.max_abstract_plans)

    if "complexity_configs" not in cfg:
        return timeout, max_abstract_plans

    if level not in cfg.complexity_configs:
        return timeout, max_abstract_plans

    level_cfg = cfg.complexity_configs[level]
    if "timeout" in level_cfg:
        timeout = float(level_cfg.timeout)
    if "max_abstract_plans" in level_cfg:
        max_abstract_plans = int(level_cfg.max_abstract_plans)

    return timeout, max_abstract_plans


def _build_box_approach_for_level(
    cfg: DictConfig,
    env_models: Any,
    level_timeout: float,
    level_max_abstract_plans: int,
) -> BoxApproach:
    """Create a BoxApproach with per-level complexity settings."""
    return BoxApproach(
        env_models,
        seed=int(cfg.seed),
        max_abstract_plans=level_max_abstract_plans,
        samples_per_step=int(cfg.approach.samples_per_step),
        max_skill_horizon=int(cfg.approach.max_skill_horizon),
        heuristic_name=str(cfg.approach.heuristic_name),
        skeleton_batch_size=int(cfg.approach.skeleton_batch_size),
        num_training_skeletons_per_problem=int(
            cfg.approach.num_training_skeletons_per_problem
        ),
        training_planning_timeout=level_timeout,
        exploration_constant=float(cfg.approach.exploration_constant),
        training_label_mode=str(cfg.approach.training_label_mode),
        failure_penalty_multiplier=float(cfg.approach.failure_penalty_multiplier),
    )


def _build_score_matrix(approach: BoxApproach) -> np.ndarray:
    """Fetch score matrix D from BoxApproach as a defensive copy."""
    return approach.get_score_matrix_copy()


def _compute_effective_rank(matrix: np.ndarray) -> float:
    """Compute entropy-based effective rank using singular values."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = singular_values[singular_values > 0]
    if singular_values.size == 0:
        return 0.0
    prob = singular_values / np.sum(singular_values)
    entropy = -float(np.sum(prob * np.log(prob)))
    return float(np.exp(entropy))


def _compute_stable_rank(matrix: np.ndarray) -> float:
    """Compute stable rank: ||D||_F^2 / ||D||_2^2."""
    fro_sq = float(np.linalg.norm(matrix, ord="fro") ** 2)
    spectral = float(np.linalg.norm(matrix, ord=2))
    spectral_sq = spectral * spectral
    if spectral_sq == 0.0:
        return 0.0
    return fro_sq / spectral_sq


def _compute_mean_row_hamming_distance(matrix: np.ndarray) -> float:
    """Compute mean normalized pairwise Hamming distance across rows."""
    num_rows, num_cols = matrix.shape
    if num_rows < 2 or num_cols == 0:
        return 0.0

    pair_distances: list[float] = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            dist = float(np.mean(matrix[i] != matrix[j]))
            pair_distances.append(dist)

    if not pair_distances:
        return 0.0
    return float(np.mean(pair_distances))


def _compute_mean_row_l1_distance(matrix: np.ndarray) -> float:
    """Compute mean normalized pairwise L1 distance across rows."""
    num_rows, num_cols = matrix.shape
    if num_rows < 2 or num_cols == 0:
        return 0.0

    pair_distances: list[float] = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            dist = float(np.mean(np.abs(matrix[i] - matrix[j])))
            pair_distances.append(dist)

    if not pair_distances:
        return 0.0
    return float(np.mean(pair_distances))


def _compute_mean_row_l2_distance(matrix: np.ndarray) -> float:
    """Compute mean normalized pairwise L2 distance across rows."""
    num_rows, num_cols = matrix.shape
    if num_rows < 2 or num_cols == 0:
        return 0.0

    pair_distances: list[float] = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            dist = float(np.linalg.norm(matrix[i] - matrix[j]) / np.sqrt(num_cols))
            pair_distances.append(dist)

    if not pair_distances:
        return 0.0
    return float(np.mean(pair_distances))


def _compute_column_entropies(col_means: np.ndarray) -> np.ndarray:
    """Compute Bernoulli entropy per column from column means."""
    eps = 1e-12
    probs = np.clip(col_means, eps, 1.0 - eps)
    entropies = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))
    return entropies


def _compute_avg_abs_column_correlation(matrix: np.ndarray) -> float:
    """Compute mean absolute off-diagonal correlation across columns."""
    _, num_cols = matrix.shape
    if num_cols < 2:
        return 0.0

    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    upper = np.triu_indices(num_cols, k=1)
    if upper[0].size == 0:
        return 0.0
    return float(np.mean(np.abs(corr[upper])))


def _compute_matrix_summary(
    level: str,
    num_obstructions: int,
    num_train_seeds: int,
    training_label_mode: str,
    matrix: np.ndarray,
) -> MatrixSummary:
    """Compute diagnostics for one BOX score matrix."""
    rank = int(np.linalg.matrix_rank(matrix))
    effective_rank = _compute_effective_rank(matrix)
    stable_rank = _compute_stable_rank(matrix)

    num_rows = matrix.shape[0]
    unique_rows = np.unique(matrix, axis=0).shape[0]
    duplicate_row_ratio = 0.0
    if num_rows > 0:
        duplicate_row_ratio = 1.0 - (unique_rows / num_rows)

    diagnostics_mode = "binary" if training_label_mode == "binary" else "effort"

    mean_row_hamming_distance = 0.0
    if diagnostics_mode == "binary":
        mean_row_hamming_distance = _compute_mean_row_hamming_distance(matrix)

    mean_row_l1_distance = _compute_mean_row_l1_distance(matrix)
    mean_row_l2_distance = _compute_mean_row_l2_distance(matrix)

    if matrix.shape[1] == 0:
        col_means = np.zeros(0, dtype=float)
        col_entropies = np.zeros(0, dtype=float)
        col_variances = np.zeros(0, dtype=float)
    else:
        col_means = np.mean(matrix, axis=0)
        if diagnostics_mode == "binary":
            col_entropies = _compute_column_entropies(col_means)
        else:
            col_entropies = np.zeros_like(col_means)
        col_variances = np.var(matrix, axis=0)

    if matrix.shape[0] == 0:
        row_means = np.zeros(0, dtype=float)
    else:
        row_means = np.mean(matrix, axis=1)

    avg_abs_column_correlation = _compute_avg_abs_column_correlation(matrix)

    if col_means.size == 0:
        col_mean_mean = 0.0
        col_mean_std = 0.0
        col_mean_min = 0.0
        col_mean_max = 0.0
    else:
        col_mean_mean = float(np.mean(col_means))
        col_mean_std = float(np.std(col_means))
        col_mean_min = float(np.min(col_means))
        col_mean_max = float(np.max(col_means))

    if col_entropies.size == 0:
        col_entropy_mean = 0.0
        col_entropy_std = 0.0
        col_entropy_min = 0.0
        col_entropy_max = 0.0
    else:
        col_entropy_mean = float(np.mean(col_entropies))
        col_entropy_std = float(np.std(col_entropies))
        col_entropy_min = float(np.min(col_entropies))
        col_entropy_max = float(np.max(col_entropies))

    if col_variances.size == 0:
        col_variance_mean = 0.0
        col_variance_std = 0.0
        col_variance_min = 0.0
        col_variance_max = 0.0
    else:
        col_variance_mean = float(np.mean(col_variances))
        col_variance_std = float(np.std(col_variances))
        col_variance_min = float(np.min(col_variances))
        col_variance_max = float(np.max(col_variances))

    if row_means.size == 0:
        row_mean_mean = 0.0
        row_mean_std = 0.0
        row_mean_min = 0.0
        row_mean_max = 0.0
    else:
        row_mean_mean = float(np.mean(row_means))
        row_mean_std = float(np.std(row_means))
        row_mean_min = float(np.min(row_means))
        row_mean_max = float(np.max(row_means))

    return MatrixSummary(
        level=level,
        num_obstructions=num_obstructions,
        num_train_seeds=num_train_seeds,
        training_label_mode=training_label_mode,
        diagnostics_mode=diagnostics_mode,
        num_rows=int(matrix.shape[0]),
        vocab_size=int(matrix.shape[1]),
        rank=rank,
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        duplicate_row_ratio=duplicate_row_ratio,
        mean_row_hamming_distance=mean_row_hamming_distance,
        mean_row_l1_distance=mean_row_l1_distance,
        mean_row_l2_distance=mean_row_l2_distance,
        avg_abs_column_correlation=avg_abs_column_correlation,
        col_mean_mean=col_mean_mean,
        col_mean_std=col_mean_std,
        col_mean_min=col_mean_min,
        col_mean_max=col_mean_max,
        col_entropy_mean=col_entropy_mean,
        col_entropy_std=col_entropy_std,
        col_entropy_min=col_entropy_min,
        col_entropy_max=col_entropy_max,
        col_variance_mean=col_variance_mean,
        col_variance_std=col_variance_std,
        col_variance_min=col_variance_min,
        col_variance_max=col_variance_max,
        row_mean_mean=row_mean_mean,
        row_mean_std=row_mean_std,
        row_mean_min=row_mean_min,
        row_mean_max=row_mean_max,
    )


def _save_results(output_dir: str, summaries: list[MatrixSummary], raw: dict[str, Any]) -> None:
    """Write summary CSV and raw JSON outputs."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.csv")
    raw_path = os.path.join(output_dir, "raw_scores.json")

    fieldnames = list(asdict(summaries[0]).keys()) if summaries else []
    if fieldnames:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(asdict(summary))

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    print(f"[BOX-Experiment] Wrote summary CSV: {summary_path}")
    print(f"[BOX-Experiment] Wrote raw score JSON: {raw_path}")


@hydra.main(config_path="conf", config_name="box_matrix_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train BOX per complexity level and export score-matrix diagnostics."""
    prbench.register_all_environments()

    levels: list[str] = [str(x) for x in cfg.levels]
    train_seed_start = int(cfg.train_seed_start)
    num_train_seeds = int(cfg.num_train_seeds)

    summaries: list[MatrixSummary] = []
    raw_payload: dict[str, Any] = {
        "seed": int(cfg.seed),
        "levels": levels,
        "train_seed_start": train_seed_start,
        "num_train_seeds": num_train_seeds,
        "training_label_mode": str(cfg.approach.training_label_mode),
        "approach": {k: cfg.approach[k] for k in cfg.approach},
        "complexity_configs": (
            {k: {kk: vv for kk, vv in cfg.complexity_configs[k].items()} for k in cfg.complexity_configs}
            if "complexity_configs" in cfg
            else {}
        ),
        "results": {},
    }

    for level in levels:
        num_obstructions = int(level[1:])
        env_id = str(cfg.env.id_template).format(level=level)

        print(
            "[BOX-Experiment] "
            f"Level={level}, num_obstructions={num_obstructions}, env={env_id}"
        )

        level_timeout, level_max_abstract_plans = _get_complexity_config(cfg, level)
        print(
            "[BOX-Experiment] "
            f"Complexity settings: timeout={level_timeout}, "
            f"max_abstract_plans={level_max_abstract_plans}"
        )

        env = prbench.make(env_id)
        env_models = create_bilevel_planning_models(
            str(cfg.env.model_name),
            env.observation_space,
            env.action_space,
            num_obstructions=num_obstructions,
        )

        approach = _build_box_approach_for_level(
            cfg,
            env_models,
            level_timeout=level_timeout,
            level_max_abstract_plans=level_max_abstract_plans,
        )

        for seed in range(train_seed_start, train_seed_start + num_train_seeds):
            obs, _ = env.reset(seed=seed)
            approach.train(obs)

        approach._build_box_model()  # pylint: disable=protected-access

        score_matrix = _build_score_matrix(approach)

        summary = _compute_matrix_summary(
            level=level,
            num_obstructions=num_obstructions,
            num_train_seeds=num_train_seeds,
            training_label_mode=str(cfg.approach.training_label_mode),
            matrix=score_matrix,
        )
        summaries.append(summary)

        col_means = np.mean(score_matrix, axis=0) if score_matrix.shape[1] > 0 else np.zeros(0)
        if str(cfg.approach.training_label_mode) == "binary" and col_means.size > 0:
            col_entropies = _compute_column_entropies(col_means)
        else:
            col_entropies = np.zeros(0)
        col_variances = np.var(score_matrix, axis=0) if score_matrix.shape[1] > 0 else np.zeros(0)
        row_means = np.mean(score_matrix, axis=1) if score_matrix.shape[0] > 0 else np.zeros(0)

        raw_payload["results"][level] = {
            "num_obstructions": num_obstructions,
            "complexity_timeout": level_timeout,
            "complexity_max_abstract_plans": level_max_abstract_plans,
            "num_rows": int(score_matrix.shape[0]),
            "vocab_size": int(score_matrix.shape[1]),
            "score_matrix": score_matrix.tolist(),
            "column_means": col_means.tolist(),
            "column_entropies": col_entropies.tolist(),
            "column_variances": col_variances.tolist(),
            "row_means": row_means.tolist(),
            "diagnostics_mode": summary.diagnostics_mode,
            "summary": asdict(summary),
        }

        env.close()  # type: ignore[no-untyped-call]

    _save_results(str(cfg.output_dir), summaries, raw_payload)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
