"""Run BOX vs baselines across complexity levels with Hydra.

This is a Hydra-based experiment counterpart of the extensive BOX visualization
benchmark from tests. It compares planning-time performance across approaches,
and writes per-seed raw results, summary CSV, and a plot.

Example:
    python experiments/run_box_effort_visualization_experiment.py
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import prbench
from omegaconf import DictConfig
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.box_approach import BoxApproach
from alphatamp.approaches.pure_planning_approach import PurePlanningApproach


@dataclass(frozen=True)
class ComplexityConfig:
    """Configuration for a specific complexity level."""

    timeout: float
    max_abstract_plans: int


def _get_complexity_config(cfg: DictConfig, level: str) -> ComplexityConfig:
    """Get per-level timeout and max_abstract_plans (with approach defaults)."""
    timeout = float(cfg.approach.training_planning_timeout)
    max_abstract_plans = int(cfg.approach.max_abstract_plans)

    if "complexity_configs" in cfg and level in cfg.complexity_configs:
        level_cfg = cfg.complexity_configs[level]
        if "timeout" in level_cfg:
            timeout = float(level_cfg.timeout)
        if "max_abstract_plans" in level_cfg:
            max_abstract_plans = int(level_cfg.max_abstract_plans)

    return ComplexityConfig(timeout=timeout, max_abstract_plans=max_abstract_plans)


def _build_box_approach_for_level(
    cfg: DictConfig,
    env_models: Any,
    level_cfg: ComplexityConfig,
) -> BoxApproach:
    """Create BoxApproach with per-level complexity settings."""
    label_mode = str(cfg.approach.training_label_mode)
    failure_penalty_multiplier = float(cfg.approach.failure_penalty_multiplier)

    return BoxApproach(
        env_models,
        seed=int(cfg.seed),
        max_abstract_plans=level_cfg.max_abstract_plans,
        samples_per_step=int(cfg.approach.samples_per_step),
        max_skill_horizon=int(cfg.approach.max_skill_horizon),
        heuristic_name=str(cfg.approach.heuristic_name),
        skeleton_batch_size=int(cfg.approach.skeleton_batch_size),
        num_training_skeletons_per_problem=int(
            cfg.approach.num_training_skeletons_per_problem
        ),
        training_planning_timeout=level_cfg.timeout,
        exploration_constant=float(cfg.approach.exploration_constant),
        training_label_mode=label_mode,
        failure_penalty_multiplier=failure_penalty_multiplier,
    )


def _run_single_test(approach: Any, seed: int, env_name: str, timeout: float) -> float:
    """Run one seed and return planning duration with timeout penalty on failure."""
    env = prbench.make(env_name)
    obs, _ = env.reset(seed=seed)

    start = time.perf_counter()
    duration = timeout

    try:
        plan = approach.run_planning(obs, timeout=timeout)
        duration = time.perf_counter() - start
        if plan is None:
            duration = max(duration, timeout)
        else:
            for action in plan.actions:
                _, _, done, _, _ = env.step(action)
                if done:
                    break
    except Exception as err:  # pylint: disable=broad-exception-caught
        print("-" * 40)
        print(
            "[BOX-Visualization] "
            f"Planning failed for seed={seed}, env={env_name}, err={err}"
        )
        print("-" * 40)
        duration = max(time.perf_counter() - start, timeout)
    finally:
        env.close()  # type: ignore[no-untyped-call]

    return duration


def _copy_training_data(source: BoxApproach, target: BoxApproach) -> None:
    """Copy training corpus so target can build priors from source data."""
    target._data = copy.deepcopy(source._data)  # pylint: disable=protected-access
    target._training_initial_states = copy.deepcopy(  # pylint: disable=protected-access
        source._training_initial_states  # pylint: disable=protected-access
    )

    # If source has already built its BOX model, copy it directly to avoid
    # re-running backfill/model-building during timed evaluation.
    if source._model_built:  # pylint: disable=protected-access
        target._skeletons_vocab = copy.deepcopy(  # pylint: disable=protected-access
            source._skeletons_vocab  # pylint: disable=protected-access
        )
        target._skeleton_to_idx = copy.deepcopy(  # pylint: disable=protected-access
            source._skeleton_to_idx  # pylint: disable=protected-access
        )
        target._prior_mu = np.array(  # pylint: disable=protected-access
            source._prior_mu, copy=True  # pylint: disable=protected-access
        )
        target._prior_sigma = np.array(  # pylint: disable=protected-access
            source._prior_sigma, copy=True  # pylint: disable=protected-access
        )
        target._score_matrix = np.array(  # pylint: disable=protected-access
            source._score_matrix, copy=True  # pylint: disable=protected-access
        )
        target._model_built = True  # pylint: disable=protected-access
    else:
        target._model_built = False  # pylint: disable=protected-access


def _build_diagonal_baseline(source: BoxApproach, baseline: BoxApproach) -> None:
    """Inject diagonal-covariance behavior into baseline BOX variant."""
    _copy_training_data(source, baseline)

    if baseline._model_built and baseline._prior_sigma is not None:  # pylint: disable=protected-access
        baseline._prior_sigma = np.diag(  # pylint: disable=protected-access
            np.diag(baseline._prior_sigma)  # pylint: disable=protected-access
        )
        return

    original_build = baseline._build_box_model  # pylint: disable=protected-access

    def _forced_diagonal_build() -> None:
        original_build()
        if baseline._prior_sigma is not None:  # pylint: disable=protected-access
            baseline._prior_sigma = np.diag(  # pylint: disable=protected-access
                np.diag(baseline._prior_sigma)  # pylint: disable=protected-access
            )

    baseline._build_box_model = _forced_diagonal_build  # type: ignore[method-assign]


def _save_results(
    output_dir: str,
    summary_rows: list[dict[str, Any]],
    raw_payload: dict[str, Any],
) -> None:
    """Write summary CSV and raw JSON outputs."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.csv")
    raw_path = os.path.join(output_dir, "raw_scores.json")

    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    with open(raw_path, "w", encoding="utf-8") as file:
        json.dump(raw_payload, file, indent=2)

    print(f"[BOX-Visualization] Wrote summary CSV: {summary_path}")
    print(f"[BOX-Visualization] Wrote raw score JSON: {raw_path}")


def _make_plot(
    output_dir: str,
    plot_filename: str,
    levels: list[str],
    all_results: dict[str, dict[str, list[float]]],
) -> None:
    """Generate level-wise bar chart with mean/std execution time."""
    if not levels:
        return

    approaches = list(all_results[levels[0]].keys())
    x = np.arange(len(approaches))

    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 6), sharey=False)
    if len(levels) == 1:
        axes = [axes]

    fig.suptitle("Planning Approach Performance by Obstruction Complexity", fontsize=16)

    for axis, level in zip(axes, levels):
        data = all_results[level]
        means = [float(np.mean(data[name])) for name in approaches]
        stds = [float(np.std(data[name])) for name in approaches]
        axis.bar(x, means, yerr=stds, capsize=5, alpha=0.85)
        axis.set_title(f"Complexity: {level}")
        axis.set_ylabel("Time (s)")
        axis.set_xticks(x)
        axis.set_xticklabels(approaches, rotation=45, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"[BOX-Visualization] Wrote plot: {plot_path}")


@hydra.main(config_path="conf", config_name="box_matrix_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run BOX benchmark comparisons and export timing artifacts."""
    prbench.register_all_environments()

    levels: list[str] = [str(x) for x in cfg.levels]
    train_seed_start = int(cfg.train_seed_start)
    num_train_seeds = int(cfg.num_train_seeds)
    test_seed_start = int(cfg.test_seed_start)
    num_test_seeds = int(cfg.num_test_seeds)
    output_dir = str(cfg.output_dir)
    plot_enabled = bool(cfg.make_plot)
    plot_filename = str(cfg.plot_filename)

    all_results: dict[str, dict[str, list[float]]] = {}
    raw_payload: dict[str, Any] = {
        "seed": int(cfg.seed),
        "levels": levels,
        "train_seed_start": train_seed_start,
        "num_train_seeds": num_train_seeds,
        "test_seed_start": test_seed_start,
        "num_test_seeds": num_test_seeds,
        "approach": {k: cfg.approach[k] for k in cfg.approach},
        "complexity_configs": (
            {
                level: {key: value for key, value in cfg.complexity_configs[level].items()}
                for level in cfg.complexity_configs
            }
            if "complexity_configs" in cfg
            else {}
        ),
        "results": {},
    }

    summary_rows: list[dict[str, Any]] = []

    for level in levels:
        num_obstructions = int(level[1:])
        env_name = str(cfg.env.id_template).format(level=level)
        level_cfg = _get_complexity_config(cfg, level)

        print(
            "[BOX-Visualization] "
            f"Level={level}, env={env_name}, timeout={level_cfg.timeout}, "
            f"max_abstract_plans={level_cfg.max_abstract_plans}"
        )

        train_env = prbench.make(env_name)
        env_models = create_bilevel_planning_models(
            str(cfg.env.model_name),
            train_env.observation_space,
            train_env.action_space,
            num_obstructions=num_obstructions,
        )

        box_approach = _build_box_approach_for_level(cfg, env_models, level_cfg)
        for seed in range(train_seed_start, train_seed_start + num_train_seeds):
            obs, _ = train_env.reset(seed=seed)
            box_approach.train(obs)

        # Build once before timed evaluation so expensive backfill/model build
        # does not occur inside per-seed timing.
        box_approach._build_box_model()  # pylint: disable=protected-access

        baseline_approach = _build_box_approach_for_level(cfg, env_models, level_cfg)
        _build_diagonal_baseline(box_approach, baseline_approach)

        pure_approach = PurePlanningApproach(
            env_models,
            seed=int(cfg.seed),
            samples_per_step=int(cfg.approach.samples_per_step),
            max_abstract_plans=level_cfg.max_abstract_plans,
        )

        class FilteredWrapper:
            def __init__(self, approach: BoxApproach) -> None:
                self._approach = approach

            def run_planning(self, obs: Any, timeout: float) -> Any:
                return self._approach.run_planning_filtered(obs, timeout)

        class SuccessfulFirstWrapper:
            def __init__(self, approach: BoxApproach) -> None:
                self._approach = approach

            def run_planning(self, obs: Any, timeout: float) -> Any:
                return self._approach.run_planning_successful_first(obs, timeout)

        approaches: dict[str, Any] = {
            "BOX": box_approach,
            "Baseline": baseline_approach,
            "Pure": pure_approach,
            "Filtered": FilteredWrapper(box_approach),
            "SuccessFirst": SuccessfulFirstWrapper(box_approach),
        }

        test_seeds = list(range(test_seed_start, test_seed_start + num_test_seeds))
        level_results: dict[str, list[float]] = {name: [] for name in approaches}

        for seed in test_seeds:
            for name, approach in approaches.items():
                dur = _run_single_test(
                    approach=approach,
                    seed=seed,
                    env_name=env_name,
                    timeout=level_cfg.timeout,
                )
                level_results[name].append(dur)
                print(f"[BOX-Visualization] level={level} seed={seed} {name}={dur:.4f}s")

        all_results[level] = level_results
        raw_payload["results"][level] = {
            "num_obstructions": num_obstructions,
            "timeout": level_cfg.timeout,
            "max_abstract_plans": level_cfg.max_abstract_plans,
            "test_seeds": test_seeds,
            "durations": level_results,
        }

        for approach_name, durations in level_results.items():
            summary_rows.append(
                {
                    "level": level,
                    "num_obstructions": num_obstructions,
                    "approach": approach_name,
                    "num_examples": len(durations),
                    "timeout": level_cfg.timeout,
                    "max_abstract_plans": level_cfg.max_abstract_plans,
                    "mean_time": float(np.mean(durations)) if durations else 0.0,
                    "std_time": float(np.std(durations)) if durations else 0.0,
                    "min_time": float(np.min(durations)) if durations else 0.0,
                    "max_time": float(np.max(durations)) if durations else 0.0,
                }
            )

        train_env.close()  # type: ignore[no-untyped-call]

    _save_results(output_dir=output_dir, summary_rows=summary_rows, raw_payload=raw_payload)
    if plot_enabled:
        _make_plot(
            output_dir=output_dir,
            plot_filename=plot_filename,
            levels=levels,
            all_results=all_results,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
