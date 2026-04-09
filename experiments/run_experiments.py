"""Run a single experiment or Hydra multirun sweep. This module instantiates a benchmark
and apporach, then executes planning and gets metrics.

To run: Run "python experiments/run_experiments.py"
from the alphatamp root directory (alphatamp/)
Note: If you're failing, try running "uv pip install -e ."
before the line above.
"""

import os
import time

import hydra
import kinder
import pandas as pd
from gymnasium import Env
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entrypoint called by Hydra."""

    # Get seed
    seed = int(cfg.seed)

    # Build env
    kinder.register_all_environments()
    env: Env = kinder.make(cfg.env.id)
    obs, _ = env.reset(seed=seed)

    # Build env models
    env_models = create_bilevel_planning_models(
        cfg.env.model_name,
        env.observation_space,
        env.action_space,
        **cfg.env.model_kwargs,  # unpacks dict into keyword args
    )

    # Build approach
    approach = hydra.utils.instantiate(cfg.approach, env_models, seed)
    approach.train(obs)

    metrics = _run_task_evaluation(
        env=env, approach=approach, obs=obs, timeout=float(cfg.timeout_sec)
    )

    # add metadata
    metrics.update(
        {
            "seed": seed,
            "approach_name": str(cfg.approach),
            "env_name": cfg.env.id,
            "timeout_sec": cfg.timeout_sec,
        }
    )

    all_columns = _COLUMNS + ["seed", "approach_name", "env_name", "timeout_sec"]
    df = pd.DataFrame([metrics]).reindex(columns=all_columns)
    print(df)
    results_path = cfg.get("results_path", "results.csv")
    if os.path.exists(results_path):
        df.to_csv(results_path, mode="a", header=False)
    else:
        df.to_csv(results_path)

    env.close()  # type: ignore[no-untyped-call]


_COLUMNS = [
    "success",
    "failure_reason",
    "duration",
    "cost",
    "abstract_plan",
    "avg_attempts_per_step",
    "total_sampling_attempts",
    "steps_above_5_attempts",
    "attempts_per_step",
]


def _run_task_evaluation(env, approach, obs, timeout: float) -> dict[str, object]:
    """Run planning once and compute metrics."""
    metrics: dict[str, object] = {col: None for col in _COLUMNS}

    start_time = time.perf_counter()
    plan = None
    try:
        plan = approach.run_planning(obs, timeout=timeout)
        metrics["success"] = True
    except Exception as e:  # pylint: disable=broad-except
        metrics["success"] = False
        metrics["failure_reason"] = type(e).__name__ + ": " + str(e)

    metrics["duration"] = time.perf_counter() - start_time

    if plan is not None:
        success, num_actions = _check_success(env, plan)
        metrics["success"] = success
        metrics["cost"] = num_actions

    metrics["abstract_plan"] = getattr(approach, "last_abstract_plan", None)

    ref = getattr(approach, "last_metrics", None)
    if ref is not None:
        metrics["avg_attempts_per_step"] = ref.avg_attempts_per_step
        metrics["total_sampling_attempts"] = ref.total_attempts
        metrics["steps_above_5_attempts"] = ref.steps_above_threshold(5)
        if metrics["success"]:
            metrics["attempts_per_step"] = ref.attempts_per_step

    return metrics


def _check_success(env, plan) -> tuple[bool, int]:
    """Step the plan; return (success, num_actions)."""
    num = 0
    for act in plan.actions:
        _, _, done, _, _ = env.step(act)
        num += 1
        if done:
            return True, num
    return False, num


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
