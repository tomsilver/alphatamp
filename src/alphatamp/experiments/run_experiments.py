"""Run a single experiment or Hydra multirun sweep.
This module instantiates a benchmark and apporach, then executes
planning and gets metrics."""

import os
import time

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_class
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entrypoint called by Hydra."""

    # Instantiate benchmark
    bench = hydra.utils.instantiate(cfg.benchmark)

    seed = int(cfg.seed)
    # Instead of generating tasks like in python_research_starter,
    # approach should operate on env/models/obs
    env, env_models, obs = bench.make_env_and_models(seed)

    # Build approach
    approach = hydra.utils.instantiate(cfg.approach, env_models, seed)
    approach.train(obs)  # essentially a noop, but just to keep the template

    metrics = _run_task_evaluation(
        env=env, bench=bench, approach=approach, obs=obs, timeout=float(cfg.timeout_sec)
    )

    # add metadata
    metrics.update(
        {
            "seed": seed,
            "n_blocks": int(cfg.benchmark.num_blocks),
            "approach_name": str(cfg.approach),
        }
    )

    df = pd.DataFrame([metrics])
    print(df)
    results_path = "results.csv"
    if os.path.exists(results_path):
        df.to_csv(results_path, mode="a", header=False)
    else:
        df.to_csv(results_path)

    env.close()


def _run_task_evaluation(env, bench, approach, obs, timeout: float) -> dict[str, object]:
    """Run planning once and compute metrics"""
    start_time = time.perf_counter()
    plan = approach.run_planning(obs, timeout=timeout)
    dur = time.perf_counter() - start_time

    success, num_actions = bench.check_success(env, plan)
    metrics: dict[str, object] = {}
    metrics["success"] = success
    metrics["cost"] = num_actions  # plan len was chosen arbitrarily
    metrics["duration"] = dur
    return metrics


if __name__ == "__main__":
    main()
