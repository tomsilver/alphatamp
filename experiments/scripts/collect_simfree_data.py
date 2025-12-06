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
import pandas as pd
import prbench
from gymnasium import Env
from omegaconf import DictConfig
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from alphatamp.approaches.simulator_free_base_approach import (
    sesame_models_to_sim_free,
)
from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.filter_feasibility_classifier import (
    FilterFeasibilityClassifier,
)
from alphatamp.approaches.scorers.classifier_parameter_scorer import (
    ClassifierParameterScorer,
)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entrypoint called by Hydra."""

    # Get seed
    seed = int(cfg.seed)

    # Build env
    prbench.register_all_environments()
    env: Env = prbench.make(cfg.env.id)
    obs, _ = env.reset(seed=seed)

    # Build env models
    env_models = create_bilevel_planning_models(
        cfg.env.model_name,
        env.observation_space,
        env.action_space,
        **cfg.env.model_kwargs,  # unpacks dict into keyword args
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

     # Create the classifier.
    feasibility_classifier = hydra.utils.instantiate(cfg.feasibility_classifer)

    # Create the feasibility learner.
    feasibility_classifier_learner = hydra.utils.instantiate(cfg.feasibility_classifier_learner, feasibility_classifier)

    # Create the train explorer.
    train_explorer = hydra.utils.instantiate(cfg.train_explorer, sim_free_env_models, feasibility_classifier_learner, seed)

    # Create the classifier parameter scorer
    configs = {"hidden_layer_sizes": (10, 10)}

    # Build approach
    approach = hydra.utils.instantiate(cfg.approach, env_models=sim_free_env_models, 
                                       feasibility_classifier_learner=feasibility_classifier_learner, 
                                       train_explorer=train_explorer, 
                                       parameter_scorer_class=ClassifierParameterScorer,
                                       parameter_scorer_configs={"configs": configs},
                                       seed=seed)

    approach.train()

    metrics = _run_task_evaluation(
        env=env, approach=approach, obs=obs, timeout=float(cfg.timeout_sec)
    )

    # add metadata
    metrics.update(
        {
            "seed": seed,
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

    env.close()  # type: ignore[no-untyped-call]


def _run_task_evaluation(env, approach, obs, timeout: float) -> dict[str, object]:
    """Run planning once and compute metrics."""
    start_time = time.perf_counter()
    plan = approach.run_planning(obs, timeout=timeout)
    dur = time.perf_counter() - start_time

    success, num_actions = _check_success(env, plan)
    metrics: dict[str, object] = {}
    metrics["success"] = success
    metrics["cost"] = num_actions  # plan len was chosen arbitrarily
    metrics["duration"] = dur
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
