"""Tests for simfree_feasibility_approach.py."""

import time
from typing import cast

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.feasibility_classifiers.oracle_feasibility_classifier import (
    OracleAbstractPlanClassifier,
)
from alphatamp.approaches.feasibility_classifiers.static_feasibility_classifier import (
    StaticFeasibilityClassifier,
)
from alphatamp.approaches.simfree_feasibility_approach import (
    SimFreeFeasiblityApproach,
)
from alphatamp.approaches.simulator_free_base_approach import SimulatorFreeSesameModels


def test_static_classifier_simfree_feasibility_approach():
    """Tests for SimFreeFeasiblityApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o1-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )

    # Create the oracle classifier.
    oracle_classifier = OracleAbstractPlanClassifier(env_models)

    # Create the static feasibility learner.
    static_feasibility_classifier = StaticFeasibilityClassifier(oracle_classifier)

    # Create the approach.
    approach = SimFreeFeasiblityApproach(
        env_models=cast(SimulatorFreeSesameModels, env_models),
        feasibility_classifier_learner=static_feasibility_classifier,
        seed=123,
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    approach.reset(obs, {})

    start_time = time.time()
    timeout = 4
    task_completed = False

    while time.time() - start_time < timeout:
        action = approach.step()

        obs, reward, done, _, _ = env.step(action)

        # Given new observation from the environment, update the approach
        approach.update(obs, float(reward), done, {})
        if done:
            task_completed = True
            break

    assert task_completed, "Plan did not succeed"

    env.close()
