"""Tests for simfree_feasibility_approach.py."""

import time

import numpy as np
import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.abstract_explorers.random_explorer import RandomExplorer
from alphatamp.approaches.feasibility_classifiers.oracle_feasibility_classifier import (
    OracleAbstractPlanClassifier,
)
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.simfree_feasibility_approach import (
    SimFreeFeasiblityApproach,
)
from alphatamp.approaches.simulator_free_base_approach import (
    sesame_models_to_sim_free,
)


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

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the oracle classifier.
    oracle_classifier = OracleAbstractPlanClassifier(env_models)

    # Create the static feasibility learner.
    static_feasibility_classifier = StaticFeasibilityClassifierLearner(oracle_classifier)

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, static_feasibility_classifier, 123
    )

    # Create the approach.
    approach = SimFreeFeasiblityApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=static_feasibility_classifier,
        train_explorer=train_explorer,
        seed=123,
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    # Train.
    approach.train()
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

    # Eval.
    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    approach.eval()
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


def test_random_explorer_simfree_feasibility_approach():
    """Tests for SimFreeFeasiblityApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="random_explorer")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=10,
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the oracle classifier.
    oracle_classifier = OracleAbstractPlanClassifier(env_models)

    # Create the static feasibility learner.
    static_feasibility_classifier = StaticFeasibilityClassifierLearner(oracle_classifier)

    # Create the train explorer.
    train_explorer = RandomExplorer(
        sim_free_env_models, seed=123, max_plan_length=6
    )

    # Create the approach.
    approach = SimFreeFeasiblityApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=static_feasibility_classifier,
        train_explorer=train_explorer,
        seed=123,
    )

    # Train on two problems.
    training_data = []
    for _ in range(2):
        seed = np.random.randint(1000)
        obs, _ = env.reset(seed=seed)

        # Reset the approach on the observation.
        # Train.
        approach.train()
        approach.reset(obs, {})

        # Get the current abstract plan that was generated
        abstract_plan = approach.get_abstract_plan()

        assert abstract_plan, "Empty abstract plan!"

        start_time = time.time()
        timeout = 3
        task_status = "Plan timed out"

        while time.time() - start_time < timeout:
            try:
                action = approach.step()
            except RuntimeError as e:
                task_status = e
                break

            obs, reward, done, _, _ = env.step(action)

            # Given new observation from the environment, update the approach
            approach.update(obs, float(reward), done, {})
            if done:
                task_status = "Successful plan"
                break

        skeleton_success_data = (abstract_plan, task_status)
        training_data.append(skeleton_success_data)

    assert len(training_data) == 2, "Incorrect training data length"

    env.close()
