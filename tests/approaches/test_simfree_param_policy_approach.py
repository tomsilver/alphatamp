"""Tests for simfree_param_policy_approach.py."""

import time
from pathlib import Path

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.filter_feasibility_classifier import (
    FilterFeasibilityClassifier,
)
from alphatamp.approaches.feasibility_classifiers.oracle_feasibility_classifier import (
    OracleAbstractPlanClassifier,
)
from alphatamp.approaches.parameter_scorers.classifier_scorer import ClassifierScorer
from alphatamp.approaches.parameter_scorers.naive_scorer import NaiveScorer
from alphatamp.approaches.simfree_param_policy_approach import (
    SimFreeParamPolicyApproach,
)
from alphatamp.approaches.simulator_free_base_approach import (
    sesame_models_to_sim_free,
)
from alphatamp.approaches.utils.approach_step_error import ApproachStepError


def test_naive_scorer_simfree_feasibility_approach():
    """Tests for SimFreeParamPolicyApproach()."""

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
    static_feasibility_classifier = StaticFeasibilityClassifierLearner(
        oracle_classifier
    )

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, static_feasibility_classifier, 123
    )

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=static_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=NaiveScorer,  # Use Naive Scorer
        parameter_scorer_configs={"configs": {}},
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


def test_classifier_scorer_simfree_feasibility_approach():
    """Tests for SimFreeParamPolicyApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="param-policy")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=10,
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the naive classifier.
    filter_classifier = FilterFeasibilityClassifier()

    # Train the feasibility learner to classify plans
    # pick up the target block first as infeasible

    filtered_action_strs = [("PickTgt", 0), ("target_block", 0)]
    filter_classifier.update_classifier(None, filtered_action_strs)

    # Create the naive feasibility learner.
    filter_feasibility_classifier = StaticFeasibilityClassifierLearner(
        filter_classifier
    )

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, filter_feasibility_classifier, 123
    )

    # Create the classifier parameter scorer
    configs = {"hidden_layer_sizes": (10, 10)}

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierScorer,
        parameter_scorer_configs={"configs": configs},
        seed=123,
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    # Train.
    approach.train()
    approach.reset(obs, {})

    start_time = time.time()
    timeout = 10

    while time.time() - start_time < timeout:
        try:
            action = approach.step()
        except ApproachStepError:
            break

        obs, reward, done, _, _ = env.step(action)

        # Given new observation from the environment, update the approach
        approach.update(obs, float(reward), done, {})
        if done:
            break

    parameter_dataset = approach.get_parameter_dataset()

    assert parameter_dataset, "Did not find any parameters"

    path = Path("tests/datasets/success_classifier_parameter_dataset.pkl")
    approach.save_parameter_dataset(path)

    # Eval.
    obs, _ = env.reset(seed=124)

    # Filter obstruction 6
    filtered_action_strs = [("obstruction6", 0)]
    filter_classifier.update_classifier(None, filtered_action_strs)

    approach.eval()
    approach.reset(obs, {})

    start_time = time.time()
    timeout = 10

    while time.time() - start_time < timeout:
        try:
            action = approach.step()
        except ApproachStepError:
            break

        obs, reward, done, _, _ = env.step(action)

        # Given new observation from the environment, update the approach
        approach.update(obs, float(reward), done, {})
        if done:
            break

    parameter_dataset = approach.get_parameter_dataset()

    assert len(parameter_dataset) == 4, "Should not store additional parameters."
    env.close()


def test_train_scorer_simfree_feasbility_approach():
    """Tests for SimFreeParamPolicyApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="param-policy")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=10,
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the naive classifier.
    filter_classifier = FilterFeasibilityClassifier()

    # Filter bad abstract plans
    filtered_action_strs = [
        ("PickTgt", 0),
        ("target_block", 0),
        ("obstruction5", 0),
        ("obstruction6", 0),
    ]
    filter_classifier.update_classifier(None, filtered_action_strs)

    # Create the naive feasibility learner.
    filter_feasibility_classifier = StaticFeasibilityClassifierLearner(
        filter_classifier
    )

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, filter_feasibility_classifier, 123
    )

    # Create the classifier parameter scorer
    configs = {"hidden_layer_sizes": (10, 10)}

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierScorer,
        parameter_scorer_configs={"configs": configs},
        seed=123,
    )

    # Eval.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    approach.eval()
    approach.reset(obs, {})

    # Load in successful training dataset from pickle.
    path = Path("tests/datasets") / "success_classifier_parameter_dataset.pkl"
    success_dataset = approach.load_parameter_dataset(path)

    # Train the scorer on the datasets.
    approach.train_parameter_policy(success_dataset)

    # Evaluate the approach on environment.
    start_time = time.time()
    timeout = 10
    task_completed = False
    while time.time() - start_time < timeout:
        try:
            action = approach.step()
        except ApproachStepError:
            break

        obs, reward, done, _, _ = env.step(action)

        # Given new observation from the environment, update the approach
        approach.update(obs, float(reward), done, {})
        if done:
            task_completed = True
            break

    assert task_completed, "Plan did not succeed"
    env.close()


def test_train_scorer_simfree_feasbility_approach():
    
    training_data = {
        "test": [((0.6299402045896808, 0.927407258525167, 0.12710809229482242), 'success'), ((0.2652161321933526, 0.6520188384409447, 0.927480196623524), 'success'), ((0.10609773749426321, 0.15332923005332866, 0.16022568832130613), 'success')]

    }

    """Tests for SimFreeParamPolicyApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="param-policy")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=10,
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the naive classifier.
    filter_classifier = FilterFeasibilityClassifier()

    # Train the feasibility learner to classify plans
    # pick up the target block first as infeasible

    filtered_action_strs = [("PickTgt", 0), ("target_block", 0)]
    filter_classifier.update_classifier(None, filtered_action_strs)

    # Create the naive feasibility learner.
    filter_feasibility_classifier = StaticFeasibilityClassifierLearner(
        filter_classifier
    )

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, filter_feasibility_classifier, 123
    )

    # Create the classifier parameter scorer
    configs = {"hidden_layer_sizes": (10, 10)}
    classifier_scorer = ClassifierScorer(configs)

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer=classifier_scorer,
        seed=123,
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    # Train.
    approach.train()
    approach.reset(obs, {})

    start_time = time.time()
    timeout = 10

    while time.time() - start_time < timeout:
        try:
            action = approach.step()
        except ApproachStepError:
            break

        obs, reward, done, _, _ = env.step(action)

        # Given new observation from the environment, update the approach
        approach.update(obs, float(reward), done, {})
        if done:
            break

    parameter_dataset = approach.get_parameter_dataset()

    print(parameter_dataset)

    assert parameter_dataset, "Did not find any parameters"
