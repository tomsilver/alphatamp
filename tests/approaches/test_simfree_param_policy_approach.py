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
from alphatamp.approaches.scorers.classifier_parameter_scorer import (
    ClassifierParameterScorer,
)
from alphatamp.approaches.scorers.naive_parameter_scorer import NaiveParameterScorer
from alphatamp.approaches.scorers.regressor_abstract_action_scorer import (
    AbstractActionScorer,
)
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

    # Create the classifier parameter scorer configs
    parameter_configs = {"hidden_layer_sizes": (10, 10)}

    # Create the abstract action scorer configs
    abstract_action_configs = {
        "input_dim": 2,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 10,
    }

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=static_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=NaiveParameterScorer,  # Use Naive Scorer
        parameter_scorer_configs={"configs": parameter_configs},
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
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


def test_dataset_collection_simfree_feasibility_approach():
    """Tests for collecting datasets for the SimFreeParamPolicyApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="param-policy-datasets")

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

    # Create the classifier parameter scorer configs
    parameter_configs = {"hidden_layer_sizes": (10, 10)}

    # Create the abstract action scorer configs
    abstract_action_configs = {
        "input_dim": 2,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 10,
    }

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs={"configs": parameter_configs},
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
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

    abstract_plan_dataset = approach.get_abstract_plan_dataset()
    abstract_action_dataset = approach.get_abstract_action_dataset()

    assert abstract_plan_dataset, "Did not find any abstract plans"
    assert abstract_action_dataset, "Did not store any abstract actions data"
    env.close()


def test_save_datasets_simfree_feasibility_approach():
    """Tests for saving datasets for the SimFreeParamPolicyApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="param-policy-parameter")

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

    # Create the classifier parameter scorer configs
    parameter_configs = {"hidden_layer_sizes": (10, 10)}

    # Create the abstract action scorer configs
    abstract_action_configs = {
        "input_dim": 2,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 10,
    }

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs={"configs": parameter_configs},
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
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

    path = Path("tests/datasets/")
    approach.save_datasets(path)

    # Eval.
    obs, _ = env.reset(seed=124)

    # Filter obstruction 6
    filtered_action_strs = [("obstruction6", 0)]
    filter_classifier.update_classifier(None, filtered_action_strs)

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
    """Tests for training the parameter scorer for the SimFreeParamPolicyApproach()."""

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

    # Create the classifier parameter scorer configs
    parameter_configs = {"hidden_layer_sizes": (10, 10)}

    # Create the abstract action scorer configs
    abstract_action_configs = {
        "input_dim": 2,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 10,
    }

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs={"configs": parameter_configs},
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
        seed=123,
    )

    # Eval.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    approach.eval()
    approach.reset(obs, {})

    # Load in successful training dataset from pickle.
    path = Path("tests/datasets") / "success_classifier_parameter_dataset.pkl"
    success_dataset = approach.load_abstract_action_level_dataset(path)

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


def test_train_abstract_action_scorer_simfree_feasbility_approach():
    """Tests for training the abstract action scorers for the
    SimFreeParamPolicyApproach()."""

    # Test in a PRBench environment.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="q-function")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=10,
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the naive classifier.
    filter_classifier = FilterFeasibilityClassifier()

    # Create the naive feasibility learner.
    filter_feasibility_classifier = StaticFeasibilityClassifierLearner(
        filter_classifier
    )

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, filter_feasibility_classifier, 123
    )

    # Create the classifier parameter scorer configs
    parameter_configs = {"hidden_layer_sizes": (10, 10)}

    # Create the abstract action scorer configs
    abstract_action_configs = {
        "input_dim": 2,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 10,
    }

    # Create the approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=filter_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs={"configs": parameter_configs},
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
        seed=123,
    )

    # Eval.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    approach.eval()
    approach.reset(obs, {})

    # Load in abstract action training dataset from pickle.
    path = Path("tests/datasets") / "abstract_action_dataset.pkl"
    dataset = approach.load_abstract_action_level_dataset(path)

    # Train the abstract action scorers on the datasets.
    approach.train_abstract_action_scorer(dataset)

    abstract_action_descriptor = "'PlaceTgt(robot, target_block, target_region)'"

    abstract_action_score = approach.get_abstract_action_score(
        abstract_action_descriptor
    )

    assert abstract_action_score < 1, "Should not need any resamples!"

    env.close()
