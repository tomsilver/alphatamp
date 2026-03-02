"""Tests for the Dynamic 2D obstruction environment."""

import kinder
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.filter_feasibility_classifier import (
    FilterFeasibilityClassifier,
)
from alphatamp.approaches.practice_makes_perfect.base_approach import (
    PracticeMakesPerfectApproach,
)
from alphatamp.approaches.scorers.parameter_scorers.naive_parameter_scorer import (
    NaiveParameterScorer,
)
from alphatamp.approaches.simulator_free_base_approach import (
    sesame_models_to_sim_free,
)
from alphatamp.approaches.utils.approach_step_error import ApproachStepError


def test_practice_makes_perfect_approach():
    """Test PracticeMakesPerfectApproach() on Dynamic 2D environment."""

    # Test in a kinder environment.
    kinder.register_all_environments()
    env = kinder.make("kinder/DynObstruction2D-o0-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="dyn2d-pmp")

    env_models = create_bilevel_planning_models(
        "dynobstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )

    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Create the naive classifier.
    filter_classifier = FilterFeasibilityClassifier()

    # Create the static feasibility learner.
    static_feasibility_classifier = StaticFeasibilityClassifierLearner(
        filter_classifier
    )

    # Create the train explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, static_feasibility_classifier, 123
    )

    # Create the classifier parameter scorer configs
    parameter_configs = {"hidden_layer_sizes": (10, 10)}

    # Create the approach.
    approach = PracticeMakesPerfectApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=static_feasibility_classifier,
        train_explorer=train_explorer,
        parameter_scorer_class=NaiveParameterScorer,  # Use Naive Scorer
        parameter_scorer_configs={"configs": parameter_configs},
        seed=123,
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    # Reset the approach on the observation.
    # Train.
    approach.train()
    approach.reset(obs, {})

    num_steps = 200
    steps_taken = 0

    for _ in range(num_steps):
        try:
            action = approach.step()
        except ApproachStepError:
            break

        # Verify action is valid for the environment
        assert env.action_space.contains(action), "Action not in action space"

        obs, reward, done, _, _ = env.step(action)
        steps_taken += 1

        # Given new observation from the environment, update the approach
        approach.update(obs, float(reward), done, {})

    # Verify agent executed all steps
    assert steps_taken == 200, "Approach failed to execute all steps"

    # Verify agent is storing data during exploration
    parameter_dataset = approach.get_parameter_dataset()
    assert parameter_dataset, "Approach failed to store data"
    env.close()
