"""Tests for oracle_skeleton_generator_approach.py."""

import kinder
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.cluttered_storage.generalized_oracle_approach import (
    GeneralizedOracleApproach,
)


def test_generalized_oracle_approach():
    """Tests for OracleSkeletonGeneratorApproach()."""

    # Test in a kinder environment
    kinder.register_all_environments()
    env = kinder.make("kinder/ClutteredStorage2D-b7-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    # 2) Create bilevel models for this domain
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d",
        env.observation_space,
        env.action_space,
        num_blocks=7,
    )

    # Create the approach.
    approach = GeneralizedOracleApproach(
        env_models, seed=123, samples_per_step=10, training_planning_timeout=10
    )

    # Train the approach
    obs, _ = env.reset(seed=123)

    approach.train(obs)  # no-op, but keeps the pattern consistent

    # Create a plan
    plan = approach.run_planning(obs, timeout=500)

    # Execute the plan
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    env.close()
