"""Tests for oracle_skeleton_generator_approach.py."""

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.oracle_skeleton_generator_approach import (
    OracleSkeletonGeneratorApproach,
)


def test_oracle_skeleton_generator_approach():
    """Tests for OracleSkeletonGeneratorApproach()."""

    # Test in a PRBench environment
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredStorage2D-b3-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    # 2) Create bilevel models for this domain
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d",
        env.observation_space,
        env.action_space,
        num_blocks=3,
    )

    # Create the approach.
    approach = OracleSkeletonGeneratorApproach(
        env_models, seed=123, samples_per_step=10, training_planning_timeout=10
    )

    # Train the approach
    obs, _ = env.reset(seed=123)

    import imageio.v2 as iio

    img = env.render()
    iio.imsave("debug.png", img)

    approach.train(obs)  # no-op, but keeps the pattern consistent

    # Create a plan
    plan = approach.run_planning(obs, timeout=100)

    # Execute the plan
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    env.close()
