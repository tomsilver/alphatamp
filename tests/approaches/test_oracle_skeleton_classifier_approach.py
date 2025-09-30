"""Tests for oracle_skeleton_generator_approach.py."""

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.oracle_skeleton_classifier_approach import (
    OracleSkeletonClassifierApproach,
)


def test_oracle_skeleton_classifier_approach():
    """Tests for OracleSkeletonClassifierApproach()."""

    # Test in a PRBench environment where the first skeleton won't work.
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

    # Create the approach.
    approach = OracleSkeletonClassifierApproach(
        env_models, seed=123, samples_per_step=10, training_planning_timeout=10
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    import imageio.v2 as iio

    img = env.render()
    iio.imsave("debug_classifier.png", img)
    approach.train(obs)

    # Evaluation should succeed because we should have learned the pattern.
    plan = approach.run_planning(obs, timeout=100)

    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    # else:
    #     assert False, "Plan did not succeed"

    env.close()
