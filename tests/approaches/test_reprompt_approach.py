"""Tests for oracle_skeleton_generator_approach.py."""

import imageio.v2 as iio
import kinder
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.cluttered_storage.reprompt_approach import (
    RepromptApproach,
)


#@pytest.mark.skip(reason="Requires LLM calls - run manually when needed")
def test_reprompt_approach():
    """Tests for RepromptApproach()."""

    # Test in a kinder environment
    kinder.register_all_environments()
    env = kinder.make("kinder/ClutteredStorage2D-b3-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    # 2) Create bilevel models for this domain
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d",
        env.observation_space,
        env.action_space,
        num_blocks = 3
    )

    # Create the approach.
    approach = RepromptApproach(
        env_models, seed=125, samples_per_step=10, training_planning_timeout=10
    )

    # Train the approach
    obs, _ = env.reset(seed=125)

    img = env.render()
    iio.imsave("debug.png", img)

    approach.train(obs)  # no-op, but keeps the pattern consistent

    # Create a plan
    plan = approach.run_planning(obs, timeout=200)

    # Execute the plan
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    env.close()  # type: ignore[no-untyped-call]
