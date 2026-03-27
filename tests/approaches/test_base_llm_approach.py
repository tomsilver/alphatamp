"""Tests for oracle_skeleton_generator_approach.py."""

import imageio.v2 as iio
import kinder
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.cluttered_storage.llm_approach import (
    BaseLLMApproach,
)


#@pytest.mark.skip(reason="Requires LLM calls - run manually when needed")
def test_base_llm_approach():
    """Tests for OracleSkeletonGeneratorApproach()."""

    # Test in a PRBench environment
    kinder.register_all_environments()
    env = kinder.make("kinder/ClutteredStorage2D-b3-v0", render_mode="rgb_array")

    if False:
        env = RecordVideo(env, "unit_test_videos")

    # 2) Create bilevel models for this domain
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d",
        env.observation_space,
        env.action_space,
        num_blocks=3,
    )

    # Create the approach.
    approach = BaseLLMApproach(
        env_models, seed=120, samples_per_step=10, training_planning_timeout=240
    )

    # Train the approach
    obs, _ = env.reset(seed=120)

    img = env.render()
    iio.imsave("debug.png", img)

    approach.train(obs)  # no-op, but keeps the pattern consistent

    try:
        # Create a plan
        plan = approach.run_planning(obs, timeout=240)

        # Execute the plan
        for action in plan.actions:
            _, _, done, _, _ = env.step(action)
            if done:
                break
        else:
            assert False, "Plan did not succeed"
    finally:
        m = approach.last_metrics
        if m is not None:
            print("\n=== Refinement Metrics ===")
            print(f"  Plan length (abstract steps): {m.num_steps}")
            print(f"  Attempts per step:            {m.attempts_per_step}")
            print(f"  Avg attempts per step:        {m.avg_attempts_per_step:.2f}")
            print(f"  Total sampling attempts:      {m.total_attempts}")
            print(f"  Steps with >5 attempts:       {m.steps_above_threshold(5)}")
        env.close()  # type: ignore[no-untyped-call]
