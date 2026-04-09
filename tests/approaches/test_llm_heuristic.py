"""Tests for heuristic_approach.py."""

import imageio.v2 as iio
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.cluttered_storage.oracle_heuristic_approach import (
    OracleHeuristicApproach,
)


#@pytest.mark.skip(reason="Requires LLM calls - run manually when needed")
def test_heuristic_approach():
    """Tests for HeuristicLLMApproach()."""

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
    approach = OracleHeuristicApproach(
        env_models, seed=123, samples_per_step=10, training_planning_timeout=10, use_stored_heuristic=False
    )

    # Train the approach
    obs, _ = env.reset(seed=123)

    img = env.render()
    iio.imsave("debug.png", img)

    approach.train(obs)  # no-op, but keeps the pattern consistent

    # Create a plan
    plan = approach.run_planning(obs, timeout=200)

    # Print refinement metrics
    m = approach.last_metrics
    if m is not None:
        print(f"\n=== Refinement Metrics ===")
        print(f"  Plan length (abstract steps): {m.num_steps}")
        print(f"  Attempts per step:            {m.attempts_per_step}")
        print(f"  Avg attempts per step:        {m.avg_attempts_per_step:.2f}")
        print(f"  Total sampling attempts:      {m.total_attempts}")
        print(f"  Steps with >5 attempts:       {m.steps_above_threshold(5)}")

    # Execute the plan
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    env.close()  # type: ignore[no-untyped-call]
