"""Tests for collaborative_filtering_approach.py."""

import kinder
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.collaborative_filtering_approach import (
    CollaborativeFilteringApproach,
)


def test_collaborative_filtering_approach():
    """Tests for CollaborativeFilteringApproach()."""

    # Test in a kinder environment where the first skeleton won't work.
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    # Create the approach.
    approach = CollaborativeFilteringApproach(
        env_models, seed=123, samples_per_step=2, training_planning_timeout=10
    )

    # Train on just one problem.
    obs, _ = env.reset(seed=123)
    approach.train(obs)

    # Evaluation should succeed because we should have learned the pattern.
    plan = approach.run_planning(obs, timeout=100)

    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    env.close()
