"""Tests for non_relational_approach.py."""

import time

import prbench
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.non_relational_approach import (
    NonRelationalApproach,
)


def test_non_relational_approach():
    """Tests for NonRelationalApproach()."""

    # Test in a PRBench environment where there are multiple objects to move around.
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0")
    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )

    # Create the approach.
    approach = NonRelationalApproach(env_models, seed=123)

    # Train on just one problem.
    obs, _ = env.reset(seed=123)

    # Training should do nothing
    approach.train(obs)

    # Evaluation should take a long time due to bad heuristic.
    start_planning_time = time.time()
    plan = approach.run_planning(obs, timeout=100)

    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    total_planning_time = time.time() - start_planning_time
    print(total_planning_time)
    env.close()
