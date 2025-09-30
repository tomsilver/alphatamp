"""Smoke test for the BookshelfPolicy on PRBench ClutteredStorage2D-b3-v0."""

import time
import prbench
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.oracle_approach import BookshelfPolicy


def test_bookshelf_policy_on_cluttered_storage():
    # 1) Bring up the env
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredStorage2D-b3-v0")

    # 2) Create bilevel models for this domain
    #    Domain key is "clutteredstorage2d" 
    env_models = create_bilevel_planning_models(
        "clutteredstorage2d",
        env.observation_space,
        env.action_space,
        num_blocks = 3,
    )

    # 3) Make the approach
    approach = BookshelfPolicy(env_models, seed=123)

    # 4) Reset -> problem
    obs, _ = env.reset(seed=123)
    approach.train(obs)  # no-op, but keeps the pattern consistent

    # 5) Plan
    start = time.time()
    plan = approach.run_planning(obs, timeout=120.0)  # keep timeout modest for tests
    plan_time = time.time() - start
    print(f"[BookshelfPolicy] planning time: {plan_time:.2f}s, actions: {len(plan.actions)}")

    # 6) Execute
    done = False
    for a in plan.actions:
        _, _, done, _, _ = env.step(a)
        if done:
            break

    assert done, "Plan did not achieve goal (placing 3 books in the shelf)."
    env.close()
