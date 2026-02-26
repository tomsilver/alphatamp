"""Tests for ngram_approach.py."""

import prbench
import pytest
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.ngram_approach import NGramApproach


def test_ngram_approach() -> None:
    """Test NGramApproach on Obstruction2D environment."""

    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    # Init approach
    approach: NGramApproach = NGramApproach(
        env_models,
        seed=123,
        samples_per_step=2,
        max_ngram_size=3,  # Track up to trigrams
        training_planning_timeout=10,
    )

    # Train on the problem
    obs, _ = env.reset(seed=101)
    approach.train(obs)

    # Print learned n-gram statistics for debugging
    print("\n" + "=" * 70)
    print("Learned N-gram Statistics:")
    print("=" * 70)
    ngram_summary = approach.get_ngram_summary()

    if ngram_summary:
        # Sort by success rate for easier interpretation
        sorted_ngrams = sorted(
            ngram_summary.items(), key=lambda x: x[1]["success_rate"], reverse=True
        )

        for ngram, stats in sorted_ngrams[:15]:  # Show top 15
            ngram_str = " -> ".join(ngram)
            success_rate = stats["success_rate"]
            total = stats["total_count"]
            print(
                f"  {ngram_str:60s} | "
                f"Rate: {success_rate:.2f} ({stats['success_count']}/{total})"
            )
    else:
        print("No n-grams learned.")

    print("\n" + "=" * 70)
    print("Running Planning with Learned N-grams:")
    print("=" * 70)

    plan = approach.run_planning(obs, timeout=100)

    # Execute the plan to verify it works
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    print("Test passed: Plan successfully executed!")

    env.close()  # type: ignore[no-untyped-call]


@pytest.mark.slow
def test_ngram_approach_no_training() -> None:
    """Test that the approach works even without training data."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o0-v0")  # No obstruction (easier)
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=0
    )

    approach: NGramApproach = NGramApproach(
        env_models,
        seed=123,
        samples_per_step=10,
        training_planning_timeout=5,
    )

    # Skip training
    obs, _ = env.reset(seed=123)

    # Should still work (no learned guidance, but planner still tries skeletons)
    plan = approach.run_planning(obs, timeout=100)

    # Execute to verify
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed"

    print("Test passed: Approach works without training data!")

    env.close()  # type: ignore[no-untyped-call]


def test_ngram_generalization() -> None:
    """Test that n-grams learned on one problem generalize to another.

    Train on seed=123, test on seed=456.
    """
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach: NGramApproach = NGramApproach(
        env_models,
        seed=123,
        samples_per_step=2,
        training_planning_timeout=10,
    )

    # Train on seed=123
    print("Training on seed=123...")
    obs_train, _ = env.reset(seed=123)
    approach.train(obs_train)

    # Test on different seed
    print("\nTesting on seed=456...")
    obs_test, _ = env.reset(seed=456)
    plan = approach.run_planning(obs_test, timeout=100)

    # Execute to verify
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed on test problem"

    print("Test passed: N-grams generalized to new problem!")

    env.close()  # type: ignore[no-untyped-call]
