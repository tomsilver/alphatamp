"""Tests for box_approach.py."""

import time
import numpy as np
import prbench
import pytest
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from bilevel_planning.structs import Plan, PlanningProblem
from dataclasses import dataclass

from alphatamp.approaches.box_approach import BoxApproach
from alphatamp.approaches.pure_planning_approach import PurePlanningApproach

from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import math
import copy


def test_box_approach_no_training() -> None:
    """Test that the approach works even without training data"""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o0-v0")  # No obstruction
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=0
    )

    approach = BoxApproach(
        env_models,
        seed=123,
        samples_per_step=10,
        training_planning_timeout=5,
    )

    # Skip training
    obs, _ = env.reset(seed=123)

    # Should still work
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


def test_box_approach_basic_flow() -> None:
    """Test basic training and execution flow"""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = BoxApproach(
        env_models,
        seed=123,
        samples_per_step=2,
        training_planning_timeout=10,
    )

    # Train on seed=123
    print("Training on seed=123...")
    obs_train, _ = env.reset(seed=123)
    approach.train(obs_train)

    # Test on same seed
    print("\nTesting on seed=123...")
    obs_test, _ = env.reset(seed=123)
    plan = approach.run_planning(obs_test, timeout=100)

    # Execute to verify
    for action in plan.actions:
        _, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not succeed on test problem"

    print("Test passed: Basic training and execution successful!")

    env.close()  # type: ignore[no-untyped-call]


def test_box_backfills(monkeypatch) -> None:
    """Ensure missing skeleton gaps are backfilled before D statistics are computed."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=0
    )

    approach = BoxApproach(
        env_models,
        seed=123,
        samples_per_step=1,
        num_training_skeletons_per_problem=1,
        training_planning_timeout=2,
    )

    obs0, _ = env.reset(seed=10)
    approach.train(obs0)
    obs1, _ = env.reset(seed=11)
    approach.train(obs1)

    if len(approach._data) < 2 or len(approach._data[0]) == 0:
        pytest.skip("Insufficient training skeletons for backfill test.")

    source_idx = 0
    target_idx = 1
    skeleton = next(iter(approach._data[source_idx].keys()))
    approach._data[target_idx].pop(skeleton, None)

    assert skeleton not in approach._data[target_idx]

    call_count = 0

    def _fake_refiner(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return object()

    monkeypatch.setattr(approach, "_refiner", _fake_refiner)

    approach._build_box_model()

    assert call_count >= 1
    assert skeleton in approach._data[target_idx]
    assert approach._data[target_idx][skeleton][0] <= 0.0
    assert approach._data[target_idx][skeleton][0] > -1.0
    assert all(
        len(problem_data) == len(approach._skeletons_vocab)
        for problem_data in approach._data
    )

    assert approach._prior_mu is not None
    skel_idx = approach._skeleton_to_idx[skeleton]
    assert approach._prior_mu[skel_idx] <= 0.0

    env.close()  # type: ignore[no-untyped-call]


def test_score_from_refinement_legacy_linear_binary_mode() -> None:
    """Legacy linear helper keeps binary behavior in binary mode."""
    approach = BoxApproach.__new__(BoxApproach)
    approach._training_label_mode = "binary"  # pylint: disable=protected-access

    assert approach._score_from_refinement_legacy_linear(True, 10, 3) == 1.0  # pylint: disable=protected-access
    assert approach._score_from_refinement_legacy_linear(False, 10, 3) == 0.0  # pylint: disable=protected-access


def test_score_from_refinement_legacy_linear_effort_mode() -> None:
    """Legacy linear helper keeps prior effort behavior."""
    approach = BoxApproach.__new__(BoxApproach)
    approach._training_label_mode = "effort"  # pylint: disable=protected-access
    approach._samples_per_step = 10  # pylint: disable=protected-access
    approach._failure_penalty_multiplier = 2.0  # pylint: disable=protected-access

    assert approach._score_from_refinement_legacy_linear(True, 0, 3) == 60.0  # pylint: disable=protected-access
    assert approach._score_from_refinement_legacy_linear(True, 15, 3) == 45.0  # pylint: disable=protected-access
    assert approach._score_from_refinement_legacy_linear(True, 60, 3) == 0.0  # pylint: disable=protected-access
    assert approach._score_from_refinement_legacy_linear(True, 80, 3) == 0.0  # pylint: disable=protected-access
    assert approach._score_from_refinement_legacy_linear(False, 1, 3) == 0.0  # pylint: disable=protected-access


def test_score_from_refinement_process_time_non_timeout() -> None:
    """Process-time scoring should be negative elapsed process time."""
    approach = BoxApproach.__new__(BoxApproach)
    approach._failure_penalty_multiplier = 2.0  # pylint: disable=protected-access

    score = approach._score_from_refinement(0.25, 1.0, False)  # pylint: disable=protected-access
    assert score == -0.25


def test_score_from_refinement_process_time_timeout_penalty() -> None:
    """Timeout should map to timeout * penalty, then negated."""
    approach = BoxApproach.__new__(BoxApproach)
    approach._failure_penalty_multiplier = 2.5  # pylint: disable=protected-access

    score = approach._score_from_refinement(0.10, 4.0, True)  # pylint: disable=protected-access
    assert score == -10.0


def test_get_score_matrix_copy_requires_built_model() -> None:
    """Accessor should error when score matrix is unavailable."""
    approach = BoxApproach.__new__(BoxApproach)
    approach._model_built = False  # pylint: disable=protected-access
    approach._score_matrix = None  # pylint: disable=protected-access

    with pytest.raises(RuntimeError):
        approach.get_score_matrix_copy()


def test_get_score_matrix_copy_is_defensive_copy() -> None:
    """Accessor should return a copy rather than mutating internal state."""
    approach = BoxApproach.__new__(BoxApproach)
    approach._model_built = True  # pylint: disable=protected-access
    approach._score_matrix = np.array([[1.0, 2.0]], dtype=float)  # pylint: disable=protected-access

    matrix = approach.get_score_matrix_copy()
    matrix[0, 0] = 999.0

    assert approach._score_matrix is not None  # pylint: disable=protected-access
    assert approach._score_matrix[0, 0] == 1.0  # pylint: disable=protected-access


def _run_single_test(
    approach,
    seed: int,
    env_name: str,
    video_folder: str,
    name_prefix: str,
    timeout: float,
) -> float:
    """Helper method to run a single test and return execution time."""
    env = prbench.make(env_name)
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, 
            video_folder, 
            name_prefix=f"{name_prefix}_seed_{seed}_",
            episode_trigger=lambda x: True # TODO: does force recording work?
        )

    obs, _ = env.reset(seed=seed)
    start = time.perf_counter()
    try:
        plan = approach.run_planning(obs, timeout=timeout)
        duration = time.perf_counter() - start

        if plan is None:
            duration = max(duration, timeout) # no plan found penalty
        else:
            # execute to generate video
            for action in plan.actions:
                _, _, done, _, _ = env.step(action)
                if done:
                    break
    except Exception as e:
        print("-" * 40)
        print(
            f"Planning timed out for approach {name_prefix}, seed {seed}, environment {env_name}"
        )
        print(f"Exception: {e}")
        print("-" * 40)
        duration = max(time.perf_counter() - start, timeout)  # Penalty for failure

    env.close()
    return duration


DEFAULT_TEST_TIMEOUT = 20.0
DEFAULT_MAX_ABSTRACT_PLANS = 20.0
SAMPLES_PER_STEP = 10


@dataclass(frozen=True)
class ComplexityConfig:
    """Configuration for a specific complexity level."""

    timeout: float = DEFAULT_TEST_TIMEOUT
    max_abstract_plans: int = DEFAULT_MAX_ABSTRACT_PLANS


# Environment-specific complexity configurations
COMPLEXITY_CONFIGS: dict[str, ComplexityConfig] = {
    "o0": ComplexityConfig(timeout=20, max_abstract_plans=10),
    "o1": ComplexityConfig(timeout=30, max_abstract_plans=15),
    "o2": ComplexityConfig(timeout=40, max_abstract_plans=20),
    "o3": ComplexityConfig(timeout=50, max_abstract_plans=25),
    "o4": ComplexityConfig(timeout=60, max_abstract_plans=30),
}

def _get_complexity_config(complexity: str) -> ComplexityConfig:
    """Get complexity configuration for a given complexity level."""
    return COMPLEXITY_CONFIGS.get(
        complexity,
        ComplexityConfig(),  # Default config
    )

@pytest.mark.slow
def _run_box_vs_baseline_performance(
    obstruction_level: str = "o2",
) -> dict[str, list[float]]:
    """
    Compare BOX against baselines with configurable complexity.

    Returns:
        Dictionary mapping approach name to list of execution times.
    """

    config = _get_complexity_config(obstruction_level)
    print(f"Running test with obstruction_level={obstruction_level}, config={config}")

    prbench.register_all_environments()

    env_name = f"prbench/Obstruction2D-{obstruction_level}-v0"
    num_examples = 10
    TRAIN_SEED_START = 100
    TEST_SEED_START = 1000


    env = prbench.make(env_name)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "box_approach_videos")

    # Extract num_obstructions from level string
    num_obstructions = int(obstruction_level[1:])

    env_models = create_bilevel_planning_models(
        "obstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=num_obstructions,
    )

    # Train BOX Approach
    box_approach = BoxApproach(
        env_models,
        seed=123,
        samples_per_step=SAMPLES_PER_STEP,
        max_abstract_plans=config.max_abstract_plans,
        training_planning_timeout=config.timeout,
        exploration_constant=math.sqrt(2),
    )

    # Train on a few problems
    print("Training BOX approach...")
    for seed in range(TRAIN_SEED_START, TRAIN_SEED_START + num_examples):
        obs, _ = env.reset(seed=seed)
        box_approach.train(obs)

    # Create Baseline Approach: BOX w diagonal covariance
    baseline_approach = BoxApproach(
        env_models,
        seed=123,
        samples_per_step=SAMPLES_PER_STEP,
        max_abstract_plans=config.max_abstract_plans,
        training_planning_timeout=config.timeout,
        exploration_constant=math.sqrt(2),
    )

    # Copy training data from BOX to Baseline 
    baseline_approach._data = copy.deepcopy(box_approach._data)

    # Hack: Override _build_box_model to force diagonal covariance
    original_build = baseline_approach._build_box_model

    def forced_diagonal_build() -> None:
        original_build()
        if baseline_approach._prior_sigma is not None:
            # Zero out off-diagonals
            M = baseline_approach._prior_sigma.shape[0]
            baseline_approach._prior_sigma = np.diag( # create diagonal matrix w off-diags 0
                np.diag(baseline_approach._prior_sigma) # extract diagonals
            )

    baseline_approach._build_box_model = forced_diagonal_build  # type: ignore

    # Create Pure Planning Approach
    pure_approach = PurePlanningApproach(
        env_models,
        seed=123,
        samples_per_step=SAMPLES_PER_STEP,
        max_abstract_plans=config.max_abstract_plans
    )

    # Create Filtered Approach as comparison
    class FilteredWrapper:
        def __init__(self, approach: BoxApproach):
            self.approach = approach

        def run_planning(self, obs, timeout: float) -> Plan:
            return self.approach.run_planning_filtered(obs, timeout)

    filtered_approach = FilteredWrapper(box_approach)

    # Create Successful-First Approach as comparison
    class SuccessfulFirstWrapper:
        def __init__(self, approach: BoxApproach):
            self.approach = approach

        def run_planning(self, obs, timeout: float) -> Plan:
            return self.approach.run_planning_successful_first(obs, timeout)

    successful_first_approach = SuccessfulFirstWrapper(box_approach)

    # 6. Compare Performance on Test Set
    print("\nComparing Performance...")
    test_seeds = range(TEST_SEED_START, TEST_SEED_START + num_examples)
    video_folder = "box_approach_videos"

    results = {
        "BOX": [],
        "Baseline": [],
        "Pure": [],
        "Filtered": [],
        "SuccessFirst": [],
    }

    for seed in test_seeds:
        # Run BOX
        t_box = _run_single_test(
            box_approach, seed, env_name, video_folder, "box", config.timeout
        )
        results["BOX"].append(t_box)
        print(f"Seed {seed} | BOX: {t_box:.4f}s")

        # Run Baseline
        t_base = _run_single_test(
            baseline_approach, seed, env_name, video_folder, "base", config.timeout
        )
        results["Baseline"].append(t_base)
        print(f"Seed {seed} | Baseline: {t_base:.4f}s")

        # Run Pure
        t_pure = _run_single_test(
            pure_approach, seed, env_name, video_folder, "pure", config.timeout
        )
        results["Pure"].append(t_pure)
        print(f"Seed {seed} | Pure: {t_pure:.4f}s")

        # Run Filtered
        t_filtered = _run_single_test(
            filtered_approach, seed, env_name, video_folder, "filtered", config.timeout
        )
        results["Filtered"].append(t_filtered)
        print(f"Seed {seed} | Filtered: {t_filtered:.4f}s")

        # Run Successful First
        t_success = _run_single_test(
            successful_first_approach,
            seed,
            env_name,
            video_folder,
            "success_first",
            config.timeout,
        )
        results["SuccessFirst"].append(t_success)
        print(f"Seed {seed} | SuccessFirst: {t_success:.4f}s")

    # Print summary for this level
    print(f"\n--- Results for {obstruction_level} ---")
    for name, times in results.items():
        print(f"{name}: Mean={np.mean(times):.4f}s, Std={np.std(times):.4f}s")

    env.close()  # type: ignore[no-untyped-call]
    return results

@pytest.mark.slow
def test_box_vs_baseline_performance() -> None:
    """Compare BOX against baselines (Test 3)."""
    _run_box_vs_baseline_performance("o2")

@pytest.mark.slow
def test_box_o1() -> None:
    """Test BOX approach on obstruction level o1."""
    _run_box_vs_baseline_performance("o1")


@pytest.mark.slow
def test_extensive_visualization() -> None:
    """Run extensive benchmarks on o0-o4 and generate a summary plot."""
    levels = ["o0", "o1", "o2", "o3", "o4"]
    # levels = ['o2'] # For quicker testing during development
    all_results = {}

    # Collect Data
    for level in levels:
        print(f"\n{'='*50}\nRunning Benchmark for {level}\n{'='*50}")
        all_results[level] = _run_box_vs_baseline_performance(level)

    # Generate Visualization
    print("\nGenerating Visualization...")

    # Setup plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), sharey=False)
    fig.suptitle("Planning Approach Performance by Obstruction Complexity", fontsize=16)

    approaches = list(all_results[levels[0]].keys())
    colors = ["b", "g", "r", "c", "m"]

    for i, level in enumerate(levels):
        ax = axes[i]
        data = all_results[level]

        means = [np.mean(data[app]) for app in approaches]
        stds = [np.std(data[app]) for app in approaches]

        # Create bar chart
        bars = ax.bar(approaches, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

        ax.set_title(f"Complexity: {level}")
        ax.set_ylabel("Time (s)")
        ax.set_xticklabels(approaches, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_file = "box_approach_performance_summary.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
