"""Tests for transfer learning with NGramApproach across complexity levels."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import prbench
import pytest
import seaborn as sns
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.ngram_approach import NGramApproach
from alphatamp.approaches.pure_planning_approach import PurePlanningApproach

# Handle conftest import for both pytest and standalone execution
try:
    from conftest import MAKE_VIDEOS
except ModuleNotFoundError:
    # Running as standalone script - add tests directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from conftest import MAKE_VIDEOS
    except ModuleNotFoundError:
        # If still can't import, use default value
        MAKE_VIDEOS = False
        print(
            "Warning: Could not import MAKE_VIDEOS from conftest, defaulting to False"
        )

# Environment configurations: {env_name: [complexity_levels]}
ENV_CONFIGS = {
    "ClutteredRetrieval2D": ["o1", "o10", "o25"],
    "ClutteredStorage2D": ["b1", "b3", "b7", "b15"],
    "DynObstruction2D": ["o0", "o1", "o2", "o3", "o4"],
    "DynPushPullHook": ["o0", "o1", "o5"],
    "Motion2D": ["p0", "p1", "p2", "p3", "p4", "p5"],
    "Obstruction2D": ["o0", "o1", "o2", "o3", "o4"],
    "StickButton2D": ["b1", "b2", "b3", "b5", "b10"],
}

# Default configuration
DEFAULT_ENV_NAME = "Obstruction2D"
COMPLEXITY_LEVELS = ENV_CONFIGS[DEFAULT_ENV_NAME]  # Default to Obstruction2D

NUM_TEST_SEEDS = 10  # Number of test seeds for averaging
NUM_TRAIN_PROBLEMS = 10  # Number of training problems per complexity level
TRAIN_SEED_START = 0
TEST_SEED_START = 100  # Avoid overlap with training seeds

# Timeouts (in seconds)
TRAINING_TIMEOUT = 30
DEFAULT_TEST_TIMEOUT = 60
DEFAULT_MAX_ABSTRACT_PLANS = 50

# Number of samples per planning step
DEFAULT_SAMPLES_PER_STEP = 2

# Environment-specific timeouts (optional, uses defaults if not specified)
TEST_TIMEOUT_BY_COMPLEXITY: dict[str, float] = {
    "o0": 30,
    "o1": 60,
    "o2": 90,
    "o3": 120,
    "o4": 150,
}

# Environment-specific planning budgets
MAX_ABSTRACT_PLANS_BY_COMPLEXITY: dict[str, int] = {
    "o0": 20,
    "o1": 30,
    "o2": 40,
    "o3": 50,
    "o4": 60,
}


def _measure_ttf_and_execute(
    approach: Any, env: Any, obs: Any, timeout: float = 60.0
) -> float:
    """Measure TTF and execute the plan in the environment (for video generation)."""
    start = time.perf_counter()
    try:
        plan = approach.run_planning(obs, timeout=timeout)
        ttf = time.perf_counter() - start

        # Execute plan in environment to generate video frames
        if plan is not None:
            for action in plan.actions:
                _, _, done, _, _ = env.step(action)
                if done:
                    break

        return ttf
    except TimeoutError:
        return float("inf")


def _create_env_and_models(
    environment_name: str, complexity: str, render_mode: str | None = None
):
    """Create environment and models for any supported environment type.

    Args:
        environment_name: Environment name (e.g., "Obstruction2D")
        complexity: Complexity level (e.g., "o0", "b3", "p2")
        render_mode: Optional render mode ("rgb_array" or None)

    Returns:
        Tuple of (env, env_models)
    """
    full_env_name = f"prbench/{environment_name}-{complexity}-v0"

    if render_mode:
        env = prbench.make(full_env_name, render_mode=render_mode)
    else:
        env = prbench.make(full_env_name)

    # Convert environment name to lowercase for model creation
    env_models = create_bilevel_planning_models(
        environment_name.lower(),
        env.observation_space,
        env.action_space,
        # Parse complexity parameter based on environment type
        **_parse_complexity_params(environment_name, complexity),
    )

    return env, env_models


def _parse_complexity_params(environment_name: str, complexity: str) -> dict[str, Any]:
    """Parse complexity string into kwargs for models.

    Args:
        environment_name: Environment name
        complexity: Complexity string

    Returns:
        Dictionary of kwargs for model creation
    """
    # Extract numeric value from complexity string
    complexity_value = int(complexity[1:])

    # Map environment names to their parameter names
    param_mapping = {
        "Obstruction2D": "num_obstructions",
        "DynObstruction2D": "num_obstructions",
        "ClutteredRetrieval2D": "num_obstructions",
        "ClutteredStorage2D": "num_blocks",
        "DynPushPullHook": "num_obstructions",
        "Motion2D": "num_polygons",
        "StickButton2D": "num_buttons",
    }

    param_name = param_mapping.get(environment_name)
    if param_name:
        return {param_name: complexity_value}
    return {}


def _train_approach_helper(
    approach: NGramApproach,
    environment_name: str,
    train_complexity: str,
    num_train_problems: int = NUM_TRAIN_PROBLEMS,
) -> None:
    """Train an approach on multiple problems of given complexity."""
    for i in range(num_train_problems):
        train_env, _ = _create_env_and_models(
            environment_name, train_complexity, render_mode="rgb_array"
        )
        train_obs, _ = train_env.reset(seed=TRAIN_SEED_START + i)
        approach.train(train_obs)
        train_env.close()


def _test_approach_on_seed(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    approach: Any,
    approach_name: str,
    environment_name: str,
    test_complexity: str,
    test_seed: int,
    video_folder: str | None = None,
) -> float:
    """Test an approach on a single seed (helper function)."""
    test_env, _ = _create_env_and_models(
        environment_name, test_complexity, render_mode="rgb_array"
    )

    if MAKE_VIDEOS and video_folder:
        test_env = RecordVideo(
            test_env,
            video_folder,
            name_prefix=f"{approach_name}_seed{test_seed}",
        )

    test_obs, _ = test_env.reset(seed=test_seed)
    timeout = TEST_TIMEOUT_BY_COMPLEXITY.get(test_complexity, DEFAULT_TEST_TIMEOUT)

    ttf = _measure_ttf_and_execute(approach, test_env, test_obs, timeout=timeout)
    test_env.close()

    return ttf


def _compute_average_ttf(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    approach: Any,
    approach_name: str,
    environment_name: str,
    test_complexity: str,
    num_seeds: int = NUM_TEST_SEEDS,
    video_base_folder: str | None = None,
) -> tuple[float, list[float]]:
    """Compute average TTF over multiple test seeds."""
    ttfs = []
    timeout = TEST_TIMEOUT_BY_COMPLEXITY.get(test_complexity, DEFAULT_TEST_TIMEOUT)

    for i in range(num_seeds):
        test_seed = TEST_SEED_START + i

        # Video folder for this seed
        if video_base_folder:
            video_folder = f"{video_base_folder}/seed{test_seed}"
        else:
            video_folder = None

        ttf = _test_approach_on_seed(
            approach,
            approach_name,
            environment_name,
            test_complexity,
            test_seed,
            video_folder,
        )
        ttfs.append(ttf)
        # Check if timeout occurred (ttf >= timeout means planning failed)
        if ttf >= timeout:
            print(f"Seed {test_seed}: TIMEOUT (>={timeout}s)")
        else:
            print(f"Seed {test_seed}: {ttf:.2f}s")

    # Compute average, treating timeouts (ttf >= timeout) as the timeout value
    ttfs_for_avg = [min(t, timeout) for t in ttfs]
    avg_ttf = float(np.mean(ttfs_for_avg))

    return avg_ttf, ttfs


def save_heatmap(
    ttf_ratio_matrix: np.ndarray,
    output_path: str,
    title: str = "Transfer Learning TTF Ratio Matrix",
    complexity_levels: list[str] | None = None,
) -> None:
    """Generate and save heatmap visualization.

    Args:
        ttf_ratio_matrix: Matrix of TTF ratios
        output_path: Path to save the heatmap
        title: Title for the heatmap
        complexity_levels: List of complexity level labels (e.g., ["o0", "o1", ...])
    """
    if complexity_levels is None:
        complexity_levels = [str(i) for i in range(len(ttf_ratio_matrix))]

    plt.figure(figsize=(10, 8))

    # Create heatmap with custom colormap
    # Values < 1.0 = faster (green), > 1.0 = slower (red)
    sns.heatmap(
        ttf_ratio_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",  # Red for bad, yellow for neutral, green for good
        center=1.0,  # Center colormap at 1.0 (no change)
        vmin=0.5,  # 2x faster
        vmax=2.0,  # 2x slower
        xticklabels=complexity_levels,
        yticklabels=complexity_levels,
        cbar_kws={"label": "TTF Ratio (Learned / Pure)\n← Faster | Slower →"},
    )

    plt.xlabel("Test Complexity", fontsize=12, fontweight="bold")
    plt.ylabel("Train Complexity", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.figtext(
        0.5,
        0.02,
        "Values < 1.0 = learned is faster, > 1.0 = learned is slower",
        ha="center",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_path}")


@pytest.mark.slow
def test_transfer_matrix(  # pylint: disable=too-many-locals
    environment: str | None = None,
) -> None:
    """Pairwise transfer evaluation across complexity levels.

    Args:
        environment: Environment name. If None, uses DEFAULT_ENV_NAME.
    """
    if environment is None:
        environment = DEFAULT_ENV_NAME

    if environment not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown environment: {environment}. "
            f"Supported environments: {list(ENV_CONFIGS.keys())}"
        )

    complexity_levels = ENV_CONFIGS[environment]

    prbench.register_all_environments()

    # Results storage
    num_levels = len(complexity_levels)
    learned_ttfs = np.zeros((num_levels, num_levels))
    pure_ttfs = np.zeros((num_levels, num_levels))
    ttf_ratio_matrix = np.zeros((num_levels, num_levels))

    output_dir = Path(f"unit_test_videos/transfer_matrix/{environment.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"TRANSFER MATRIX EVALUATION - {environment}")
    print("=" * 70)

    # PHASE 1: TRAIN ONCE PER COMPLEXITY
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)
    print(f"Environment: {environment}")
    print(f"Complexities: {complexity_levels}")
    print(f"Training problems per complexity: {NUM_TRAIN_PROBLEMS}")
    print("=" * 70)

    trained_ngram_stats = {}

    for train_complexity in complexity_levels:
        print(
            f"\nTraining on {train_complexity} " f"({NUM_TRAIN_PROBLEMS} problems)..."
        )

        # Create environment models for training
        _, train_env_models = _create_env_and_models(environment, train_complexity)

        # Create temporary approach for training
        train_max_plans = MAX_ABSTRACT_PLANS_BY_COMPLEXITY.get(
            train_complexity, DEFAULT_MAX_ABSTRACT_PLANS
        )

        # IMPORTANT: grounded or lifted, failure/success mode
        train_approach: NGramApproach = NGramApproach(
            train_env_models,
            seed=123,
            samples_per_step=DEFAULT_SAMPLES_PER_STEP,
            training_planning_timeout=TRAINING_TIMEOUT,
            max_abstract_plans=train_max_plans,
            use_grounded_ngrams=False,
            failure_penalty_mode=True,
        )

        # Train on multiple problems
        _train_approach_helper(
            train_approach, environment, train_complexity, NUM_TRAIN_PROBLEMS
        )

        # Cache learned patterns
        # pylint: disable=protected-access
        trained_ngram_stats[train_complexity] = train_approach._ngram_stats.copy()

        ngram_count = len(trained_ngram_stats[train_complexity])
        print(f"Learned {ngram_count} unique n-grams")

    # PHASE 2: TEST PURE PLANNING ONCE PER TEST COMPLEXITY
    print("\n" + "=" * 70)
    print("PHASE 2: TESTING PURE PLANNING BASELINE")
    print("=" * 70)
    print(f"Environment: {environment}")
    print(f"Test complexities: {complexity_levels}")
    print(f"Test seeds per complexity: {NUM_TEST_SEEDS}")
    print("=" * 70)

    # Cache pure planning results for each test complexity
    pure_planning_cache = {}

    for test_complexity in complexity_levels:
        print(f"\nTesting Pure Planning on {test_complexity}:")

        # Create environment models for testing
        _, test_env_models = _create_env_and_models(environment, test_complexity)

        # Create pure planning baseline
        max_plans = MAX_ABSTRACT_PLANS_BY_COMPLEXITY.get(
            test_complexity, DEFAULT_MAX_ABSTRACT_PLANS
        )
        pure_approach: PurePlanningApproach = PurePlanningApproach(
            test_env_models,
            seed=123,
            samples_per_step=DEFAULT_SAMPLES_PER_STEP,
            max_abstract_plans=max_plans,
        )

        # Test pure planning approach (only once per test complexity!)
        video_folder_pure = None
        if MAKE_VIDEOS:
            base = f"unit_test_videos/transfer_matrix/{environment.lower()}"
            video_folder_pure = f"{base}/pure_{test_complexity}/Pure_Planning"

        avg_pure, ttfs_pure = _compute_average_ttf(
            pure_approach,
            "pure",
            environment,
            test_complexity,
            NUM_TEST_SEEDS,
            video_folder_pure,
        )

        # Cache results for reuse across all training complexities
        pure_planning_cache[test_complexity] = {
            "avg_ttf": avg_pure,
            "ttfs": ttfs_pure,
        }

        print(f"Pure Planning avg: {avg_pure:.2f}s")

    # PHASE 3: TEST LEARNED APPROACHES
    print("\n" + "=" * 70)
    print("PHASE 3: TESTING LEARNED APPROACHES")
    print("=" * 70)
    print(f"Environment: {environment}")
    print(f"Train complexities: {complexity_levels}")
    print(f"Test complexities: {complexity_levels}")
    print(f"Test seeds per pair: {NUM_TEST_SEEDS}")
    total_tests = num_levels * num_levels
    msg = f"Total test cases: {total_tests} learned tests (pure cached)"
    print(msg)
    print("=" * 70)

    # Iterate over all (train, test) pairs
    for train_idx, train_complexity in enumerate(complexity_levels):
        for test_idx, test_complexity in enumerate(complexity_levels):
            pair_name = f"train_{train_complexity}_test_{test_complexity}"

            print(
                f"\nEvaluating: Train on {train_complexity} "
                f"-> Test on {test_complexity}"
            )

            # Create environment models for testing
            _, test_env_models = _create_env_and_models(environment, test_complexity)

            max_plans = MAX_ABSTRACT_PLANS_BY_COMPLEXITY.get(
                test_complexity, DEFAULT_MAX_ABSTRACT_PLANS
            )

            # Create learned approach and transfer pre-trained patterns
            learned_approach: NGramApproach = NGramApproach(
                test_env_models,
                seed=123,
                samples_per_step=DEFAULT_SAMPLES_PER_STEP,
                training_planning_timeout=TRAINING_TIMEOUT,
                max_abstract_plans=max_plans,
                use_grounded_ngrams=False,
                failure_penalty_mode=True,
            )
            # Transfer cached patterns (no training!)
            # pylint: disable=protected-access
            learned_approach._ngram_stats = trained_ngram_stats[train_complexity].copy()

            # Test learned approach
            print(f"\nTesting Learned approach on {test_complexity}:")
            num_ngrams = len(learned_approach._ngram_stats)
            print(f"Using {num_ngrams} n-grams from " f"{train_complexity} training")
            video_folder_learned = None
            if MAKE_VIDEOS:
                base = f"unit_test_videos/transfer_matrix/{environment.lower()}"
                video_folder_learned = f"{base}/{pair_name}/Learned"

            avg_learned, _ = _compute_average_ttf(
                learned_approach,
                "learned",
                environment,
                test_complexity,
                NUM_TEST_SEEDS,
                video_folder_learned,
            )

            # Reuse cached pure planning results
            cached_pure = pure_planning_cache[test_complexity]
            avg_pure_cached = cast(float, cached_pure["avg_ttf"])
            print(f"Pure Planning avg: {avg_pure_cached:.2f}s (cached)")

            # Store results
            learned_ttfs[train_idx, test_idx] = avg_learned
            pure_ttfs[train_idx, test_idx] = avg_pure_cached

            # Calculate TTF ratio (learned/pure)
            if avg_pure_cached != float("inf") and avg_pure_cached > 0:
                ttf_ratio = avg_learned / avg_pure_cached
            else:
                ttf_ratio = float("inf") if avg_learned == float("inf") else 0.0

            ttf_ratio_matrix[train_idx, test_idx] = ttf_ratio

    # Save heatmap
    heatmap_path = output_dir / "ttf_ratio_heatmap.png"
    title = (
        f"{environment} N-Gram Transfer Learning: "
        f"TTF Ratio (Learned / Pure Planning)"
    )
    save_heatmap(
        ttf_ratio_matrix,
        str(heatmap_path),
        title=title,
        complexity_levels=complexity_levels,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run transfer matrix evaluation for N-Gram approach"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV_NAME,
        choices=list(ENV_CONFIGS.keys()),
        help=f"Environment name (default: {DEFAULT_ENV_NAME})",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available environments and their complexities",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TEST_TIMEOUT,
        help="Default timeout for test planning (seconds)",
    )
    parser.add_argument(
        "--samples-per-step",
        type=int,
        default=2,
        help="Number of samples per planning step (default: 2)",
    )

    args = parser.parse_args()
    DEFAULT_TEST_TIMEOUT = args.timeout
    DEFAULT_SAMPLES_PER_STEP = args.samples_per_step

    if args.list_envs:
        print("Available environments and complexities:")
        print("=" * 60)
        for env_name, complexities in sorted(ENV_CONFIGS.items()):
            print(f"{env_name:25s} {', '.join(complexities)}")
        print("=" * 60)
    else:
        test_transfer_matrix(environment=args.env)
