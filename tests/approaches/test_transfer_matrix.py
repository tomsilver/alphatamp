import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import prbench
import pytest
import seaborn as sns
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.ngram_approach import NGramApproach
from alphatamp.approaches.pure_planning_approach import PurePlanningApproach

COMPLEXITY_LEVELS = list(range(5)) # Obstruction levels 0 to 4
NUM_TEST_SEEDS = 10 # Number of test seeds for averaging
NUM_TRAIN_PROBLEMS = 10 # Number of training problems per complexity level
TRAIN_SEED_START = 0
TEST_SEED_START = 100  # Avoid overlap with training seeds

# Timeouts (in seconds)
TRAINING_TIMEOUT = 30 
TEST_TIMEOUT_BY_COMPLEXITY = {
    0: 30,
    1: 60,
    2: 90,
    3: 120,
    4: 150,
}

# Planning budgets
MAX_ABSTRACT_PLANS_BY_COMPLEXITY = {
    0: 20,
    1: 30,
    2: 40,
    3: 50,
    4: 60,
}

def _measure_ttf_and_execute(
    approach: Any, env: Any, obs: Any, timeout: float = 60.0
) -> float:
    """
    Measure TTF and execute the plan in the environment (for video generation).
    """
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


def _create_obstruction_env_and_models(num_obstructions: int, render_mode: str = None):
    """
    Create Obstruction2D environment and models.
    """
    env_name = f"prbench/Obstruction2D-o{num_obstructions}-v0"
    
    if render_mode:
        env = prbench.make(env_name, render_mode=render_mode)
    else:
        env = prbench.make(env_name)
    
    env_models = create_bilevel_planning_models(
        "obstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=num_obstructions,
    )
    
    return env, env_models


def _train_approach_helper(
    approach: NGramApproach,
    train_complexity: int,
    num_train_problems: int = NUM_TRAIN_PROBLEMS,
) -> None:
    """
    Train an approach on multiple problems of given complexity.
    """
    for i in range(num_train_problems):
        train_env, _ = _create_obstruction_env_and_models(
            train_complexity, render_mode="rgb_array"
        )
        train_obs, _ = train_env.reset(seed=TRAIN_SEED_START + i)
        approach.train(train_obs)
        train_env.close()


def _test_approach_on_seed(
    approach: Any,
    approach_name: str,
    test_complexity: int,
    test_seed: int,
    video_folder: str = None,
) -> float:
    """
    Test an approach on a single seed (helper function).
    """
    test_env, _ = _create_obstruction_env_and_models(
        test_complexity, render_mode="rgb_array"
    )
    
    if MAKE_VIDEOS and video_folder:
        test_env = RecordVideo(
            test_env,
            video_folder,
            name_prefix=f"{approach_name}_seed{test_seed}",
        )
    
    test_obs, _ = test_env.reset(seed=test_seed)
    timeout = TEST_TIMEOUT_BY_COMPLEXITY[test_complexity]
    
    ttf = _measure_ttf_and_execute(approach, test_env, test_obs, timeout=timeout)
    test_env.close()
    
    return ttf


def _compute_average_ttf(
    approach: Any,
    approach_name: str,
    test_complexity: int,
    num_seeds: int = NUM_TEST_SEEDS,
    video_base_folder: str = None,
) -> tuple[float, list[float]]:
    """
    Compute average TTF over multiple test seeds.
    """
    ttfs = []
    timeout = TEST_TIMEOUT_BY_COMPLEXITY[test_complexity]
    
    for i in range(num_seeds):
        test_seed = TEST_SEED_START + i
        
        # Video folder for this seed
        if video_base_folder:
            video_folder = f"{video_base_folder}/seed{test_seed}"
        else:
            video_folder = None
        
        ttf = _test_approach_on_seed(
            approach, approach_name, test_complexity, test_seed, video_folder
        )
        ttfs.append(ttf)
        # Check if timeout occurred (ttf >= timeout means planning failed)
        if ttf >= timeout:
            print(f"Seed {test_seed}: TIMEOUT (>={timeout}s)")
        else:
            print(f"Seed {test_seed}: {ttf:.2f}s")
    
    # Compute average, treating timeouts (ttf >= timeout) as the timeout value
    ttfs_for_avg = [min(t, timeout) for t in ttfs]
    avg_ttf = np.mean(ttfs_for_avg)
    
    return avg_ttf, ttfs


def save_heatmap(
    ttf_ratio_matrix: np.ndarray,
    output_path: str,
    title: str = "Transfer Learning TTF Ratio Matrix",
) -> None:
    """
    Generate and save heatmap visualization.
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    # Values < 1.0 = faster (green), > 1.0 = slower (red)
    ax = sns.heatmap(
        ttf_ratio_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",  # Red for bad, yellow for neutral, green for good
        center=1.0,  # Center colormap at 1.0 (no change)
        vmin=0.5,  # 2x faster
        vmax=2.0,  # 2x slower
        xticklabels=[f"o{i}" for i in COMPLEXITY_LEVELS],
        yticklabels=[f"o{i}" for i in COMPLEXITY_LEVELS],
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
def test_transfer_matrix() -> None:
    """
    Pairwise transfer evaluation across complexity levels.
    """
    prbench.register_all_environments()
    
    # Results storage
    num_levels = len(COMPLEXITY_LEVELS)
    learned_ttfs = np.zeros((num_levels, num_levels))  # [train][test]
    pure_ttfs = np.zeros((num_levels, num_levels))
    ttf_ratio_matrix = np.zeros((num_levels, num_levels))
    
    output_dir = Path("unit_test_videos/transfer_matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TRANSFER MATRIX EVALUATION")
    print("=" * 70)
    
    # PHASE 1: TRAIN ONCE PER COMPLEXITY
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)
    print(f"Complexities: {COMPLEXITY_LEVELS}")
    print(f"Training problems per complexity: {NUM_TRAIN_PROBLEMS}")
    print("=" * 70)
    
    trained_ngram_stats = {}
    
    for train_complexity in COMPLEXITY_LEVELS:
        print(f"\nTraining on o{train_complexity} ({NUM_TRAIN_PROBLEMS} problems)...")
        
        # Create environment models for training
        _, train_env_models = _create_obstruction_env_and_models(train_complexity)
        
        # Create temporary approach for training
        train_max_plans = MAX_ABSTRACT_PLANS_BY_COMPLEXITY[train_complexity]

        # IMPORTANT: Check whether we are using grounded or lifted, failure/success mode
        # TODO: Add parameters to test both modes?
        train_approach = NGramApproach(
            train_env_models,
            seed=123,
            samples_per_step=2,
            training_planning_timeout=TRAINING_TIMEOUT,
            max_abstract_plans=train_max_plans,
            use_grounded_ngrams=False, # NOTE: Can use grounded or lifted here!
            failure_penalty_mode=True
        )
        
        # Train on multiple problems
        _train_approach_helper(train_approach, train_complexity, NUM_TRAIN_PROBLEMS)
        
        # Cache learned patterns
        trained_ngram_stats[train_complexity] = train_approach._ngram_stats.copy()
        
        print(f"Learned {len(trained_ngram_stats[train_complexity])} unique n-grams")
    
    # PHASE 2: TEST PURE PLANNING ONCE PER TEST COMPLEXITY
    print("\n" + "=" * 70)
    print("PHASE 2: TESTING PURE PLANNING BASELINE")
    print("=" * 70)
    print(f"Test complexities: {COMPLEXITY_LEVELS}")
    print(f"Test seeds per complexity: {NUM_TEST_SEEDS}")
    print("=" * 70)
    
    # Cache pure planning results for each test complexity
    pure_planning_cache = {}
    
    for test_complexity in COMPLEXITY_LEVELS:
        print(f"\nTesting Pure Planning on o{test_complexity}:")
        
        # Create environment models for testing
        _, test_env_models = _create_obstruction_env_and_models(test_complexity)
        
        # Create pure planning baseline
        max_plans = MAX_ABSTRACT_PLANS_BY_COMPLEXITY[test_complexity]
        pure_approach = PurePlanningApproach(
            test_env_models,
            seed=123,
            samples_per_step=2,
            max_abstract_plans=max_plans,
        )
        
        # Test pure planning approach (only once per test complexity!)
        video_folder_pure = (
            f"unit_test_videos/transfer_matrix/pure_o{test_complexity}/Pure_Planning"
            if MAKE_VIDEOS
            else None
        )
        avg_pure, ttfs_pure = _compute_average_ttf(
            pure_approach,
            "pure",
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
    print(f"Train complexities: {COMPLEXITY_LEVELS}")
    print(f"Test complexities: {COMPLEXITY_LEVELS}")
    print(f"Test seeds per pair: {NUM_TEST_SEEDS}")
    print(f"Total test cases: {num_levels * num_levels} learned tests (pure already cached)")
    print("=" * 70)
    
    # Iterate over all (train, test) pairs
    for train_idx, train_complexity in enumerate(COMPLEXITY_LEVELS):
        for test_idx, test_complexity in enumerate(COMPLEXITY_LEVELS):
            pair_name = f"train_o{train_complexity}_test_o{test_complexity}"
            
            print(f"Evaluating: Train on o{train_complexity} -> Test on o{test_complexity}")
            
            # Create environment models for testing
            _, test_env_models = _create_obstruction_env_and_models(test_complexity)
            
            max_plans = MAX_ABSTRACT_PLANS_BY_COMPLEXITY[test_complexity]
            
            # Create learned approach and transfer pre-trained patterns
            learned_approach = NGramApproach(
                test_env_models,
                seed=123,
                samples_per_step=2,
                training_planning_timeout=TRAINING_TIMEOUT,
                max_abstract_plans=max_plans,
                use_grounded_ngrams=False,
                failure_penalty_mode=True
            )
            # Transfer cached patterns (no training!)
            learned_approach._ngram_stats = trained_ngram_stats[train_complexity].copy()
            
            # Test learned approach
            print(f"\nTesting Learned approach on o{test_complexity}:")
            print(f"Using {len(learned_approach._ngram_stats)} n-grams from o{train_complexity} training")
            video_folder_learned = (
                f"unit_test_videos/transfer_matrix/{pair_name}/Learned"
                if MAKE_VIDEOS
                else None
            )
            avg_learned, ttfs_learned = _compute_average_ttf(
                learned_approach,
                "learned",
                test_complexity,
                NUM_TEST_SEEDS,
                video_folder_learned,
            )
            
            # Reuse cached pure planning results
            avg_pure = pure_planning_cache[test_complexity]["avg_ttf"]
            ttfs_pure = pure_planning_cache[test_complexity]["ttfs"]
            print(f"Pure Planning avg: {avg_pure:.2f}s (cached)")
            
            # Store results
            learned_ttfs[train_idx, test_idx] = avg_learned
            pure_ttfs[train_idx, test_idx] = avg_pure
            
            # Calculate TTF ratio (learned/pure)
            if avg_pure != float("inf") and avg_pure > 0:
                ttf_ratio = avg_learned / avg_pure
            else:
                ttf_ratio = float("inf") if avg_learned == float("inf") else 0.0
            
            ttf_ratio_matrix[train_idx, test_idx] = ttf_ratio
    
    # Save heatmap
    heatmap_path = output_dir / "ttf_ratio_heatmap.png"
    save_heatmap(
        ttf_ratio_matrix,
        str(heatmap_path),
        title="N-Gram Transfer Learning: TTF Ratio (Learned / Pure Planning)",
    )

if __name__ == "__main__":
    test_transfer_matrix()
