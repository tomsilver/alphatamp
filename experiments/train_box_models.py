"""Hydra script for training BOX models and saving their internal state and pickles.

Adapted from experiments/run_box_matrix_experiment.py.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import hydra
import numpy as np
import prbench
from omegaconf import DictConfig
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.box_approach import BoxApproach, FrozenSkeleton


# --- Serialization Helpers ---

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def serialize_skeleton(skel: FrozenSkeleton) -> dict[str, list[str]]:
    """Convert a FrozenSkeleton into a JSON-compatible dict."""
    states, operators = skel
    return {
        "states": [str(s) for s in states],
        "operators": [str(op) for op in operators],
    }


def serialize_state(state: Any) -> dict[str, Any]:
    """Serialize an ObjectCentricState (or similar) into a JSON-compatible dictionary."""
    serialized_data = {}
    # Assuming ObjectCentricState has .data attribute mapping Object -> features
    if hasattr(state, "data") and isinstance(state.data, dict):
        for obj, features in state.data.items():
            # Use str(obj) as key (e.g., 'block1', 'robot')
            val = features
            if isinstance(val, np.ndarray):
                val = val.tolist()
            serialized_data[str(obj)] = val
    else:
        # Fallback for other state types
        serialized_data["raw_str"] = str(state)
        
    return serialized_data


def serialize_box_data(
    data: list[dict[FrozenSkeleton, tuple[float, bool]]]
) -> list[list[dict[str, Any]]]:
    """Serialize the _data from BoxApproach."""
    serialized_data = []
    for problem_data in data:
        problem_serialized = []
        for skel, (score, success) in problem_data.items():
            problem_serialized.append({
                "skeleton_key": str(skel),
                "score": score,
                "success": bool(success),
            })
        serialized_data.append(problem_serialized)
    return serialized_data


# --- Configuration Helpers ---

def _get_complexity_config(cfg: DictConfig, level: str) -> tuple[float, int]:
    """Get per-level timeout and max_abstract_plans (with approach defaults)."""
    timeout = float(cfg.approach.training_planning_timeout)
    max_abstract_plans = int(cfg.approach.max_abstract_plans)

    if "complexity_configs" not in cfg:
        return timeout, max_abstract_plans

    if level not in cfg.complexity_configs:
        return timeout, max_abstract_plans

    level_cfg = cfg.complexity_configs[level]
    if "timeout" in level_cfg:
        timeout = float(level_cfg.timeout)
    if "max_abstract_plans" in level_cfg:
        max_abstract_plans = int(level_cfg.max_abstract_plans)

    return timeout, max_abstract_plans


def _build_box_approach_for_level(
    cfg: DictConfig,
    env_models: Any,
    level_timeout: float,
    level_max_abstract_plans: int,
    seed: int,
) -> BoxApproach:
    """Create a BoxApproach with per-level complexity settings."""
    return BoxApproach(
        env_models,
        seed=seed,
        max_abstract_plans=level_max_abstract_plans,
        samples_per_step=int(cfg.approach.samples_per_step),
        max_skill_horizon=int(cfg.approach.max_skill_horizon),
        heuristic_name=str(cfg.approach.heuristic_name),
        skeleton_batch_size=int(cfg.approach.skeleton_batch_size),
        num_training_skeletons_per_problem=int(
            cfg.approach.num_training_skeletons_per_problem
        ),
        training_planning_timeout=level_timeout,
        exploration_constant=float(cfg.approach.exploration_constant),
        training_label_mode=str(cfg.approach.training_label_mode),
        failure_penalty_multiplier=float(cfg.approach.failure_penalty_multiplier),
    )


def save_training_results(approach: BoxApproach, output_dir: str, level: str, seed: int) -> None:
    """Save the model pickle and serialized internal state."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Pickle (Only picklable data needed to rebuild)
    # We cannot pickle the full approach because it contains local functions (transition_fn).
    # Instead, we save the hyperparameters and learned data.
    pickle_data = {
        # Hyperparameters
        "seed": approach._seed,
        "max_abstract_plans": approach._max_abstract_plans,
        "samples_per_step": approach._samples_per_step,
        "max_skill_horizon": approach._max_skill_horizon,
        "heuristic_name": approach._heuristic_name,
        "skeleton_batch_size": approach._skeleton_batch_size,
        "num_training_skeletons_per_problem": approach._num_training_skeletons_per_problem,
        "training_planning_timeout": approach._training_planning_timeout,
        "exploration_constant": approach._exploration_constant,
        "training_label_mode": approach._training_label_mode,
        "failure_penalty_multiplier": approach._failure_penalty_multiplier,
        # Learned Model Data
        "_data": approach._data,
        "_training_initial_states": approach._training_initial_states,
        "_skeletons_vocab": approach._skeletons_vocab,
        "_skeleton_to_idx": approach._skeleton_to_idx,
        "_prior_mu": approach._prior_mu,
        "_prior_sigma": approach._prior_sigma,
        "_score_matrix": approach._score_matrix,
    }

    pkl_path = os.path.join(output_dir, "box_approach_data.pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(pickle_data, f)
        print(f"[BoxTraining] Saved model data pickle to {pkl_path}")
    except Exception as e:
        print(f"[BoxTraining] Failed to save pickle: {e}")

    # 2. Serialize and Save Metadata
    # Access protected members to extract internal state
    # pylint: disable=protected-access
    
    # Serialize vocab
    vocab_serialized = []
    if hasattr(approach, "_skeletons_vocab"):
        for skel in approach._skeletons_vocab:
            vocab_serialized.append({
                "key": str(skel),
                "structure": serialize_skeleton(skel)
            })

    # Serialize initial states
    initial_states_serialized = []
    if hasattr(approach, "_training_initial_states"):
        for state in approach._training_initial_states:
            initial_states_serialized.append(serialize_state(state))

    metadata = {
        "level": level,
        "seed": seed,
        "vocab_size": len(vocab_serialized),
        "skeletons_vocab": vocab_serialized,
        "training_data": serialize_box_data(approach._data) if hasattr(approach, "_data") else [],
        "initial_states": initial_states_serialized,
        "matrix_shape": approach._score_matrix.shape if hasattr(approach, "_score_matrix") and approach._score_matrix is not None else None,
        # Matrices (serialized via NumpyEncoder)
        "prior_mu": approach._prior_mu if hasattr(approach, "_prior_mu") else None,
        "prior_sigma": approach._prior_sigma if hasattr(approach, "_prior_sigma") else None,
        "score_matrix": approach._score_matrix if hasattr(approach, "_score_matrix") else None,
    }
    
    json_path = os.path.join(output_dir, "box_model_state.json")
    try:
        with open(json_path, "w") as f:
            json.dump(metadata, f, cls=NumpyEncoder, indent=2)
        print(f"[BoxTraining] Saved model state to {json_path}")
    except Exception as e:
        print(f"[BoxTraining] Failed to save JSON metdata: {e}")


@hydra.main(config_path="conf", config_name="box_matrix_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train BOX per complexity level and save models."""
    prbench.register_all_environments()

    levels: list[str] = [str(x) for x in cfg.levels]
    train_seed_start = int(cfg.train_seed_start)
    num_train_seeds = int(cfg.num_train_seeds)
    seed_arg = int(cfg.seed)

    base_output_dir = str(cfg.output_dir)
    print(f"[BoxTraining] Starting training on levels: {levels}")

    for level in levels:
        num_obstructions = int(level[1:])
        env_id = str(cfg.env.id_template).format(level=level)

        print(
            "[BoxTraining] "
            f"Level={level}, num_obstructions={num_obstructions}, env={env_id}"
        )

        level_timeout, level_max_abstract_plans = _get_complexity_config(cfg, level)
        print(
            "[BoxTraining] "
            f"Complexity settings: timeout={level_timeout}, "
            f"max_abstract_plans={level_max_abstract_plans}"
        )

        env = prbench.make(env_id)
        # Note: assuming prbench naming convention matches run_box_matrix_experiment.py
        model_name = str(cfg.env.model_name)
        
        env_models = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            num_obstructions=num_obstructions,
        )

        # Initialize one approach per level
        approach = _build_box_approach_for_level(
            cfg,
            env_models,
            level_timeout=level_timeout,
            level_max_abstract_plans=level_max_abstract_plans,
            seed=seed_arg
        )

        # Train across the specified range of seeds for this level
        for seed in range(train_seed_start, train_seed_start + num_train_seeds):
            print(f"[BoxTraining] Training on problem seed {seed}...")
            obs, _ = env.reset(seed=seed)
            approach.train(obs)

        # Build the BOX model matrices
        # pylint: disable=protected-access
        approach._build_box_model()

        # Save results under the complexity level folder
        # e.g., output_dir/o1/box_approach.pkl
        level_output_dir = os.path.join(base_output_dir, level)
        save_training_results(approach, level_output_dir, level, seed_arg)
        
        env.close()


if __name__ == "__main__":
    main()