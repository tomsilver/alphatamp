"""Quick integrity checks for encoder dataset pickle artifacts.

Usage:
    python experiments/check_encoder_dataset.py
    python experiments/check_encoder_dataset.py --artifacts-dir artifacts/encoder_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import kinder
import numpy as np
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

try:
    import dill
except ImportError as exc:
    raise ImportError(
        "check_encoder_dataset.py requires dill. Install with: pip install dill"
    ) from exc


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        obj = dill.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(obj)}")
    return obj


def _is_binary_matrix(arr: np.ndarray) -> bool:
    return bool(np.all((arr == 0.0) | (arr == 1.0)))


def _check_split_payload(path: Path) -> None:
    payload = _load_pickle(path)
    required_top = {"split", "seed_ids", "config", "dataset"}
    missing_top = required_top - set(payload)
    if missing_top:
        raise KeyError(f"{path.name}: missing top-level keys: {sorted(missing_top)}")

    split = payload["split"]
    seed_ids = payload["seed_ids"]
    dataset = payload["dataset"]
    config = payload["config"]

    if not isinstance(dataset, dict):
        raise TypeError(f"{path.name}: dataset field must be dict")

    required_dataset = {
        "seed_ids",
        "op_sequence_vocab",
        "applicability",
        "success",
        "refinement_time",
        "initial_low_level_states",
        "initial_abstract_states",
        "problem_goals",
    }
    missing_dataset = required_dataset - set(dataset)
    if missing_dataset:
        raise KeyError(f"{path.name}: missing dataset keys: {sorted(missing_dataset)}")

    ds_seed_ids = dataset["seed_ids"]
    op_vocab = dataset["op_sequence_vocab"]
    applicability = np.asarray(dataset["applicability"])
    success = np.asarray(dataset["success"])
    refinement_time = np.asarray(dataset["refinement_time"])

    n = len(ds_seed_ids)
    m = len(op_vocab)

    if len(seed_ids) != n:
        raise ValueError(
            f"{path.name}: payload seed count {len(seed_ids)} != dataset seed count {n}"
        )
    if list(seed_ids) != list(ds_seed_ids):
        raise ValueError(f"{path.name}: payload seed_ids and dataset seed_ids differ")

    expected_shape = (n, m)
    if applicability.shape != expected_shape:
        raise ValueError(
            f"{path.name}: applicability shape {applicability.shape} != {expected_shape}"
        )
    if success.shape != expected_shape:
        raise ValueError(
            f"{path.name}: success shape {success.shape} != {expected_shape}"
        )
    if refinement_time.shape != expected_shape:
        raise ValueError(
            f"{path.name}: refinement_time shape {refinement_time.shape} != {expected_shape}"
        )

    if not _is_binary_matrix(applicability):
        raise ValueError(f"{path.name}: applicability must be binary (0/1)")
    if not _is_binary_matrix(success):
        raise ValueError(f"{path.name}: success must be binary (0/1)")

    timeout = float(config["training_planning_timeout"])
    inapplicable = applicability == 0.0

    if np.any(success[inapplicable] != 0.0):
        raise ValueError(
            f"{path.name}: found inapplicable entries with nonzero success"
        )

    if np.any(~np.isclose(refinement_time[inapplicable], timeout)):
        raise ValueError(
            f"{path.name}: found inapplicable entries with refinement_time != timeout"
        )

    if np.any(success > applicability):
        raise ValueError(f"{path.name}: found success=1 where applicability=0")

    if len(dataset["initial_low_level_states"]) != n:
        raise ValueError(f"{path.name}: initial_low_level_states length != N")
    if len(dataset["initial_abstract_states"]) != n:
        raise ValueError(f"{path.name}: initial_abstract_states length != N")
    if len(dataset["problem_goals"]) != n:
        raise ValueError(f"{path.name}: problem_goals length != N")

    applicable_count = int(np.sum(applicability))
    success_count = int(np.sum(success))
    avg_time = float(np.mean(refinement_time)) if refinement_time.size else 0.0
    print(
        f"[OK] {path.name:<30} split={split:<10} "
        f"N={n:<4} M={m:<4} applicable={applicable_count:<6} "
        f"success={success_count:<6} avg_time={avg_time:.2f}s"
    )


def _check_vocab_payload(path: Path) -> None:
    payload = _load_pickle(path)
    required = {"vocabulary", "config", "vocab_seed_ids"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"{path.name}: missing keys: {sorted(missing)}")

    vocabulary = payload["vocabulary"]
    config = payload["config"]
    vocab_size = int(config["vocabulary_size"])

    if not isinstance(vocabulary, list):
        raise TypeError(f"{path.name}: vocabulary must be a list")
    if len(vocabulary) > vocab_size:
        raise ValueError(
            f"{path.name}: vocabulary length {len(vocabulary)} exceeds configured size {vocab_size}"
        )

    print(
        f"[OK] {path.name:<30} vocab_len={len(vocabulary)} "
        f"configured_k={vocab_size}"
    )


def _bootstrap_env_model_modules(config: dict[str, Any]) -> None:
    """Register dynamic env-model modules needed for dill deserialization.

    Some serialized objects reference modules generated during
    create_bilevel_planning_models(...) (e.g., obstruction2d_module).
    """
    env_id = str(config["env_id"])
    model_name = str(config["model_name"])
    num_obstructions = int(config["num_obstructions"])

    kinder.register_all_environments()
    env = kinder.make(env_id)
    try:
        _ = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            num_obstructions=num_obstructions,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate encoder dataset pickle artifacts"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts") / "encoder_dataset",
        help="Directory containing encoder_*_dataset.pkl and encoder_vocab.pkl",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    split_files = [
        artifacts_dir / "encoder_validation_dataset.pkl",
        artifacts_dir / "encoder_test_dataset.pkl",
        artifacts_dir / "encoder_train_dataset.pkl",
    ]
    vocab_file = artifacts_dir / "encoder_vocab.pkl"

    for file_path in split_files + [vocab_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing expected artifact: {file_path}")

    print(f"Checking artifacts in: {artifacts_dir}")
    vocab_payload = _load_pickle(vocab_file)
    _check_vocab_payload(vocab_file)
    _bootstrap_env_model_modules(vocab_payload["config"])
    for split_file in split_files:
        _check_split_payload(split_file)
    print("All checks passed.")


if __name__ == "__main__":
    main()
