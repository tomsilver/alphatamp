"""Quick sanity-check for an encoder train dataset artifact.

Usage:
    python experiments/check_encoder_train_dataset.py [path/to/encoder_train_dataset.pkl]

Defaults to artifacts/encoder_o2_small/encoder_train_dataset.pkl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import dill
import kinder
import numpy as np
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

ARTIFACT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "artifacts/encoder_o2_small/encoder_train_dataset.pkl"
)

# Vocab pickle that carries the config (plain Python objects — safe to load
# without pre-bootstrapping env-model modules).
VOCAB_FILE = Path(
    sys.argv[2] if len(sys.argv) > 2 else "artifacts/encoder_o2/vocab.pkl"
)

PAYLOAD_KEYS = {"split", "seed_ids", "config", "dataset"}
DATASET_KEYS = {
    "seed_ids",
    "op_sequence_vocab",
    "applicability",
    "success",
    "refinement_time",
    "initial_low_level_states",
    "initial_abstract_states",
    "problem_goals",
}


def check(condition: bool, msg: str) -> None:
    """Print a PASS/FAIL check line and raise on failure."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise AssertionError(msg)


def _bootstrap_env_modules(config: dict) -> None:
    """Register dynamic env-model modules so dill can deserialize env objects."""
    kinder.register_all_environments()
    env = kinder.make(config["env_id"])
    try:
        create_bilevel_planning_models(
            config["model_name"],
            env.observation_space,
            env.action_space,
            num_obstructions=config["num_obstructions"],
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]


def main() -> None:
    """Load and validate a train split dataset pickle against its vocabulary config."""
    # Step 1: load vocab pkl to obtain config (plain Python objects; safe before
    # bootstrapping env-model modules).
    print(f"Loading config from vocab file {VOCAB_FILE} ...")
    with open(VOCAB_FILE, "rb") as f:
        vocab_payload = dill.load(f)
    config = vocab_payload["config"]
    print(f"       config: {config}")

    # Step 2: bootstrap dynamic env-model modules so dill can deserialize
    # env-object references inside the dataset pkl.
    print("\nBootstrapping env model modules ...")
    _bootstrap_env_modules(config)

    # Step 3: load the actual dataset.
    print(f"\nLoading dataset from {ARTIFACT} ...")
    with open(ARTIFACT, "rb") as f:
        payload = dill.load(f)

    # ---- 1. Top-level payload keys ----
    print("\n[1] Checking top-level payload keys ...")
    missing_payload = PAYLOAD_KEYS - set(payload.keys())
    check(
        not missing_payload,
        f"No missing top-level keys (found: {sorted(payload.keys())})",
    )
    print(f"       split: {payload['split']}")

    ds = payload["dataset"]
    outer_seed_ids = payload["seed_ids"]

    # ---- 2. Dataset keys ----
    print("\n[2] Checking dataset keys ...")
    missing_ds = DATASET_KEYS - set(ds.keys())
    check(not missing_ds, f"No missing dataset keys (found: {sorted(ds.keys())})")

    # ---- 3. seed_ids ----
    print("\n[3] Checking seed_ids ...")
    seed_ids = ds["seed_ids"]
    check(
        isinstance(seed_ids, list),
        f"seed_ids is a list (got {type(seed_ids).__name__})",
    )
    check(len(seed_ids) > 0, f"seed_ids is non-empty (len={len(seed_ids)})")
    check(
        sorted(seed_ids) == sorted(outer_seed_ids),
        f"inner seed_ids match outer seed_ids: {seed_ids} vs {outer_seed_ids}",
    )
    print(f"       seed_ids ({len(seed_ids)} entries): {seed_ids}")

    # ---- 4. vocab ----
    print("\n[4] Checking op_sequence_vocab ...")
    vocab = ds["op_sequence_vocab"]
    check(vocab is not None, "vocab is not None")
    vocab_size = len(vocab) if hasattr(vocab, "__len__") else "?"
    print(f"       vocab size: {vocab_size}")
    if vocab_size != "?":
        check(
            vocab_size == config.get("vocabulary_size", vocab_size),
            "vocab size matches config "
            f"({vocab_size} == {config.get('vocabulary_size')})",
        )

    # ---- 5. numpy arrays ----
    print("\n[5] Checking numpy arrays ...")
    for key in ("applicability", "success", "refinement_time"):
        arr = ds[key]
        check(
            isinstance(arr, np.ndarray),
            f"{key} is np.ndarray (got {type(arr).__name__})",
        )
        check(arr.ndim == 2, f"{key} is 2-D (shape={arr.shape})")
        check(
            arr.shape[0] == len(seed_ids),
            f"{key} rows == num seeds ({arr.shape[0]} == {len(seed_ids)})",
        )
        if vocab_size != "?":
            check(
                arr.shape[1] == vocab_size,
                f"{key} cols == vocab size ({arr.shape[1]} == {vocab_size})",
            )
        print(
            f"       {key}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={arr.min():.4g}, max={arr.max():.4g}, "
            f"mean={arr.mean():.4g}"
        )

    # ---- 6. list fields ----
    print("\n[6] Checking list fields ...")
    for key in ("initial_low_level_states", "initial_abstract_states", "problem_goals"):
        lst = ds[key]
        check(isinstance(lst, list), f"{key} is a list")
        check(
            len(lst) == len(seed_ids),
            f"{key} length == num seeds ({len(lst)} == {len(seed_ids)})",
        )
        print(f"       {key}: {len(lst)} entries, first type: {type(lst[0]).__name__}")

    # ---- 7. applicability / success consistency ----
    print("\n[7] Checking applicability / success consistency ...")
    app = ds["applicability"]
    suc = ds["success"]
    bad = np.any((suc == 1) & (app == 0))
    check(
        not bad, "No seed×op_seq is success=1 but applicable=0  (success ⊆ applicable)"
    )

    app_rate = app.mean()
    suc_rate = suc.mean()
    print(f"       applicability rate : {app_rate:.2%}")
    print(f"       success rate       : {suc_rate:.2%}")
    check(app_rate > 0, "At least some applicable entries")
    check(suc_rate >= 0, "Success rate is non-negative")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
