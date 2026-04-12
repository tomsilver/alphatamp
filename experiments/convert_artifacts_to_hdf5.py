"""Convert dill-pickled encoder dataset artifacts to self-contained HDF5 files.

The bootstrap (kinder env + bilevel_planning model registration) MUST happen
before dill.load — the pickled GroundOperator objects reference dynamically-
generated modules (e.g. obstruction2d_module) that don't exist until
create_bilevel_planning_models is called.

Usage
-----
    # Convert a single difficulty:
    python experiments/convert_artifacts_to_hdf5.py o3

    # Convert multiple difficulties:
    python experiments/convert_artifacts_to_hdf5.py o2 o3 o4

    # Default (no args): convert o2 o3 o4 for backward compatibility
    python experiments/convert_artifacts_to_hdf5.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dill
import kinder
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.data.skeleton_dataset import write_skeleton_dataset

# ---------------------------------------------------------------------------
# Environment parameters per difficulty (mirrors conf/encoder_dataset_difficulty/)
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIGS: dict[str, dict] = {
    "o2": {
        "env_id": "kinder/Obstruction2D-o2-v0",
        "model_name": "obstruction2d",
        "model_kwargs": {"num_obstructions": 2},
    },
    "o3": {
        "env_id": "kinder/Obstruction2D-o3-v0",
        "model_name": "obstruction2d",
        "model_kwargs": {"num_obstructions": 3},
    },
    "o4": {
        "env_id": "kinder/Obstruction2D-o4-v0",
        "model_name": "obstruction2d",
        "model_kwargs": {"num_obstructions": 4},
    },
    "sb1": {
        "env_id": "kinder/StickButton2D-b1-v0",
        "model_name": "stickbutton2d",
        "model_kwargs": {"num_buttons": 1},
    },
    "sb2": {
        "env_id": "kinder/StickButton2D-b2-v0",
        "model_name": "stickbutton2d",
        "model_kwargs": {"num_buttons": 2},
    },
    "sb3": {
        "env_id": "kinder/StickButton2D-b3-v0",
        "model_name": "stickbutton2d",
        "model_kwargs": {"num_buttons": 3},
    },
    "sb5": {
        "env_id": "kinder/StickButton2D-b5-v0",
        "model_name": "stickbutton2d",
        "model_kwargs": {"num_buttons": 5},
    },
    "sb10": {
        "env_id": "kinder/StickButton2D-b10-v0",
        "model_name": "stickbutton2d",
        "model_kwargs": {"num_buttons": 10},
    },
}

VALID_DIFFICULTIES = sorted(DIFFICULTY_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_bootstrapped: set[str] = set()


def bootstrap(difficulty: str) -> None:
    """Register dynamic modules for a given difficulty level.

    Must be called BEFORE dill.load — the pickled objects reference modules
    that only exist after create_bilevel_planning_models runs.
    Skips if already called for this difficulty (idempotent).
    """
    if difficulty in _bootstrapped:
        return
    cfg = DIFFICULTY_CONFIGS[difficulty]
    kinder.register_all_environments()
    env = kinder.make(cfg["env_id"])
    try:
        create_bilevel_planning_models(
            cfg["model_name"],
            env.observation_space,
            env.action_space,
            **cfg["model_kwargs"],
        )
    finally:
        env.close()
    _bootstrapped.add(difficulty)
    print(f"  [bootstrap] registered modules for {difficulty} ({cfg['env_id']})")


def artifact_dir(difficulty: str) -> Path:
    """Map difficulty name to its pickle artifact root directory."""
    if difficulty.startswith("o"):
        return Path(f"artifacts_ob/encoder_{difficulty}")
    if difficulty.startswith("sb"):
        return Path(f"artifacts_sb/encoder_{difficulty}")
    return Path(f"artifacts/encoder_{difficulty}")


HDF5_ROOT = Path("artifacts_hdf5")


def convert_difficulty(difficulty: str) -> None:
    """Convert all three splits for a single difficulty."""
    pkl_root = artifact_dir(difficulty)
    hdf5_dir = HDF5_ROOT / f"encoder_{difficulty}"

    for split in ["train", "validation", "test"]:
        pkl_path = pkl_root / f"encoder_{split}_filtered_dataset.pkl"
        hdf5_path = hdf5_dir / f"{split}.h5"

        if not pkl_path.exists():
            print(f"  SKIP {pkl_path} (not found)")
            continue

        print(f"Converting {pkl_path} ...")

        # Bootstrap BEFORE dill.load — modules must exist before unpickling
        bootstrap(difficulty)

        with open(pkl_path, "rb") as f:
            payload = dill.load(f)

        # The actual dataset dict is nested under "dataset"
        dataset_dict = payload["dataset"]

        # Zero out refinement_time for inapplicable entries in case older
        # artifacts stored the timeout value there instead of 0.
        inapplicable = dataset_dict["applicability"] < 0.5
        dataset_dict["refinement_time"][inapplicable] = 0.0

        write_skeleton_dataset(
            hdf5_path,
            dataset_dict,
            source_description=f"{difficulty} {split} filtered",
        )
        print(
            f"  -> {hdf5_path}  "
            f"(N={len(dataset_dict['seed_ids'])}, "
            f"M={len(dataset_dict['op_sequence_vocab'])})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dill-pickled encoder artifacts to HDF5.",
    )
    parser.add_argument(
        "difficulties",
        nargs="*",
        default=["o2", "o3", "o4"],
        choices=VALID_DIFFICULTIES,
        metavar="DIFFICULTY",
        help=(
            f"Difficulty name(s) to convert. "
            f"Valid: {', '.join(VALID_DIFFICULTIES)}. "
            f"Default: o2 o3 o4."
        ),
    )
    args = parser.parse_args()

    for difficulty in args.difficulties:
        convert_difficulty(difficulty)

    print("Done.")


if __name__ == "__main__":
    main()
