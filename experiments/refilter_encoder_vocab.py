"""Re-run encoder vocabulary filtering from an existing filter dataset artifact.

This utility is offline-only: it does not run planning or simulation.
It loads an existing ``encoder_filter_dataset.pkl`` artifact, applies the current
filtering logic, and writes updated filtered outputs.

Typical usage:
    python experiments/refilter_encoder_vocab.py \
      --filter-artifact artifacts/encoder_o2/encoder_filter_dataset.pkl \
      --output-dir artifacts/encoder_o2 \
      --threshold 0.0 \
      --split-name all_filtered
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import dill
import kinder
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.encoder_approach import EncoderApproach


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as file:
        payload = dill.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def _save_pickle(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        dill.dump(payload, file)


def _bootstrap_dill_modules() -> None:
    """Register dynamic planning modules required by dill deserialization."""
    kinder.register_all_environments()

    # Encoder filter artifacts in this workflow are generated from Obstruction2D
    # difficulties, so pre-create these model modules before dill.load().
    bootstrap_envs = [
        ("kinder/Obstruction2D-o0-v0", 0),
        ("kinder/Obstruction2D-o1-v0", 1),
        ("kinder/Obstruction2D-o2-v0", 2),
        ("kinder/Obstruction2D-o3-v0", 3),
        ("kinder/Obstruction2D-o4-v0", 4),
    ]

    bootstrapped: list[str] = []
    for env_id, num_obstructions in bootstrap_envs:
        env = kinder.make(env_id)
        try:
            _ = create_bilevel_planning_models(
                "obstruction2d",
                env.observation_space,
                env.action_space,
                num_obstructions=num_obstructions,
            )
            bootstrapped.append(env_id)
        finally:
            env.close()  # type: ignore[no-untyped-call]

    print("Bootstrapped model modules for dill:")
    for env_id in bootstrapped:
        print(f" - {env_id}")


def main() -> None:
    """Main entry point for refiltering encoder vocab from an existing filter dataset
    artifact."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-run offline encoder vocab filtering from an existing "
            "encoder_filter_dataset.pkl artifact."
        )
    )
    parser.add_argument(
        "--filter-artifact",
        type=Path,
        required=True,
        help="Path to encoder_filter_dataset.pkl (payload with key 'dataset').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where filtered vocab/dataset artifacts are written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Success-rate threshold used by filter_vocab_by_success_rate.",
    )
    parser.add_argument(
        "--min-appl-count",
        type=int,
        default=0,
        help=(
            "Minimum number of filter-seed rows in which a skeleton must be applicable "
            "to be eligible for retention. Skeletons with 0 < applicable_count < this "
            "value are removed (insufficient data). Default 0 keeps original behaviour."
        ),
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="all_filtered",
        help="Split suffix used for filtered vocab filename.",
    )
    parser.add_argument(
        "--filtered-dataset-name",
        type=str,
        default="encoder_filter_dataset_filtered.pkl",
        help=(
            "Filename for the filtered version of --filter-artifact within --output-dir. "
            "Defaults to 'encoder_filter_dataset_filtered.pkl' (original behaviour). "
            "Pass 'encoder_train_filtered_dataset.pkl' when filtering the training split."
        ),
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the validation dataset pkl. When provided, the same "
            "keep_indices derived from --filter-artifact are applied and the result "
            "is saved as encoder_validation_filtered_dataset.pkl in --output-dir."
        ),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the test dataset pkl. When provided, the same "
            "keep_indices derived from --filter-artifact are applied and the result "
            "is saved as encoder_test_filtered_dataset.pkl in --output-dir."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip dynamic module bootstrap before loading dill artifacts.",
    )

    args = parser.parse_args()

    filter_artifact = args.filter_artifact.resolve()
    output_dir = args.output_dir.resolve()

    if not filter_artifact.exists():
        raise FileNotFoundError(f"Filter artifact not found: {filter_artifact}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")
    if args.min_appl_count < 0:
        raise ValueError(f"--min-appl-count must be >= 0, got {args.min_appl_count}")

    if not args.skip_bootstrap:
        _bootstrap_dill_modules()

    payload = _load_pickle(filter_artifact)
    if "dataset" not in payload:
        raise KeyError(f"Expected key 'dataset' in filter artifact: {filter_artifact}")
    dataset = payload["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected payload['dataset'] to be dict, got {type(dataset)}")

    required_dataset_keys = {
        "seed_ids",
        "op_sequence_vocab",
        "applicability",
        "success",
        "refinement_time",
        "steps_completed_fraction",
        "skeleton_lengths",
    }
    missing = required_dataset_keys - set(dataset)
    if missing:
        raise KeyError(f"Dataset missing required keys: {sorted(missing)}")

    filtered_vocab, keep_indices, stats = EncoderApproach.filter_vocab_by_success_rate(
        dataset,
        args.threshold,
        min_appl_count=args.min_appl_count,
    )
    filtered_dataset = EncoderApproach.apply_vocab_filter_to_dataset(
        dataset,
        keep_indices,
    )

    filtered_vocab_path = output_dir / f"encoder_vocab_filtered_{args.split_name}.pkl"
    filtered_dataset_path = output_dir / args.filtered_dataset_name

    vocab_payload: dict[str, Any] = {
        "vocabulary": filtered_vocab,
        "vocabulary_full": list(dataset["op_sequence_vocab"]),
        "keep_indices": keep_indices,
        "filter_success_rate_threshold": args.threshold,
        "filter_min_appl_count": args.min_appl_count,
        "filter_stats": stats,
        "split": args.split_name,
    }
    for key in (
        "filter_seed_ids",
        "filter_seed_start",
        "filter_seed_stop",
        "config",
    ):
        if key in payload:
            vocab_payload[key] = payload[key]

    filtered_dataset_payload: dict[str, Any] = {
        "split": f"{args.split_name}_filtered",
        "dataset": filtered_dataset,
        "filter_success_rate_threshold": args.threshold,
        "filter_min_appl_count": args.min_appl_count,
        "filter_stats": stats,
    }
    for key in (
        "seed_ids",
        "filter_seed_start",
        "filter_seed_stop",
        "config",
    ):
        if key in payload:
            filtered_dataset_payload[key] = payload[key]

    _save_pickle(filtered_vocab_path, vocab_payload, overwrite=args.overwrite)
    _save_pickle(filtered_dataset_path, filtered_dataset_payload, overwrite=args.overwrite)

    print(f"Loaded filter artifact: {filter_artifact}")
    print(f"Threshold: {args.threshold}  min_appl_count: {args.min_appl_count}")
    print(
        "Vocabulary: "
        f"{stats['original_size']} -> {stats['filtered_size']} "
        f"(removed={stats['removed_count']}, "
        f"insufficient_data={stats['insufficient_data_count']}, "
        f"never_applicable_kept={stats['never_applicable_kept_count']})"
    )
    print(f"Saved filtered vocab:    {filtered_vocab_path}")
    print(f"Saved filtered dataset:  {filtered_dataset_path}")

    # ── Propagate vocab filter to val / test splits ──────────────────────────
    for split_path_arg, split_label, out_filename in [
        (args.val_path, "validation", "encoder_validation_filtered_dataset.pkl"),
        (args.test_path, "test", "encoder_test_filtered_dataset.pkl"),
    ]:
        if split_path_arg is None:
            continue
        split_path = split_path_arg.resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"{split_label} artifact not found: {split_path}")
        split_payload = _load_pickle(split_path)
        if "dataset" not in split_payload:
            raise KeyError(f"Expected key 'dataset' in {split_label} artifact: {split_path}")
        split_filtered = EncoderApproach.apply_vocab_filter_to_dataset(
            split_payload["dataset"], keep_indices
        )
        split_out_payload: dict[str, Any] = {
            "split": f"{split_label}_filtered",
            "dataset": split_filtered,
            "filter_success_rate_threshold": args.threshold,
            "filter_min_appl_count": args.min_appl_count,
            "filter_stats": stats,
        }
        for key in ("seed_ids", "config"):
            if key in split_payload:
                split_out_payload[key] = split_payload[key]
        out_path = output_dir / out_filename
        _save_pickle(out_path, split_out_payload, overwrite=args.overwrite)
        print(f"Saved filtered {split_label} dataset: {out_path}")


if __name__ == "__main__":
    main()
