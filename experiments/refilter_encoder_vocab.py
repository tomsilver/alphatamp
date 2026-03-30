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


def main() -> None:
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
        "--split-name",
        type=str,
        default="all_filtered",
        help="Split suffix used for filtered vocab filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )

    args = parser.parse_args()

    filter_artifact = args.filter_artifact.resolve()
    output_dir = args.output_dir.resolve()

    if not filter_artifact.exists():
        raise FileNotFoundError(f"Filter artifact not found: {filter_artifact}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")

    payload = _load_pickle(filter_artifact)
    if "dataset" not in payload:
        raise KeyError(
            f"Expected key 'dataset' in filter artifact: {filter_artifact}"
        )
    dataset = payload["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError(
            f"Expected payload['dataset'] to be dict, got {type(dataset)}"
        )

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
    )
    filtered_dataset = EncoderApproach.apply_vocab_filter_to_dataset(
        dataset,
        keep_indices,
    )

    filtered_vocab_path = output_dir / f"encoder_vocab_filtered_{args.split_name}.pkl"
    filtered_filter_dataset_path = output_dir / "encoder_filter_dataset_filtered.pkl"

    vocab_payload: dict[str, Any] = {
        "vocabulary": filtered_vocab,
        "vocabulary_full": list(dataset["op_sequence_vocab"]),
        "keep_indices": keep_indices,
        "filter_success_rate_threshold": args.threshold,
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
        "split": "filter_filtered",
        "dataset": filtered_dataset,
        "filter_success_rate_threshold": args.threshold,
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
    _save_pickle(
        filtered_filter_dataset_path,
        filtered_dataset_payload,
        overwrite=args.overwrite,
    )

    print(f"Loaded filter artifact: {filter_artifact}")
    print(f"Threshold: {args.threshold}")
    print(
        "Vocabulary: "
        f"{stats['original_size']} -> {stats['filtered_size']} "
        f"(removed={stats['removed_count']})"
    )
    print(f"Saved filtered vocab: {filtered_vocab_path}")
    print(f"Saved filtered filter-dataset: {filtered_filter_dataset_path}")


if __name__ == "__main__":
    main()
