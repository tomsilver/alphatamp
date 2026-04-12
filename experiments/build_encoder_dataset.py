"""Build and save one EncoderApproach dataset split per run.

This script:
1) Creates EncoderApproach with fixed hyperparameters,
2) Builds a shared grounded-op vocabulary,
3) Builds one dataset split (optionally in parallel across workers),
4) Saves artifacts as pickles.

Run from repo root, e.g.:
    python experiments/build_encoder_dataset.py

Modes (run.mode):
    "all"     - build vocab then dataset in one shot (default, original behaviour)
    "vocab"   - only build and save the vocabulary, then exit
    "dataset" - skip vocab collection, load vocab from run.vocab_file instead
    "all_filtered" - build uncapped full vocab, then build a small filter-seed
                                     reference dataset (optionally pre-capped to top-k via
                                     vocab.limit_full_vocab_before_filter), then filter offline

Parallelism (run.num_workers):
  1  - sequential, no subprocess overhead (default, original behaviour)
  N  - split seed range into N equal chunks and run each in a separate process
       using concurrent.futures.ProcessPoolExecutor
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from pathlib import Path
from typing import Any

import hydra
import kinder
import numpy as np
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from omegaconf import DictConfig

try:
    import dill
except ImportError as exc:
    raise ImportError(
        "build_encoder_dataset.py requires dill for serializing dataset artifacts. "
        "Install with: pip install dill"
    ) from exc

from alphatamp.approaches.encoder_approach import EncoderApproach
from alphatamp.structs import FrozenGroundOpSequence

# ---------------------------------------------------------------------------
# Module-level dataclass for worker arguments
# (must be at module level for ProcessPoolExecutor pickle compatibility)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _WorkerArgs:
    """All data a worker process needs to build one seed chunk."""

    env_id: str
    model_name: str
    model_kwargs: dict
    max_abstract_plans: int
    samples_per_step: int
    max_skill_horizon: int
    num_training_skeletons_per_problem: int
    training_planning_timeout: float
    vocabulary_size: int
    vocab: list[FrozenGroundOpSequence]
    seed_chunk: list[int]
    worker_idx: int


# ---------------------------------------------------------------------------
# Module-level worker function
# ---------------------------------------------------------------------------


def _dataset_worker(args: _WorkerArgs) -> bytes:
    """Build a dataset for one chunk of seeds inside a worker process.

    This function is intentionally top-level so that ProcessPoolExecutor can pickle it
    without issues.
    """
    print(
        f"[worker {args.worker_idx}] starting "
        f"{len(args.seed_chunk)} seeds: {args.seed_chunk[:3]}..."
    )

    kinder.register_all_environments()
    env = kinder.make(args.env_id)
    try:
        obs, _ = env.reset(seed=0)
        del obs
        env_models = create_bilevel_planning_models(
            args.model_name,
            env.observation_space,
            env.action_space,
            **args.model_kwargs,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]

    approach: EncoderApproach[Any, Any, Any] = EncoderApproach(
        env_models=env_models,
        seed=0,
        max_abstract_plans=args.max_abstract_plans,
        samples_per_step=args.samples_per_step,
        max_skill_horizon=args.max_skill_horizon,
        num_training_skeletons_per_problem=args.num_training_skeletons_per_problem,
        training_planning_timeout=args.training_planning_timeout,
        vocabulary_size=args.vocabulary_size,
        env_id=args.env_id,
    )
    approach.set_vocab(args.vocab)

    partial = approach.build_dataset(args.seed_chunk, show_progress=False)
    print(f"[worker {args.worker_idx}] finished {len(args.seed_chunk)} seeds")
    return dill.dumps(partial)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
    """Persist payload to pickle, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(payload, f)


def _merge_partial_datasets(
    partials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Concatenate per-chunk dataset dicts returned by workers.

    Seeds are kept in the order they appear in *partials*, which callers must ensure
    matches the original seed order (i.e. pass chunks in order).
    """
    if not partials:
        raise ValueError("partials must be non-empty")

    merged_seed_ids: list[int] = []
    merged_low: list[Any] = []
    merged_abs: list[Any] = []
    merged_goals: list[Any] = []
    applicability_parts: list[np.ndarray] = []
    success_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    steps_parts: list[np.ndarray] = []

    for p in partials:
        merged_seed_ids.extend(p["seed_ids"])
        merged_low.extend(p["initial_low_level_states"])
        merged_abs.extend(p["initial_abstract_states"])
        merged_goals.extend(p["problem_goals"])
        applicability_parts.append(p["applicability"])
        success_parts.append(p["success"])
        time_parts.append(p["refinement_time"])
        steps_parts.append(p["steps_completed_fraction"])

    return {
        "seed_ids": merged_seed_ids,
        "op_sequence_vocab": partials[0]["op_sequence_vocab"],
        "applicability": np.concatenate(applicability_parts, axis=0),
        "success": np.concatenate(success_parts, axis=0),
        "refinement_time": np.concatenate(time_parts, axis=0),
        "steps_completed_fraction": np.concatenate(steps_parts, axis=0),
        "skeleton_lengths": partials[0]["skeleton_lengths"],
        "initial_low_level_states": merged_low,
        "initial_abstract_states": merged_abs,
        "problem_goals": merged_goals,
    }


def _build_approach(
    env_id: str,
    model_name: str,
    model_kwargs: dict,
    max_abstract_plans: int,
    samples_per_step: int,
    max_skill_horizon: int,
    num_training_skeletons_per_problem: int,
    training_planning_timeout: float,
    vocabulary_size: int,
) -> EncoderApproach:  # type: ignore[type-arg]
    """Create env models and return a freshly constructed EncoderApproach."""
    kinder.register_all_environments()
    env = kinder.make(env_id)
    try:
        obs, _ = env.reset(seed=0)
        del obs
        env_models = create_bilevel_planning_models(
            model_name,
            env.observation_space,
            env.action_space,
            **model_kwargs,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]

    return EncoderApproach(
        env_models=env_models,
        seed=0,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        max_skill_horizon=max_skill_horizon,
        num_training_skeletons_per_problem=num_training_skeletons_per_problem,
        training_planning_timeout=training_planning_timeout,
        vocabulary_size=vocabulary_size,
        env_id=env_id,
    )


def _build_dataset_parallel(
    vocab: list[FrozenGroundOpSequence],
    split_seeds: list[int],
    num_workers: int,
    env_id: str,
    model_name: str,
    model_kwargs: dict,
    max_abstract_plans: int,
    samples_per_step: int,
    max_skill_horizon: int,
    num_training_skeletons_per_problem: int,
    training_planning_timeout: float,
    vocabulary_size: int,
) -> dict[str, Any]:
    """Partition seeds, dispatch to workers, and merge results."""
    # Ensure any dynamically created model modules are importable in the
    # parent process before we unpickle worker results.
    _parent_approach = _build_approach(
        env_id=env_id,
        model_name=model_name,
        model_kwargs=model_kwargs,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        max_skill_horizon=max_skill_horizon,
        num_training_skeletons_per_problem=num_training_skeletons_per_problem,
        training_planning_timeout=training_planning_timeout,
        vocabulary_size=vocabulary_size,
    )
    del _parent_approach

    # Split seeds into at most num_workers contiguous chunks.
    actual_workers = min(num_workers, len(split_seeds))
    chunks: list[list[int]] = [
        chunk.tolist()
        for chunk in np.array_split(split_seeds, actual_workers)
        if len(chunk) > 0
    ]
    print(
        f"Dispatching {len(split_seeds)} seeds across "
        f"{len(chunks)} workers (requested {num_workers})..."
    )

    worker_args = [
        _WorkerArgs(
            env_id=env_id,
            model_name=model_name,
            model_kwargs=model_kwargs,
            max_abstract_plans=max_abstract_plans,
            samples_per_step=samples_per_step,
            max_skill_horizon=max_skill_horizon,
            num_training_skeletons_per_problem=num_training_skeletons_per_problem,
            training_planning_timeout=training_planning_timeout,
            vocabulary_size=vocabulary_size,
            vocab=vocab,
            seed_chunk=chunk,
            worker_idx=idx,
        )
        for idx, chunk in enumerate(chunks)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        # Submit in order so partials arrive in seed order.
        futures = [executor.submit(_dataset_worker, args) for args in worker_args]
        partials = [dill.loads(f.result()) for f in futures]

    return _merge_partial_datasets(partials)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="conf",
    config_name="build_encoder_dataset_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Build and save one encoder dataset split."""

    env_id = str(cfg.env.id)
    model_name = str(cfg.env.model_name)
    from omegaconf import OmegaConf
    model_kwargs = dict(OmegaConf.to_container(cfg.env.model_kwargs, resolve=True))

    max_abstract_plans = int(cfg.encoder.max_abstract_plans)
    samples_per_step = int(cfg.encoder.samples_per_step)
    max_skill_horizon = int(cfg.encoder.max_skill_horizon)
    num_training_skeletons_per_problem = int(
        cfg.encoder.num_training_skeletons_per_problem
    )
    training_planning_timeout = float(cfg.encoder.training_planning_timeout)
    vocabulary_size = int(cfg.encoder.vocabulary_size)

    split_name = str(cfg.run.split_name)
    split_seed_start = int(cfg.run.seed_start)
    split_seed_stop = int(cfg.run.seed_stop)
    if split_seed_stop <= split_seed_start:
        raise ValueError("run.seed_stop must be greater than run.seed_start")
    split_seeds = list(range(split_seed_start, split_seed_stop))

    mode = str(cfg.run.get("mode", "all"))
    num_workers = int(cfg.run.get("num_workers", 1))
    vocab_file_cfg = cfg.run.get("vocab_file", None)
    vocab_file: Path | None = Path(str(vocab_file_cfg)) if vocab_file_cfg else None

    if mode not in {"all", "vocab", "dataset", "all_filtered"}:
        raise ValueError(
            "run.mode must be 'all', 'vocab', 'dataset', or 'all_filtered'; "
            f"got {mode!r}"
        )
    if num_workers < 1:
        raise ValueError("run.num_workers must be >= 1")
    if mode == "dataset" and vocab_file is None:
        raise ValueError("run.vocab_file must be set when run.mode='dataset'")

    # all_filtered needs filter-seed config entries.
    filter_seed_start: int | None = None
    filter_seed_stop: int | None = None
    filter_threshold: float = 0.0
    filter_min_appl_count: int = 0
    limit_full_vocab_before_filter = False
    if mode == "all_filtered":
        filter_seed_start = int(cfg.vocab.get("filter_seed_start", 500))
        filter_seed_stop = int(cfg.vocab.get("filter_seed_stop", 525))
        filter_threshold = float(cfg.vocab.get("filter_success_rate_threshold", 0.0))
        filter_min_appl_count = int(cfg.vocab.get("filter_min_appl_count", 0))
        limit_full_vocab_before_filter_cfg = cfg.vocab.get(
            "limit_full_vocab_before_filter", False
        )
        if isinstance(limit_full_vocab_before_filter_cfg, bool):
            limit_full_vocab_before_filter = limit_full_vocab_before_filter_cfg
        elif isinstance(limit_full_vocab_before_filter_cfg, str):
            cfg_value = limit_full_vocab_before_filter_cfg.strip().lower()
            if cfg_value in {"1", "true", "yes", "y", "on"}:
                limit_full_vocab_before_filter = True
            elif cfg_value in {"0", "false", "no", "n", "off"}:
                limit_full_vocab_before_filter = False
            else:
                raise ValueError(
                    "vocab.limit_full_vocab_before_filter must be boolean; "
                    f"got {limit_full_vocab_before_filter_cfg!r}"
                )
        else:
            raise ValueError(
                "vocab.limit_full_vocab_before_filter must be boolean; "
                f"got {limit_full_vocab_before_filter_cfg!r}"
            )
        if filter_seed_stop <= filter_seed_start:
            raise ValueError(
                "vocab.filter_seed_stop must be greater than vocab.filter_seed_start"
            )
        if filter_threshold < 0.0 or filter_threshold > 1.0:
            raise ValueError(
                f"vocab.filter_success_rate_threshold must be in [0, 1], "
                f"got {filter_threshold}"
            )
        if filter_min_appl_count < 0:
            raise ValueError(
                f"vocab.filter_min_appl_count must be >= 0, got {filter_min_appl_count}"
            )

    output_dir = Path(str(cfg.output_dir))

    approach_kwargs: dict[str, Any] = {
        "env_id": env_id,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "max_abstract_plans": max_abstract_plans,
        "samples_per_step": samples_per_step,
        "max_skill_horizon": max_skill_horizon,
        "num_training_skeletons_per_problem": num_training_skeletons_per_problem,
        "training_planning_timeout": training_planning_timeout,
        "vocabulary_size": vocabulary_size,
    }
    config_dict: dict[str, Any] = dict(approach_kwargs)

    # ------------------------------------------------------------------
    # Step 1: obtain vocabulary
    # ------------------------------------------------------------------
    vocab: list[FrozenGroundOpSequence]
    # all_filtered tracks two vocabularies:
    # 1) uncapped full vocabulary saved to disk
    # 2) optional capped Stage-B vocabulary used for filter-seed simulation
    full_vocab_uncapped: list[FrozenGroundOpSequence] | None = None
    stage_b_vocab: list[FrozenGroundOpSequence] | None = None

    if mode == "dataset":
        # Load pre-built vocabulary from file.
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, "rb") as f:  # type: ignore[arg-type]
            vocab_payload = dill.load(f)
        vocab = list(vocab_payload["vocabulary"])
        print(f"Loaded vocabulary with {len(vocab)} sequences.")
    else:
        # Build vocabulary from scratch (mode "all", "vocab", or "all_filtered").
        vocab_seed_start = int(cfg.vocab.seed_start)
        vocab_seed_stop = int(cfg.vocab.seed_stop)
        if vocab_seed_stop <= vocab_seed_start:
            raise ValueError("vocab.seed_stop must be greater than vocab.seed_start")
        vocab_seeds = list(range(vocab_seed_start, vocab_seed_stop))

        print("Registering environments and building env models...")
        approach = _build_approach(**approach_kwargs)

        if mode == "all_filtered":
            # Collect counts for ALL observed sequences without any top-k cap.
            # Stage B can optionally run on a top-k slice from _refresh_vocabulary().
            print(
                "Building full vocabulary (uncapped) from seeds "
                f"[{vocab_seed_start}, {vocab_seed_stop})..."
            )
            approach.build_full_vocab(vocab_seeds)
            counts = approach.get_op_sequence_counts()
            # Sort by descending frequency so the order is deterministic.
            full_vocab_uncapped = sorted(counts, key=lambda seq: -counts[seq])
            vocab = list(full_vocab_uncapped)

            if limit_full_vocab_before_filter:
                # Reuse EncoderApproach top-k logic for Stage B capping.
                approach._refresh_vocabulary()  # pylint: disable=protected-access
                stage_b_vocab = approach.get_op_sequence_vocabulary()
                print(
                    "[all_filtered] Stage B pre-cap enabled: "
                    f"full_uncapped={len(full_vocab_uncapped)} → "
                    f"stage_b={len(stage_b_vocab)} "
                    f"(encoder.vocabulary_size={vocabulary_size})"
                )
            else:
                stage_b_vocab = list(full_vocab_uncapped)
                print(
                    "[all_filtered] Stage B pre-cap disabled: "
                    f"using full vocabulary ({len(stage_b_vocab)} sequences)"
                )
            vocab_out_name = f"encoder_vocab_full_{split_name}.pkl"
        else:
            print(
                "Building shared vocabulary from seeds "
                f"[{vocab_seed_start}, {vocab_seed_stop})..."
            )
            vocab = approach.build_vocab(vocab_seeds, vocabulary_size)
            vocab_out_name = f"encoder_vocab_{split_name}.pkl"

        vocab_out_path = output_dir / vocab_out_name
        _save_pickle(
            vocab_out_path,
            {
                "vocabulary": vocab,
                "config": config_dict,
                "vocab_seed_ids": vocab_seeds,
                "split": split_name,
            },
        )
        print(f"Saved vocabulary ({len(vocab)} sequences) to: {vocab_out_path}")

        if mode == "vocab":
            print("Done (vocab only).")
            return

    # ------------------------------------------------------------------
    # Step 2: build dataset
    # ------------------------------------------------------------------

    # all_filtered stops here: run Stages B + C then exit.
    if mode == "all_filtered":
        assert filter_seed_start is not None and filter_seed_stop is not None
        assert stage_b_vocab is not None
        filter_seeds = list(range(filter_seed_start, filter_seed_stop))
        stage_b_vocab_for_filter = list(stage_b_vocab)
        full_vocab_size_uncapped = (
            len(full_vocab_uncapped)
            if full_vocab_uncapped is not None
            else len(stage_b_vocab_for_filter)
        )
        stage_b_vocab_size = len(stage_b_vocab_for_filter)
        pre_stage_b_cap_applied = (
            limit_full_vocab_before_filter
            and stage_b_vocab_size < full_vocab_size_uncapped
        )

        # ---- Stage B: simulator run on filter seeds ----
        print(
            f"[all_filtered] Stage B: building filter-seed reference dataset "
            f"({len(filter_seeds)} seeds × {stage_b_vocab_size} vocab sequences)..."
        )
        print(
            f"[all_filtered] Stage B vocab sizes: uncapped_full={full_vocab_size_uncapped}, "
            f"stage_b={stage_b_vocab_size}, pre_cap_applied={pre_stage_b_cap_applied}"
        )
        print(
            f"[all_filtered] Stage B workers: {num_workers} "
            f"(run.num_workers={num_workers})"
        )
        if num_workers == 1:
            approach = _build_approach(**approach_kwargs)
            approach.set_vocab(stage_b_vocab_for_filter)
            filter_dataset = approach.build_dataset(filter_seeds)
        else:
            filter_dataset = _build_dataset_parallel(
                vocab=stage_b_vocab_for_filter,
                split_seeds=filter_seeds,
                num_workers=num_workers,
                **approach_kwargs,
            )

        filter_dataset_path = output_dir / "encoder_filter_dataset.pkl"
        _save_pickle(
            filter_dataset_path,
            {
                "split": "filter",
                "seed_ids": filter_seeds,
                "config": config_dict,
                "filter_seed_start": filter_seed_start,
                "filter_seed_stop": filter_seed_stop,
                "limit_full_vocab_before_filter": limit_full_vocab_before_filter,
                "pre_stage_b_cap_applied": pre_stage_b_cap_applied,
                "full_vocab_size_uncapped": full_vocab_size_uncapped,
                "stage_b_vocab_size": stage_b_vocab_size,
                "vocabulary_size_setting": vocabulary_size,
                "dataset": filter_dataset,
            },
        )
        print(
            "[all_filtered] Saved filter-seed reference dataset to: "
            f"{filter_dataset_path}"
        )

        # ---- Stage C: offline column-filtering (no simulator) ----
        print(
            f"[all_filtered] Stage C: filtering vocabulary offline "
            f"(threshold={filter_threshold})..."
        )
        filtered_vocab, keep_indices, stats = (
            EncoderApproach.filter_vocab_by_success_rate(
                filter_dataset, filter_threshold, min_appl_count=filter_min_appl_count
            )
        )
        filtered_dataset = EncoderApproach.apply_vocab_filter_to_dataset(
            filter_dataset, keep_indices
        )

        print(
            f"[all_filtered] Vocabulary: {stats['original_size']} → "
            f"{stats['filtered_size']} sequences "
            f"({stats['removed_count']} removed at threshold={filter_threshold})"
        )

        # Filtered vocab pickle — pass this to run.vocab_file for mode=dataset.
        filtered_vocab_path = output_dir / f"encoder_vocab_filtered_{split_name}.pkl"
        _save_pickle(
            filtered_vocab_path,
            {
                "vocabulary": filtered_vocab,
                "vocabulary_full": (
                    list(full_vocab_uncapped)
                    if full_vocab_uncapped is not None
                    else list(stage_b_vocab_for_filter)
                ),
                "vocabulary_stage_b": stage_b_vocab_for_filter,
                "keep_indices": keep_indices,
                "keep_indices_reference_vocab": "vocabulary_stage_b",
                "filter_seed_ids": filter_seeds,
                "filter_seed_start": filter_seed_start,
                "filter_seed_stop": filter_seed_stop,
                "filter_success_rate_threshold": filter_threshold,
                "limit_full_vocab_before_filter": limit_full_vocab_before_filter,
                "pre_stage_b_cap_applied": pre_stage_b_cap_applied,
                "full_vocab_size_uncapped": full_vocab_size_uncapped,
                "stage_b_vocab_size": stage_b_vocab_size,
                "vocabulary_size_setting": vocabulary_size,
                "filter_stats": stats,
                "config": config_dict,
                "split": split_name,
            },
        )
        print(f"[all_filtered] Saved filtered vocabulary to: {filtered_vocab_path}")

        # Derived (column-sliced) filter dataset — for analysis/reference.
        filtered_filter_dataset_path = (
            output_dir / "encoder_filter_dataset_filtered.pkl"
        )
        _save_pickle(
            filtered_filter_dataset_path,
            {
                "split": "filter_filtered",
                "seed_ids": filter_seeds,
                "config": config_dict,
                "filter_seed_start": filter_seed_start,
                "filter_seed_stop": filter_seed_stop,
                "filter_success_rate_threshold": filter_threshold,
                "limit_full_vocab_before_filter": limit_full_vocab_before_filter,
                "pre_stage_b_cap_applied": pre_stage_b_cap_applied,
                "full_vocab_size_uncapped": full_vocab_size_uncapped,
                "stage_b_vocab_size": stage_b_vocab_size,
                "vocabulary_size_setting": vocabulary_size,
                "filter_stats": stats,
                "dataset": filtered_dataset,
            },
        )
        print(
            f"[all_filtered] Saved filtered filter-seed dataset to: "
            f"{filtered_filter_dataset_path}"
        )
        print(
            "[all_filtered] Done. Next steps:\n"
            f"  1. Examine {filter_dataset_path} (use analyze_encoder_dataset.ipynb)\n"
            "  2. Optionally re-filter offline with a different threshold by calling\n"
            "     EncoderApproach.filter_vocab_by_success_rate / "
            "apply_vocab_filter_to_dataset\n"
            f"  3. Run mode=dataset with run.vocab_file={filtered_vocab_path}\n"
            "     to build the full train/val dataset."
        )
        return
    print(
        f"Building {split_name} dataset for {len(split_seeds)} seeds "
        f"[{split_seed_start}, {split_seed_stop}) "
        f"with {num_workers} worker(s)..."
    )

    if num_workers == 1:
        # Sequential path — no subprocess overhead, identical to original behaviour.
        print("Registering environments and building env models...")
        approach = _build_approach(**approach_kwargs)
        approach.set_vocab(vocab)
        dataset = approach.build_dataset(split_seeds)
    else:
        dataset = _build_dataset_parallel(
            vocab=vocab,
            split_seeds=split_seeds,
            num_workers=num_workers,
            **approach_kwargs,
        )

    payload: dict[str, Any] = {
        "split": split_name,
        "seed_ids": split_seeds,
        "config": config_dict,
        "dataset": dataset,
    }

    out_path = output_dir / f"encoder_{split_name}_dataset.pkl"
    _save_pickle(out_path, payload)
    print(f"Saved {split_name} artifact to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
